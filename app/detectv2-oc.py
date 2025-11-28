import argparse
import cv2
import os
import threading
from flask import Flask, Response
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)
rtsp_address = os.getenv("RTSP_ADDRESS", "rtsp://192.168.0.159:30000/unicast")

# Global variable to hold the latest frame
output_frame = None
lock = threading.Lock()

def generate_frames():
    global output_frame, lock
    
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--edgetpu', action='store_true')
    args, _ = parser.parse_known_args()

    print(f'Loading {args.model} with {args.labels}')
    interpreter = make_interpreter(args.model) if args.edgetpu else Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(rtsp_address)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        
        if args.edgetpu:
            run_inference(interpreter, cv2_im_rgb.tobytes())
        else:
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], [cv2_im_rgb])
            interpreter.invoke()

        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        frame = append_objs_to_img(frame, inference_size, objs, labels)
        
        with lock:
            output_frame = frame.copy()
    
    cap.release()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, _ = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = f'{percent}% {labels.get(obj.id, obj.id)}'
        
        cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(cv2_im, label, (x0, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

def video_stream():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            _, encoded_image = cv2.imencode('.jpg', output_frame)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=generate_frames, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
