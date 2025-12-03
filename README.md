# **RL-Task-Offloading**

## **Reinforcement Learning for Dynamic Edge Task Offloading**

This repository implements a Reinforcement Learning (RL) algorithm to solve the dynamic task offloading problem in Mobile Edge Computing (MEC) environments. The goal is to find optimal offloading decisions (local execution, edge execution, or cloud execution) for time-sensitive mobile tasks, minimizing total latency, energy consumption, or a combination of both.

The project utilizes Python for the core RL implementation and leverages containerization technologies (Kubernetes/Helm) for simulating or deploying the complex MEC environment.

## **✨ Features**

* **RL Algorithms:** Implementation of Q-learning algorithm tailored for discrete or continuous offloading decisions.  
* **MEC Environment Emulation:** A custom experimental environment to emulate user devices, edge servers, communication channels, and task queues.  
* **Custom Datasets:** Includes synthetic or real-world datasets (dataset1, dataset2) representing dynamic task arrival rates and varying channel conditions.  
* **Scalable Deployment:** Kubernetes (k8s) and Helm charts for containerizing and deploying the simulation environment or the trained model inference engine.  
* **Performance Metrics:** Tracks key performance indicators (KPIs) such as average task latency and long-term reward maximization.

## **🚀 Repository Structure**

| Directory | Description |
| :---- | :---- |
| app/ | Core RL training scripts, environment definition, and testing utilities. |
| model/ | Stores the trained weights and configurations for DRL agents. |
| dataset1/ | First set of simulation data. |
| dataset2/ | Second set of experimental data. |
| demo/ | Scripts for running quick demonstrations or inference tests with pre-trained models and demo video. |
| figures/ | Stores result plots (e.g., reward convergence, latency vs. epoch) and architectural diagrams. |
| k8s/ | Kubernetes deployment manifests for the simulation/application components. |
| helm/ | Helm charts for managing the Kubernetes deployment easily. |

## **🛠️ Installation**

### **Prerequisites**

* Python 3.8+  
* Pip package manager  
* (Optional, for deployment) Docker and a running Kubernetes cluster (e.g., Minikube)

### **Python Environment Setup**

1. **Clone the repository:**  
   git clone \[https://github.com/Siong23/RL-task-offloading.git\](https://github.com/Siong23/RL-task-offloading.git)  
   cd RL-task-offloading

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use: .\\venv\\Scripts\\activate

3. Install dependencies:  
   (Note: Add necessary packages like tensorflow, pytorch, gym, numpy, matplotlib based on your project's requirements.txt or known usage.)  
   pip install numpy pandas matplotlib  \# Base requirements  
   \# Assuming PyTorch is used for RL  
   pip install torch torchvision torchaudio  
   \# Add any specific RL framework like stable-baselines3 if applicable

## **🏃 Usage and Training**

### **1\. Preparing the Environment**

Ensure your datasets are placed in the dataset1/ and dataset2/ directories. These datasets define the task parameters and system dynamics.

### **2\. Training the RL Agent**

Run the main training script from the app/ directory.

\# Example training command (adjust script name and arguments as necessary)  
python app/train\_agent.py \--algorithm dqn \--epochs 500 \--dataset dataset1

The trained model weights will be saved to the model/ directory.

### **3\. Running a Demonstration**

Use the demo scripts to visualize the performance of a trained agent or a baseline policy.

\# Run a quick demonstration using the latest trained model  
python demo/run\_simulation.py \--model\_path model/latest\_dqn.pth

## **☁️ Deployment (Kubernetes/Helm)**

For simulating the MEC cluster environment or deploying the final decision engine, the repository includes containerization and orchestration files.

1. **Build Docker Images:**  
   \# Ensure Dockerfile is available and run build command  
   docker build \-t rl-offloader:latest .

2. **Deploy with Helm:**  
   \# Deploy the application using the provided Helm chart  
   helm install rl-offloader ./helm/rl-offloader-chart/

## **📊 Results and Analysis**

All generated plots and convergence curves are stored in the figures/ directory. Key performance metrics (latency, energy, reward over time) are analyzed here, demonstrating the effectiveness of the RL approach compared to traditional heuristic policies.

## **🤝 Contributing**

We welcome contributions\! Please feel free to open issues or submit pull requests.

## **👥 Contributors**

This project is maintained by:

* [@Siong23](https://github.com/Siong23)  
* [@yqx1412](https://github.com/yqx1412)
