import subprocess
import random
import time
import os
import signal

# ========================
# Configuration
# ========================

TIME_SLOT = 30        # seconds per slot
TOTAL_DURATION = 300  # total 5 minutes

# Load levels configuration
LOAD_LEVELS = {
    "low": {
        "cpu": 1,
        "vm": 1,
        "vm-bytes": "128M",
        "hdd": 1,
        "hdd-bytes": "256M"
    },
    "moderate": {
        "cpu": 2,
        "vm": 2,
        "vm-bytes": "512M",
        "hdd": 2,
        "hdd-bytes": "512M"
    },
    "high": {
        "cpu": 3,
        "vm": 4,
        "vm-bytes": "1G",
        "hdd": 4,
        "hdd-bytes": "1G"
    }
}

# ========================
# Utility Functions
# ========================

def run_background(cmd):
    """Start a process in the background and return the handle."""
    print(f"  ▶ Launching: {cmd}")
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

def stop_process(proc):
    """Gracefully terminate a background process."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass

def random_load_level():
    """Randomly select low, moderate, or high load."""
    return random.choice(list(LOAD_LEVELS.keys()))

# ========================
# Main stress routine
# ========================

def apply_stress_for_slot(slot_num):
    level = random_load_level()
    config = LOAD_LEVELS[level]

    print(f"\n[Slot {slot_num}] Applying {level.upper()} stress for {TIME_SLOT}s")

    # --- Start stress-ng for CPU, Memory, and Disk ---
    stress_cmd = (
        f"stress-ng --cpu {config['cpu']} "
        f"--vm {config['vm']} --vm-bytes {config['vm-bytes']} "
        f"--hdd {config['hdd']} --hdd-bytes {config['hdd-bytes']} "
        f"--timeout {TIME_SLOT}s --metrics-brief"
    )
    stress_proc = run_background(stress_cmd)

    # --- Wait for the time slot duration ---
    time.sleep(TIME_SLOT)

    # --- Stop stress-ng process ---
    stop_process(stress_proc)

    print(f"[Slot {slot_num}] Completed {level.upper()} stress cycle.")

# ========================
# Entry point
# ========================

if __name__ == "__main__":
    print("=== Random Stress Orchestrator (CPU, Memory, Disk only) ===")
    print("Total Duration: 5 minutes (10 slots × 30s)\n")

    total_slots = TOTAL_DURATION // TIME_SLOT

    for slot in range(1, total_slots + 1):
        apply_stress_for_slot(slot)
        # Optional small pause between slots to avoid overlap
        time.sleep(5)

    print("\n✅ All stress cycles completed.")
