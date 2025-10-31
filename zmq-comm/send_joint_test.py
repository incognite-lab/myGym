import zmq, json, time, argparse, os, re, shlex, subprocess, sys
import numpy as np

def run_test_and_get_npy(test_script: str, test_args: str = "", cwd: str = None) -> str:
    """
    execute test.py
      "trajectory -> <.../joint_trajectory.npy>"
    Return path of .npy file.
    """
    cmd = [sys.executable, test_script] + (shlex.split(test_args) if test_args else [])
    print("[INFO] Running:", " ".join(shlex.quote(c) for c in cmd))
    if cwd:
        print("[INFO] Working directory:", cwd)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd or None,
        bufsize=1,
        universal_newlines=True,
    )

    last_path = None
    pattern = re.compile(r"trajectory ->\s*(.+joint_trajectory\.npy)")
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")  # passthrough
        m = pattern.search(line)
        if m:
            candidate = m.group(1).strip()
            last_path = candidate

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"[ERROR] test.py exited with non-zero code: {rc}")

    if not last_path:
        raise FileNotFoundError(
            "[ERROR] Could not find 'trajectory -> <...joint_trajectory.npy>' line in test.py output."
        )

    # Make absolute path
    if not os.path.isabs(last_path):
        base = cwd if cwd else os.getcwd()
        last_path = os.path.abspath(os.path.join(base, last_path))

    if not os.path.exists(last_path):
        raise FileNotFoundError(f"[ERROR] joint_trajectory.npy not found at: {last_path}")

    print("[INFO] Detected joint_trajectory.npy:", last_path)
    return last_path

def load_traj(npy_path: str) -> np.ndarray:
    """
    Load joint_trajectory.npy
    """
    traj = np.load(npy_path)
    if traj.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {traj.shape}")

    if traj.shape[1] < 7:
        raise ValueError(f"Trajectory has only {traj.shape[1]} columns; need at least 7.")
    traj = traj[:, :7]
    return traj

def send_over_zmq(traj: np.ndarray, zmq_addr: str) -> dict:
    """
    Send trajectory over ZMQ
    """
    data = traj.tolist()
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.connect(zmq_addr)

    msg = {
        "cmd": "send",
        "joint_names": [f"arm_right_{i}_joint" for i in range(1, 8)],
        "data": data,
    }

    s.send(json.dumps(msg).encode("utf-8"))
    resp = json.loads(s.recv())
    return resp
  
def main():
    parser = argparse.ArgumentParser(description="Run test.py (optional), load joint_trajectory.npy, and send via ZMQ.")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--npy", help="Path to an existing joint_trajectory.npy")
    g.add_argument("--test-script",default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../myGym/test.py")), help="Path to test.py (to run and capture the produced joint_trajectory.npy)")
    #absolute path
    parser.add_argument("--test-args", default="", help="Arguments passed to test.py (as a single quoted string).")
    parser.add_argument("--cwd", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../myGym/")), help="Working directory to run test.py in.")
    parser.add_argument("--zmq", default="tcp://0.0.0.0:242424", help="ZMQ endpoint, e.g., tcp://0.0.0.0:242424")

    args = parser.parse_args()

    if args.test_script:
        npy_path = run_test_and_get_npy(args.test_script, args.test_args, args.cwd)
    else:
        npy_path = args.npy
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"[ERROR] npy not found: {npy_path}") 

    traj = load_traj(npy_path)
    print("[INFO] Loaded trajectory with shape:", traj.shape)

    resp = send_over_zmq(traj, args.zmq)
    print("resp:", resp)


if __name__ == "__main__":
    main()
