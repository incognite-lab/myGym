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

def make_roundtrip(traj: np.ndarray) -> np.ndarray:
    """
    reach -> pause in final position -> reverse
    """
    if traj.ndim != 2 or traj.shape[1] != 7:
        raise ValueError("traj must be [T,7]")
    
    n_pause = 30
    last = traj[-1][None,:]
    pause_block = np.repeat(last, n_pause, axis=0)
    reverse = traj[::-1]

    return np.vstack([traj, pause_block, reverse])

def parse_vec3(s: str):
    s = s.replace(","," ")
    vals = [float(t) for t in s.split() if t.strip()]
    if len(vals) != 3:
        raise ValueError("target expects 3 values, got: {s}")
    return np.asarray(vals, dtype = float)

def read_targets_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Target file not found: {path}\n"
        )
    arr = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            arr.append(line)
    if not arr:
        raise ValueError("No valid targets in file: {path}")
    return arr

def strip_existing_target(args_str: str) -> str:
    pattern = r"(?:^|\s)--target\s+([^\s]+\s[,\s]\s[^\s]+\s[,\s]\s[^\s]+)"
    return re.sub(pattern, " ", args_str or "").strip()

def main():
    parser = argparse.ArgumentParser(description="Run test.py (optional), load joint_trajectory.npy, and send via ZMQ.")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--npy", help="Path to an existing joint_trajectory.npy")
    g.add_argument("--test-script",default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../myGym/test.py")), help="Path to test.py (to run and capture the produced joint_trajectory.npy)")
    #absolute path
    parser.add_argument("--test-args", default="", help="Arguments passed to test.py (as a single quoted string).")
    parser.add_argument("--cwd", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../myGym/")), help="Working directory to run test.py in.")
    parser.add_argument("--zmq", default="tcp://0.0.0.0:242424", help="ZMQ endpoint, e.g., tcp://0.0.0.0:242424")
    #RETURN
    parser.add_argument("--return-trip",action="store_true",help="After sending the forward trajectory, wait at the final pose and send the reverse trajectory back to start")
    parser.add_argument("--pause-sec", type=float, default=1.0, help="Dwell time at the final pose before reversing")
    parser.add_argument("--hz", type=float, default=50.0, help="Assumed playback rate on the robot side to convert seconds to samples")
    
    #parser.add_argument("--targets", default=None, help="Multiple targets as a single string")
    parser.add_argument("--targets-file", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../zmq-comm/targets.txt")), help="Path to a text file with reach position")
    args = parser.parse_args()

    if args.test_script:
        targets = read_targets_file(args.targets_file)
        base_args = strip_existing_target(args.test_args)
        #stack all trajectory
        all_trajs = []
        for idx, target in enumerate(targets, 1):
            #tstr = f"{tgt[0]:.6f} {tgt[1]:.6f} {tgt[2]:.6f}"
            run_args = (base_args + " " if base_args else "") + f"--target {target}"
            print(f"[INFO] === Sim {idx}/{len(targets)}: target {target} ===")

            npy_path = run_test_and_get_npy(args.test_script, run_args, args.cwd)
            traj = load_traj(npy_path)  
            traj = make_roundtrip(traj)
            all_trajs.append(traj)
        big_traj = np.vstack(all_trajs)
        resp = send_over_zmq(big_traj, args.zmq)
    else:
        npy_path = args.npy
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"[ERROR] npy not found: {npy_path}") 

        traj = load_traj(npy_path)
        #print("[INFO] Loaded trajectory with shape:", traj.shape)
        traj=make_roundtrip(traj)
        print("[INFO] Round trip added. New shape", traj.shape)
        resp = send_over_zmq(traj, args.zmq)
        print("resp:", resp)


if __name__ == "__main__":
    main()
