#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import time
import argparse
import numpy as np
import sys
import os

# === 必要なら myGym パスを追加 ===
sys.path.append(os.path.expanduser("~/code/mygym/myGym"))
from mygym.envs.tiago_env import TiagoEnv  # ← Tiagoの環境クラスに合わせて変更（例: kuka_envなど）

def parse_args():
    parser = argparse.ArgumentParser(description="Fixed reach test (no RL)")
    parser.add_argument("--target_q_rad", type=str, required=True,
        help="目標関節角度(rad)をカンマ区切りで指定（例: '0.0,-1.2,1.57,0.0,0.8,0.0,0.0'）")
    parser.add_argument("--p_gain", type=float, default=0.15,
        help="比例ゲイン Kp (例: 0.15)")
    parser.add_argument("--fixed_steps", type=int, default=300,
        help="繰り返しステップ数 (例: 300)")
    parser.add_argument("--gui", type=int, default=1,
        help="1でGUI表示、0で非表示")
    parser.add_argument("--q_dof", type=int, default=7,
        help="制御する関節数（右腕=7）")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 目標角度を読み込み ---
    target_q = np.array([float(x) for x in args.target_q_rad.split(",")], dtype=float)
    Kp = args.p_gain
    q_dof = args.q_dof

    # --- 環境セットアップ ---
    env = TiagoEnv(gui=args.gui)  # myGym の Tiago 環境を起動（GUI付き）

    obs, _ = env.reset()
    act_dim = env.action_space.shape[0]
    print(f"Action dim = {act_dim}")

    # --- 軌道ログ ---
    joint_traj = []
    actions_traj = []

    for step in range(args.fixed_steps):
        # 現在関節角を取得（環境によっては env.robot.get_joints_states() など）
        if hasattr(env, "robot") and hasattr(env.robot, "get_joints_states"):
            q_now = np.array(env.robot.get_joints_states(), dtype=float)[:q_dof]
        else:
            q_now = np.array(obs[:q_dof], dtype=float)

        # P制御（Δ角度コマンドを出す）
        u = Kp * (target_q - q_now)
        action = np.zeros(act_dim, dtype=float)
        action[:q_dof] = u

        obs, reward, terminated, truncated, info = env.step(action)
        joint_traj.append(q_now)
        actions_traj.append(action)

        # GUIありなら少し待つ
        if args.gui:
            time.sleep(1.0 / 60.0)

    # --- 保存 ---
    os.makedirs("./fixed_reach_logs", exist_ok=True)
    np.save("./fixed_reach_logs/joint_trajectory.npy", np.array(joint_traj))
    np.save("./fixed_reach_logs/actions.npy", np.array(actions_traj))
    print("[Done] Saved trajectories to ./fixed_reach_logs/")
    env.close()

if __name__ == "__main__":
    main()