import argparse
import json
import time
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data

# ---- myGym utilities (既存の test.py と同じ import 群) ----
try:
    from myGym.train import (
        get_parser,
        get_arguments,
        configure_implemented_combos,
        configure_env,
        automatic_argument_assignment,
    )
except Exception as e:
    raise RuntimeError("myGym.train.* が import できません。myGym 環境で実行してください") from e


# ========================= GUI helpers =========================

def _ensure_gui():
    try:
        if p.isConnected():
            return 0
    except Exception:
        pass

    try:
        return p.connect(p.GUI)
    except Exception:
        try:
            return p.connect(p.SHARED_MEMORY)
        except Exception:
            return p.connect(p.DIRECT)


def _add_target_sliders(x0=0.35, y0=0.0, z0=0.10, ws=None):
    if ws is None:
        ws = dict(xmin=0.05, xmax=0.80, ymin=-0.45, ymax=0.45, zmin=0.05, zmax=0.80)
    tx = p.addUserDebugParameter("Target X (m)", ws["xmin"], ws["xmax"], x0)
    ty = p.addUserDebugParameter("Target Y (m)", ws["ymin"], ws["ymax"], y0)
    tz = p.addUserDebugParameter("Target Z (m)", ws["zmin"], ws["zmax"], z0)
    go = p.addUserDebugParameter("EXECUTE (0/1)", 0, 1, 0)
    stop = p.addUserDebugParameter("STOP/RESET (0/1)", 0, 1, 0)
    return tx, ty, tz, go, stop


def _read_target_from_sliders(tx, ty, tz):
    return np.array([
        p.readUserDebugParameter(tx),
        p.readUserDebugParameter(ty),
        p.readUserDebugParameter(tz),
    ], dtype=float)


def _draw_target_sphere(vis_id: Optional[int], pos: np.ndarray, rgba=(1, 0, 0, 0.6), radius=0.02) -> int:
    if vis_id is None:
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        bid = p.createMultiBody(baseMass=0, basePosition=pos.tolist(), baseVisualShapeIndex=shape)
        return bid
    else:
        p.resetBasePositionAndOrientation(vis_id, pos.tolist(), [0, 0, 0, 1])
        return vis_id


# ========================= Env helpers =========================

def _apply_target_to_env(env, target_xyz):
    """環境に目標座標を反映（存在する属性に優先的に書き込む）。"""
    un = getattr(env, "unwrapped", env)

    # 1) myGym の一般的な task.goal
    try:
        if hasattr(un, "task") and hasattr(un.task, "goal"):
            un.task.goal = np.array(target_xyz, dtype=float)
            return
    except Exception:
        pass

    # 2) 直接 goal 属性
    try:
        if hasattr(un, "goal"):
            setattr(un, "goal", np.array(target_xyz, dtype=float))
            return
    except Exception:
        pass


def _get_ee_position(env) -> Optional[np.ndarray]:
    un = getattr(env, "unwrapped", env)
    for attr in ["ee_pos", "ee_position", "tcp_pos", "ee_xyz"]:
        try:
            val = getattr(un, attr)
            if val is not None:
                arr = np.array(val, dtype=float).reshape(-1)
                if arr.size >= 3:
                    return arr[:3]
        except Exception:
            pass

    # 観測から拾える場合
    try:
        obs = getattr(un, "_last_obs", None)
        if obs is None:
            obs, _ = env.reset()
        if isinstance(obs, dict):
            for key in ["ee_pos", "ee_position", "tcp_pos", "ee_xyz"]:
                if key in obs:
                    arr = np.array(obs[key], dtype=float).reshape(-1)
                    if arr.size >= 3:
                        return arr[:3]
    except Exception:
        pass
    return None


# ========================= Model loader =========================

def _load_model(arg_dict, env):
    """myGym の implemented_combos 経由でロード。SB3 .zip にもフォールバック。"""
    # myGym ルート
    try:
        implemented_combos = configure_implemented_combos(arg_dict)
        loader = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0]
        model = loader.load(arg_dict["pretrained_model"], env=env)
        return model
    except Exception:
        pass

    # SB3 フォールバック
    try:
        from stable_baselines3 import PPO, SAC, TD3, A2C
        for cls in (PPO, SAC, TD3, A2C):
            try:
                return cls.load(arg_dict["pretrained_model"], env=env)
            except Exception:
                continue
    except Exception:
        pass

    raise RuntimeError("モデルのロードに失敗しました。--pretrained_model のパスと algo/train_framework を確認してください。")


# ========================= Main =========================

def main():
    # myGym の引数系をそのまま利用
    parser = get_parser()
    #parser.add_argument("--gui", type=int, default=1)
    #parser.add_argument("--success_thresh", type=float, default=0.03)
    #parser.add_argument("--episode_horizon", type=int, default=600)

    args = get_arguments(parser)
    automatic_argument_assignment(args)
    arg_dict = vars(args)

    # GUI 準備
    if arg_dict.get("gui", 0):
        _ensure_gui()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.95, cameraYaw=90, cameraPitch=-15, cameraTargetPosition=[0, 0, 0.3])

    # 環境作成（myGym 既定の方法）
    env = configure_env(args)

    # モデル読込
    model = _load_model(arg_dict, env)

    # スライダー作成
    tx, ty, tz, go, stop = _add_target_sliders()
    target_vis = None

    obs, info = env.reset()
    last_go = 0

    while True:
        target = _read_target_from_sliders(tx, ty, tz)
        _apply_target_to_env(env, target)
        target_vis = _draw_target_sphere(target_vis, target)

        go_val = int(p.readUserDebugParameter(go))
        stop_val = int(p.readUserDebugParameter(stop))

        if stop_val == 1:
            try:
                env.reset()
            finally:
                p.addUserDebugParameter("STOP/RESET (0/1)", 0, 1, 0)
            continue

        # 立ち上がりで1試行
        if last_go == 0 and go_val == 1:
            obs, info = env.reset()
            _apply_target_to_env(env, target)
            target_vis = _draw_target_sphere(target_vis, target)

            for _ in range(arg_dict["episode_horizon"]):
                # 学習ポリシーのアクション
                try:
                    action, _ = model.predict(obs, deterministic=True)
                except TypeError:
                    action = model.predict(obs)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # GUI レンダリング
                try:
                    env.render()
                except Exception:
                    pass

                # 成功 or 完了
                ee = _get_ee_position(env)
                if ee is not None and np.linalg.norm(ee - target) <= arg_dict["success_thresh"]:
                    break
                if done:
                    break

                # 手動停止
                if int(p.readUserDebugParameter(stop)) == 1:
                    p.addUserDebugParameter("STOP/RESET (0/1)", 0, 1, 0)
                    break

            # EXECUTE を戻す
            p.addUserDebugParameter("EXECUTE (0/1)", 0, 1, 0)
            go_val = 0

        last_go = go_val
        time.sleep(0.01)


if __name__ == "__main__":
    main()