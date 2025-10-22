#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstructed from photos by ChatGPT (2025-10-13).
Executable ROS2 node without line numbers.
"""

import json													##ROSメッセージをJSONでやり取り
import re														##正規表現処理　thisやthatを抽出
import time													##時間処理　ジェスチャーなどで使用
from threading import Lock
from collections import deque
															##これ以降ROSのやつ
import numpy as np
import rclpy													##ROSのpythonクライアントライブラリ
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from geometry_msgs.msg import Point
from std_msgs.msg import Bool

# ワークスペースが gesture_msgs で HRICommand を提供している場合				##ここからHRIコマンド
# ここにインポートする。インポートが失敗した場合、堅牢性を保つために最小限のスタブにフォールバックする。
try:
    from gesture_msgs.msg import HRICommand  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without the msg definition
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class HRICommand:  # very small shim used only to let the file import
        data: List[str] = field(default_factory=list)


DEICTIC_SEQUENCE_TOPIC = "/teleop_gesture_toolbox/deictic_sentence"#ジェスチャー解析の結果
LANGUAGE_TOPIC = "/modality/nlp"#NLPの解析結果
FUSION_TOPIC = "/modality/gesture_language_fusion"#融合結果
REASONER_START_TOPIC = "reasoner/start_episode"#reasonerの開始通知

# ---------------------------- helpers ----------------------------

def _to_jsonable(obj):
    """Convert possibly nested objects to JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, deque)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Point):
        return {"x": obj.x, "y": obj.y, "z": obj.z}
    try:
        # array.array or other sequence-like objects
        import array  # noqa: F401
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            try:
                return list(obj)
            except Exception:
                pass
    except Exception:
        pass
    return obj
#ROSや数値型をJSONに変換するための関数

def _dict_to_point(d):
    """Convert {'x':..., 'y':..., 'z':...} to geometry_msgs/Point."""
    return Point(x=d.get("x", 0.0), y=d.get("y", 0.0), z=d.get("z", 0.0))
#3次元座標はをgeometry_msgs/Pointに変換する関数（pythonで理解できる形）

def _loads_from_msg_data(msg) -> object:
    """Be tolerant: HRICommand.data could be string or a list[str]."""
    payload = getattr(msg, "data", None)
    if isinstance(payload, (list, tuple)):
        if not payload:
            raise IndexError("empty message data array")
        payload = payload[0]
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        payload = str(payload)
    return json.loads(payload)
#いろんな形式のHRICommand.dataを統一したJSONの形として読み込む関数

# ------------------------------ nodes ------------------------------

class RosNode(Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.last_time_livin = time.time()

    def spin_once(self):
        self.last_time_livin = time.time()
        rclpy.spin_once(self)


class DeicticSentenceInput:
    """
    Listens on DEICTIC_SEQUENCE_TOPIC and stores *all* deictic solutions
    until fetched with get_sentence().
    Expected message payload: JSON-encoded list[dict] (solutions).
    """

    def __init__(self, node: Node):
        self.node = node
        self._lock = Lock()
        self._solutions = []  # type: list[dict]
        self._enabled = True
        self._received_data = False

        self.sub = self.node.create_subscription(
            HRICommand,
            DEICTIC_SEQUENCE_TOPIC,
            self._sentence_callback,
            10,
        )

    def _sentence_callback(self, msg: HRICommand):
        if not self._enabled:
            return
        self._enabled = False

        try:
            solutions_list = _loads_from_msg_data(msg)
        except (IndexError, json.JSONDecodeError) as e:
            self.node.get_logger().error(f"[DeicticSentenceInput] invalid message: {e}")
            self._enabled = True
            return

        if not isinstance(solutions_list, list):
            self.node.get_logger().error("[DeicticSentenceInput] expected list of solutions")
            self._enabled = True
            return

        # Re-create geometry_msgs/Point fields from dicts if present
        for sol in solutions_list:
            if isinstance(sol, dict):
                if "target_object_position" in sol and isinstance(sol["target_object_position"], dict):
                    sol["target_object_position"] = _dict_to_point(sol["target_object_position"])
                if "line_points" in sol and isinstance(sol["line_points"], list) and len(sol["line_points"]) >= 2:
                    if isinstance(sol["line_points"][0], dict):
                        sol["line_points"][0] = _dict_to_point(sol["line_points"][0])
                    if isinstance(sol["line_points"][1], dict):
                        sol["line_points"][1] = _dict_to_point(sol["line_points"][1])

        with self._lock:
            self._solutions = solutions_list
            self._received_data = True

        self.node.get_logger().info(f"[DeicticSentenceInput] received {len(solutions_list)} gestures")

    def has_sentence(self) -> bool:										
        with self._lock:
            return self._received_data

    def get_sentence(self):
        """Return the list of solutions and clear the buffer."""
        with self._lock:
            sols = self._solutions
            self._solutions = []
            self._received_data = False
            self._enabled = True
        self.node.get_logger().info(f"[DeicticSentenceInput] delivering {len(sols)} gestures")
        return sols

#言語入力を取り扱っているクラス
class LanguageInput:												##LanguageInput
    """
    Handles reception of language data from NLP.
    One ROS message may contain *one* or *many* simple-sentence dicts.
    Each message is treated as ONE logical command and kept together
    as list[dict].
    """

    def __init__(self, node: Node):#selfはインスタンス自身、nodeはROSノード
        self.node = node
        self._lock = Lock()
        self._buf = []  # type: list[list[dict]]
        self._enabled = True

        self.sub = self.node.create_subscription(
            HRICommand,
            LANGUAGE_TOPIC,
            self._language_callback,
            10,
        )

    def _language_callback(self, msg: HRICommand):
        if not self._enabled:
            return
        self._enabled = False

        try:
            raw = _loads_from_msg_data(msg)
        except (IndexError, json.JSONDecodeError) as e:
            self.node.get_logger().error(f"[LanguageInput] invalid JSON: {e}")
            self._enabled = True
            return

        if isinstance(raw, dict):
            raw = [raw]
        elif not isinstance(raw, list):
            self.node.get_logger().error(f"[LanguageInput] unsupported payload type: {type(raw).__name__}, expected dict or list")
            self._enabled = True
            return

        # Normalize each elementary command
        norm_cmd = []
        for sent in raw:
            if not isinstance(sent, dict):
                continue
            norm_cmd.append({
                "action": (sent.get("action", "") or "").lower(),
                "objects": [sent.get("target_object", ""), sent.get("target_object2", "")],
                "action_param": (sent.get("action_parameter", "") or ""),						
                "objects_params": [sent.get("target_object_color", ""), sent.get("target_object_color2", "")],
                "raw_text": sent.get("raw_text", ""),
                "orig_idx": sent.get("orig_idx", 0),
            })

        with self._lock:
            self._buf.append(norm_cmd)

        self.node.get_logger().info(f"[LanguageInput] received NLP command ({len(norm_cmd)} sentence{'s' if len(norm_cmd)!=1 else ''})")

    def has_sentence(self) -> bool:#データ来た
        with self._lock:
            return len(self._buf) > 0

    def get_sentence(self):#データを取り出す
        with self._lock:
            cmd = self._buf.pop(0) if self._buf else None
            self._enabled = True
        self.node.get_logger().info(f"[LanguageInput] delivering {len(cmd) if cmd else 0} sentences")
        return cmd


class GestureLanguageMerger(RosNode):				##ジェスチャーと言語の統合管理
    """
    Correctly assigns gestures to language commands and publishes
    the merged modalities in order of their required execution.
    """

    def __init__(self, step_period: float = 0.2):
        super().__init__("gesture_language_fusion_node")
        qos_profile = QoSProfile(depth=5, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)#通信（Publish/Subscribe）を行うときの「データ伝達の信頼性や振る舞い」
        self.get_logger().info("running init...")
        self._lock = Lock()

        # Loop wait period
        self.step_period = step_period

        # Input listeners
        self.language_sub = LanguageInput(self)
        self.deictic_sentence_sub = DeicticSentenceInput(self)

        # Feedback from the reasoner
        self.create_subscription(Bool, REASONER_START_TOPIC, self._start_episode_cb, 10)
        self.waiting_for_reasoner = False

        # Fusion publisher
        self.pub = self.create_publisher(HRICommand, FUSION_TOPIC, qos_profile)

        # Main loop timer
        self.timer = self.create_timer(self.step_period, self.main_loop)

    # ---------------- main flow ----------------

    def _start_episode_cb(self, msg: Bool):
        """Callback from reasoner to start waiting for new data."""
        if msg.data:
            with self._lock:
                self.waiting_for_reasoner = False
            self.get_logger().info("Reasoning finished, starting new episode...")

    def main_loop(self):
        """
        Main loop of the fusion module, where the gestures are assigned to language commands.
        Publishes the message containing the elementary language commands enriched by the list of gestures.

        The entire message published is a list[dict], where each dict has the following keys:
            action: desired action type ('pick', 'place', 'move')
            objects: list of object names [target, ref]
            action_param: action parameter (relation)
            objects_params: list of object params [param_target, param_ref], now only colors
            raw_text: the original wording of the elementary command in the sentence
            orig_idx: order of the command in the original sentence
            gesture_role: determined role of gesture ("object" or "place"), only for 'move' action w. 1 gesture
            gestures: list of assigned gestures
        """
        # If some previous command is being executed, wait for its end
        if self.waiting_for_reasoner:
            self.get_logger().info("Waiting for reasoner...")
            return

        # 言語とジェスチャーが両方来ているか確認
        if not (self.language_sub.has_sentence() and self.deictic_sentence_sub.has_sentence()):
            # re-enable reception just in case
            self.deictic_sentence_sub._enabled = True
            self.language_sub._enabled = True
            return

        # Load inputs
        nlp_data = self.language_sub.get_sentence()
        deictic_batch = self.deictic_sentence_sub.get_sentence()

        if nlp_data is None or deictic_batch is None:
            return

        # ----- ジェスチャーを言語コマンドに変換 -----
        if len(nlp_data) == 1:#言語コマンドが1つ
            # Only one elementary sentence: attach all gestures to it.
            nlp_data[0]["gestures"] = deictic_batch

            # For 'move' with one gesture, determine the role (object/place)
            if nlp_data[0]["action"].lower() == "move" and len(deictic_batch) == 1:
                gesture_role = self.classify_gesture_for_move(nlp_data[0])
                if not gesture_role:
                    self.get_logger().warn("Cannot decide on gesture role, aborting...")
                    return
                nlp_data[0]["gesture_role"] = gesture_role

            fused_payload = nlp_data
        else:#言語コマンドが複数
            # More than one command -> greedy pairing
            exec_order = nlp_data  # list already in execution order
            spoken_order = sorted(exec_order, key=lambda s: s.get("orig_idx", 0))

            remaining = list(deictic_batch)  # detection order

            # Process sentences in spoken order; assigned gestures also stay in exec order
            for sentence in spoken_order:
                need = self.gestures_needed_number(sentence)
                self.get_logger().info(f"[Fusion] NLP sentence '{sentence['raw_text']}'  gestures needed: {need}")
                if len(remaining) < need:
                    self.get_logger().warn(f"[Fusion] Not enough gestures remaining!")
                    return
                sentence["gestures"] = remaining[:need]
                del remaining[:need]

                # For 'move' w/1 gesture, determine its role
                if sentence["action"].lower() == "move" and len(sentence["gestures"]) == 1:
                    gesture_role = self.classify_gesture_for_move(sentence)
                    if not gesture_role:
                        self.get_logger().warn("Cannot decide on gesture role, aborting...")
                        return
                    sentence["gesture_role"] = gesture_role

            if remaining:
                self.get_logger().warn(f"[Fusion] Unassigned gestures: {len(remaining)}")
                return

            fused_payload = exec_order

        # Publish the message as JSON string in HRICommand
        json_str = json.dumps(_to_jsonable(fused_payload))
        msg = HRICommand(data=[json_str])
        self.pub.publish(msg)
        self.get_logger().info(f"[Fusion] published fusion message")

        # Now wait for reasoner to process the commands
        with self._lock:
            self.waiting_for_reasoner = True

    # ---------------- 言語の発見方法 ----------------

    def classify_gesture_for_move(self, sentence_dict: dict):
        """
        Decide whether the gesture in a 'move' elementary command is meant
        for the target *object* or the target *place*.

        Returns: "object" | "place" | None
        """
        rel = (sentence_dict.get("action_param") or "").lower().strip()
        if not rel:
            self.get_logger().warn("[Fusion] Relation field empty")
            return None

        if rel == "here":
            return "place"

        raw_text = (sentence_dict.get("raw_text") or "").lower()
        raw_text = re.sub(r"[^\w\s]", " ", raw_text)
        before_rel, _sep, after_rel = raw_text.partition(rel)
        if not _sep:
            self.get_logger().warn("[Fusion] Relation field not in original sentence")
            return None

        # Demonstrative pronoun before relation -> object
        if re.search(r"\b(this|that)\b", before_rel):
            return "object"
        # Demonstrative after relation -> place
        if re.search(r"\b(this|that)\b", after_rel):
            return "place"

        self.get_logger().warn("[Fusion] Meaning of the relation is unclear")
        return None

    def gestures_needed_number(self, fused_dict: dict) -> int:
        """
        Determines the number of required gestures from the language
        based on the number of demonstrative pronouns and 'here'.
        """
        act = (fused_dict.get("action") or "").lower()
        param = (fused_dict.get("action_param") or "").lower()
        txt = (fused_dict.get("raw_text") or "").lower()

        has_this_that = bool(re.search(r"\b(this|that)\b", txt))

        # 'pick': depends only on demonstrative pronoun
        if act == "pick":
            return 1 if has_this_that else 0

        # 'place': need gesture if param is 'here', otherwise demonstrative
        if act == "place":
            if param == "here":
                return 1
            return 1 if has_this_that else 0

        # 'move': several options
        if act == "move":
            if param == "here":
                # 1 for 'here' + maybe 1 for object demonstrative before relation
                before, _sep, _after = txt.partition("here")
                obj_needs_gesture = bool(re.search(r"\b(this|that)\b", before))
                return 2 if obj_needs_gesture else 1
            # else: count demonstratives (cap at 2)
            n_demonstratives = len(re.findall(r"\b(this|that)\b", txt))
            return min(n_demonstratives, 2)

        self.get_logger().warn(f"[Fusion] Unsupported action '{act}'")
        return 0


def main(args=None):
    rclpy.init(args=args)#ROS2の初期化
    merger = GestureLanguageMerger()
    try:
        rclpy.spin(merger)#ROSノードを実行
    except KeyboardInterrupt:
        merger.get_logger().info("Ending by User Interrupt")
    finally:
        merger.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
