from io import BytesIO
import sys
import time
import cloudpickle as cp
from typing import Any
from zmq_comm.pub_sub import ParamSubscriber
from pathlib import Path
import yaml
from importlib.resources import files
import streamlit as st
from functools import lru_cache


class ZMQCommMeta(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls._zmq_config: dict[str, Any] = {}

    def __call__(cls, *args, **kwargs):
        if not cls._zmq_config:
            try:
                yaml_path = files("myGym").joinpath("ros/zmq_config.yaml")
            except ModuleNotFoundError:
                yaml_path = files("mygym_ros").joinpath("../zmq_config.yaml")
            if not yaml_path.exists():
                raise FileNotFoundError(f"ZMQ config file not found at {yaml_path}")
            with yaml_path.open("r") as f:
                cls._zmq_config = yaml.safe_load(f)
        setattr(cls, '_zmq_config', cls._zmq_config)

        constr = super().__call__(*args, **kwargs)
        return constr


class ZMQPlayback(metaclass=ZMQCommMeta):
    MIN_SPEED = 0.1
    MAX_SPEED = 10

    def __init__(self, name: str):
        self._name = name
        self._is_recording_active = False
        self._is_playing = False
        self._can_play = False
        self._can_record = True
        self._current_pos = 0
        self._speed = 1
        self._created_at = time.time()
        self._subscriber = ParamSubscriber(
            **self._zmq_config,
            callback=self._handle_data
        )
        self._subscriber.subscribe("robot_action")
        self._subscriber.subscribe("robot_grip")
        self._actions = []
        self._grips = []

    def _handle_data(self, topic, data):
        if self._is_recording_active:
            # TODO: add time stamps
            if topic == "robot_action":
                self._actions.append(data)
            elif topic == "robot_grip":
                self._grips.append(data)

    def start_recording(self):
        """Start recording new data"""
        if self.can_record and not self._is_recording_active:
            self._is_recording_active = True

    def stop_recording(self):
        """Stop recording"""
        self._is_recording_active = False

    def finalize_recording(self):
        self._is_recording_active = False
        self._can_play = True
        self._can_record = False

    def play(self):
        """Playback the recorded data"""
        if self.can_play:
            if self._is_playing:
                return
            self._is_playing = True
            for i in range(len(self)):
                # Replace with actual ZMQ sending logic
                self._current_pos = i
                yield self.data[self._current_pos]
                time.sleep(self.speed)  # Simulate playback speed
                if not self._is_playing:
                    break
            self._is_playing = False

    def pause(self):
        """Stop playback"""
        self._is_playing = False

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"{self._name}: {len(self)} items ({'Active' if self._is_recording_active else 'Inactive'})"

    def save(self, path: Path):
        """Save playback to file"""
        with open(path, 'wb') as f:
            cp.dump(self, f)

    def seek(self, pos):
        if pos < 0:
            pos = 0
        elif pos > len(self):
            pos = len(self)
        self._current_pos = pos

    @classmethod
    def load(cls, path_or_buff: Path | BytesIO) -> 'ZMQPlayback':
        """Load playback from file"""
        if isinstance(path_or_buff, BytesIO):
            # TODO: make this work
            path_or_buff.seek(0)
            return cp.load(path_or_buff)
        elif isinstance(path_or_buff, str):
            path_or_buff = Path(path_or_buff)

        if isinstance(path_or_buff, Path):
            with open(path_or_buff, 'rb') as f:
                return cp.load(f)
        else:
            raise ValueError(f"Invalid path type {type(path_or_buff)}")

    @property
    @lru_cache
    def data(self):
        # TODO: add grips and stamps
        return self._actions

    @property
    def name(self):
        return self._name

    @property
    def created_at(self):
        return self._created_at

    @property
    def is_recording_active(self):
        return self._is_recording_active

    @property
    def is_playing(self):
        return self._is_playing

    @property
    def current_pos(self):
        return self._current_pos

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        value = max(self.MIN_SPEED, min(self.MAX_SPEED, value))
        self._speed = value

    @property
    def can_record(self):
        return self._can_record

    @property
    def can_play(self):
        return self._can_play
