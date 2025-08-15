import time

import numpy as np
import streamlit as st
from pathlib import Path
from typing import Optional, List, Dict
from zmq_playback import ZMQPlayback
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from threading import Thread
from streamlit_autorefresh import st_autorefresh
import mygym_ros.transmit_plugins as transmit_plugins


class StZMQPlayback(ZMQPlayback):
    def __init__(self, name: str):
        super().__init__(name)
        self.item_count_place_holder = None
        self.playing_started_already = False

    @st.fragment
    def render(self):
        with st.container(border=True):
            # Container content
            cols = st.columns([3, 2, 2, 2])
            cols[0].subheader(self.name)
            cols[1].metric("Items", len(self))

            if self.can_record:
                cols[2].metric("Status", "●" if self.is_recording_active else "○",
                               "Recording" if self.is_recording_active else "-Stopped",
                               border=True
                               )
            else:
                if self.is_playing:
                    cols[2].metric("Status", "\u23F5", "Playing", border=True)
                else:
                    cols[2].metric("Status", "\u23F8", "-Paused", border=True)
            # Control buttons
            with cols[3]:
                if self.can_record:
                    if self.is_recording_active:
                        if st.button("⏹️ stop rec", key=f"stop_{self.name}"):
                            self.stop_recording()
                            st.rerun(scope="fragment")
                    else:
                        if st.button("▶️ record", key=f"start_{self.name}"):
                            # Stop all other playbacks
                            for other_pb in st.session_state.playbacks:
                                if other_pb != self:
                                    other_pb.stop_recording()
                            self.start_recording()
                            st.rerun(scope="fragment")

                if st.button("❌ delete", key=f"delete_{self.name}"):
                    self.stop_recording()
                    self.pause()
                    st.session_state.playbacks.remove(self)
                    st.rerun()

            # Progress bar (for playback)
            # if self.current_pos > 0:
            progress = self.current_pos / len(self) if len(self) > 0 else 0
            st.progress(progress,
                        f"Playback progress: {self.current_pos}/{len(self)} items")

            ctrl_cols = st.columns([1, 4, 1, 1])
            if self.is_recording_active:
                st.write("⏳ Recording...")
            else:
                with ctrl_cols[0]:
                    if self.is_playing:
                        if st.button("⏸️ pause", key=f"pause_{self.name}"):
                            self.pause()
                            st.rerun(scope="fragment")
                    else:
                        if st.button("▶️ play", key=f"play_{self.name}"):
                            if self.can_record:
                                self.finalize_recording()
                            self.play()
                            st.rerun(scope="fragment")
                with ctrl_cols[1]:
                    # Speed control
                    new_speed = st.slider("Playback speed [s]:",
                                            min_value=float(self.MIN_SPEED),
                                            max_value=float(self.MAX_SPEED),
                                            value=float(self.speed),
                                            step=float(0.1),
                                            key=f"speed_{self.name}")
                    if new_speed != self.speed:
                        self.speed = new_speed
                        if self.speed != new_speed:
                            st.warning(f"Playback speed changed to {self.speed}")
                if self.can_play:
                    with ctrl_cols[2]:
                        if st.button("⏮️ rewind", key=f"rewind_{self.name}"):
                            self.seek(0)
                            st.rerun()
                with ctrl_cols[3]:
                    # Save button
                    if st.button("💾 Save", key=f"save_{self.name}"):
                        save_path = Path(f"{self.name}.json")
                        self.save(save_path)
                        st.success(f"Saved to {save_path}")

        if len(self) > 0:
            with st.expander("Data"):
                st.write(self.data[:20])

    def _get_data_generator(self):
        return super().play()

    def play(self):
        if "transmitter" not in st.session_state:
            st.error("No transmitter found")
            return
        transmitter = st.session_state.transmitter
        gen = self._get_data_generator()
        for data in gen:
            if data is None:
                continue
            if isinstance(data, np.ndarray):
                d = data.tolist()
            elif isinstance(data, list):
                d = data
            elif isinstance(data, dict):
                d = list(data.values())
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            transmitter.send_data(d)
            # print(_, "a")
        st.rerun()


class StZMQInstantPlayback(StZMQPlayback):
    MIN_SPEED = 0.01

    def __init__(self, name: str):
        super().__init__(name)
        self.speed = self.MIN_SPEED

    def play(self):
        if "transmitter" not in st.session_state:
            st.error("No transmitter found")
            return
        transmitter = st.session_state.transmitter
        gen = self._get_data_generator()
        out_data = []
        joint_names = []
        for data in gen:
            if data is None:
                continue
            if isinstance(data, np.ndarray):
                d = data.tolist()
            elif isinstance(data, list):
                d = data
            elif isinstance(data, dict):
                d = list(data.values())
                if len(joint_names) == 0:
                    joint_names = list(data.keys())
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            out_data.append(d)
        transmitter.send_data({
            "joint_names": joint_names, "data": out_data}
        )
        st.rerun()


def isolate_transmit_plugins():
    plugs = transmit_plugins.__dict__.values()
    clean_plugs = [
        plug for plug in plugs if not str(plug).startswith("_") and type(plug) is type
    ]
    # clean_plugs = [
    #     str(plug) + str(type(plug)) for plug in plugs
    # ]
    # convert to dict
    plugin_dict = {str(plug): plug for plug in clean_plugs}
    # return clean_plugs
    return plugin_dict


def load_playback():
    loader = st.session_state["load_playback"]
    if loader is None:
        return
    loaded_file = loader.name
    # st.write(loader.__dict__)
    loaded_pb = StZMQInstantPlayback.load(loaded_file)
    st.session_state.playbacks.append(loaded_pb)
    # loaded_file.close()


def main():
    st.title("ZMQ Data Recorder/Playback")

    # Initialize session state
    if 'playbacks' not in st.session_state:
        st.session_state.playbacks = []

    # Sidebar controls
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 40% !important; # Set the width to your desired value
                vertical-align: top;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Controls")
    with st.sidebar:
        new_name = st.text_input("Recording name")
        # if st.button("Create New Playback"):
        #     if any(pb.name == new_name for pb in st.session_state.playbacks):
        #         st.warning(f"A playback with name '{new_name}' already exists.")
        #     else:
        #         # Ensure only one active playback
        #         for pb in st.session_state.playbacks:
        #             pb.stop_recording()

        #         new_pb = StZMQPlayback(new_name or f"Recording_{len(st.session_state.playbacks) + 1}")
        #         new_pb.start_recording()
        #         st.session_state.playbacks.append(new_pb)

        if st.button("Create New Instant Playback"):
            if any(pb.name == new_name for pb in st.session_state.playbacks):
                st.warning(f"A playback with name '{new_name}' already exists.")
            else:
                # Ensure only one active playback
                for pb in st.session_state.playbacks:
                    pb.stop_recording()

                new_pb = StZMQInstantPlayback(new_name or f"Recording_{len(st.session_state.playbacks) + 1}")
                new_pb.start_recording()
                st.session_state.playbacks.append(new_pb)

        st.divider()
        # File uploader for loading
        loaded_file = st.file_uploader(
            "Load Playback",
            type=['json'],
            key="load_playback",
            on_change=load_playback,
            accept_multiple_files=False
        )

        st.divider()
        if "transmitter" not in st.session_state:
            # Transmit plugin dropdown & selection
            if "transmit_plugins_list" not in st.session_state:
                st.session_state.transmit_plugins_list = isolate_transmit_plugins()
            transmit_plugin = st.selectbox(
                "Transmit plugin",
                list(st.session_state.transmit_plugins_list.keys()),
                placeholder="Select a transmit plugin",
                index=None,
            )
            if transmit_plugin:
                st.session_state.transmitter = st.session_state.transmit_plugins_list[transmit_plugin]()
        else:
            st.write(f"Using transmit plugin: {st.session_state.transmitter}")
            if st.button("Change Transmit Plugin"):
                del st.session_state.transmitter

    # Main display area
    for idx, pb in enumerate(st.session_state.playbacks[:]):
        # st.write(f"## {idx + 1}. {pb.name}")
        pb.render()

    st_autorefresh(interval=1000, key="fizzbuzzcounter")


if __name__ == "__main__":
    main()
