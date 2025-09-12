import os
import sys
import time
import atexit
import subprocess
from datetime import datetime

import zmq
import pandas as pd
import streamlit as st
import plotly.graph_objs as go


ADDRESS = "tcp://127.0.0.1:8003"


# -------------------------------
# Client (used from simulation code)
# -------------------------------
class DataViz:
    DISABLED = False
    _instance = None
    _socket = None
    _proc = None

    @classmethod
    def _ensure(cls):
        if cls.DISABLED:
            return
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Start Streamlit server in a subprocess
        script = os.path.abspath(__file__)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Setup ZeroMQ PUSH socket
        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.PUSH)
        self._socket.connect(ADDRESS)

        atexit.register(self._cleanup)

    def _cleanup(self):
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    @classmethod
    def log_param(cls, name, value, group="Default"):
        inst = cls._ensure()
        if inst is None:
            return
        msg = {"type": "param", "name": name, "value": float(value), "group": group, "ts": time.time()}
        inst._socket.send_json(msg)

    @classmethod
    def log_text(cls, name, text):
        inst = cls._ensure()
        if inst is None:
            return
        msg = {"type": "text", "name": name, "text": str(text), "ts": time.time()}
        inst._socket.send_json(msg)


# -------------------------------
# Server (Streamlit app)
# -------------------------------
def init_state():
    if "params" not in st.session_state:
        st.session_state.params = {}  # group -> name -> list of (t, v)
    if "texts" not in st.session_state:
        st.session_state.texts = {}   # name -> list of (t, text)
    if "pause" not in st.session_state:
        st.session_state.pause = False
    if "max_plot" not in st.session_state:
        st.session_state.max_plot = 500
    if "max_table" not in st.session_state:
        st.session_state.max_table = 50


def recv_data(sock):
    """Drain ZMQ socket into session_state."""
    try:
        while True:
            msg = sock.recv_json(flags=zmq.NOBLOCK)
            if msg["type"] == "param":
                g, n = msg["group"], msg["name"]
                st.session_state.params.setdefault(g, {}).setdefault(n, []).append((msg["ts"], msg["value"]))
                # enforce buffer limit
                if len(st.session_state.params[g][n]) > st.session_state.max_plot:
                    st.session_state.params[g][n] = st.session_state.params[g][n][-st.session_state.max_plot:]
            elif msg["type"] == "text":
                n = msg["name"]
                st.session_state.texts.setdefault(n, []).append((msg["ts"], msg["text"]))
                if len(st.session_state.texts[n]) > st.session_state.max_table:
                    st.session_state.texts[n] = st.session_state.texts[n][-st.session_state.max_table:]
    except zmq.Again:
        pass


@st.fragment
def render_params():
    st.header("Parameters")
    for g, series in st.session_state.params.items():
        st.subheader(g)
        fig = go.Figure()
        for n, vals in series.items():
            if vals:
                xs = [datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3] for t, _ in vals]
                ys = [v for _, v in vals]
                fig.add_trace(go.Scattergl(
                    x=xs, y=ys, mode="lines", name=n,
                    hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>v=%{y:.6g}<extra></extra>"
                ))
        fig.update_layout(
            margin=dict(l=30, r=10, t=30, b=30),
            paper_bgcolor="#f7f9df", plot_bgcolor="#f7f9df",
            font=dict(color="#000000"),
            xaxis=dict(gridcolor="#333333"),
            yaxis=dict(gridcolor="#333333"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


@st.fragment
def render_texts():
    st.header("Texts")
    for n, vals in st.session_state.texts.items():
        df = pd.DataFrame(
            [(datetime.fromtimestamp(t).strftime("%H:%M:%S - %d/%m/%Y"), txt) for t, txt in vals],
            columns=["time", "text"]
        )
        st.subheader(n)
        st.table(df.tail(st.session_state.max_table))


def run_server():
    if "zmq_ctx" not in st.session_state:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.bind(ADDRESS)
        st.session_state.zmq_ctx = ctx
        st.session_state.zmq_sock = sock
    else:
        # ctx = st.session_state.zmq_ctx
        sock = st.session_state.zmq_sock

    st.set_page_config(layout="wide")
    init_state()

    # controls
    st.sidebar.header("Controls")
    st.session_state.pause = st.sidebar.checkbox("Pause visualization", value=st.session_state.pause)
    st.session_state.max_plot = st.sidebar.slider("Plot buffer length", 100, 1000, st.session_state.max_plot, step=100)
    st.session_state.max_table = st.sidebar.slider("Table buffer length", 10, 100, st.session_state.max_table, step=10)

    # main loop
    placeholder_params = st.empty()
    placeholder_texts = st.empty()

    while True:
        recv_data(sock)

        if not st.session_state.pause:
            with placeholder_params.container():
                render_params()
            with placeholder_texts.container():
                render_texts()

        time.sleep(1.0)


# -------------------------------
# Entrypoint for server
# -------------------------------
if __name__ == "__main__":
    run_server()
