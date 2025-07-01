from subprocess import call
import os

MAIN_MODULE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording_viz.py")

if __name__ == "__main__":
    call(["streamlit", "run", str(MAIN_MODULE_FILE)])
