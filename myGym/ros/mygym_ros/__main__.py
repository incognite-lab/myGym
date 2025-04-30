from subprocess import call

MAIN_MODULE_FILE = "recording_viz.py"

if __name__ == "__main__":



    call(["streamlit", "run", str(main_file)])
