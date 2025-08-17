#!/usr/bin/env python3

import subprocess
import sys


def run_streamlit():
    print("Launching CineGenie Streamlit app...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/streamlit_app.py",
                "--server.port",
                "8501",
                "--server.address",
                "localhost",
            ]
        )
    except KeyboardInterrupt:
        print("Streamlit stopped.")


if __name__ == "__main__":
    run_streamlit()
