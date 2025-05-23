"""
Script to launch both the API server and the UI in separate processes
"""
import subprocess
import sys
import os
import time
from threading import Thread

def run_api():
    """Run the FastAPI server"""
    print("Starting API server...")
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def run_ui():
    """Run the Gradio UI"""
    print("Waiting for API server to start...")
    time.sleep(5)  # Give the API server time to start
    print("Starting UI...")
    subprocess.run([sys.executable, "-m", "utils.ui"])

if __name__ == "__main__":
    # Start the API server in a separate thread
    api_thread = Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()
    
    # Start the UI in the main thread
    run_ui()
