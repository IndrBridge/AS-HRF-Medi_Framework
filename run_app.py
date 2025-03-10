"""
Script to run the Medicinal Harmonic Resonance AI app.
"""
import os
import sys
import subprocess

# Define the base path
base_path = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(base_path, 'med_harmonic_ai', 'api', 'app.py')

# Directly run streamlit through Python module
print(f"Running Streamlit app: {app_path}")
cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run", 
    app_path,
    "--server.port=8502"
]

print(f"Command: {' '.join(cmd)}")
subprocess.run(cmd) 