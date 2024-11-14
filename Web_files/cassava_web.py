# Import subprocess for ngrok and other configurations
import subprocess
import sys
import os

# Install necessary packages using pip
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "ipykernel", "streamlit", "pyngrok"])

# Path to your ngrok executable
ngrok_path = r"C:\Users\vishr\AppData\Local\ngrok\ngrok.exe"

# Run ngrok to expose the Streamlit app
try:
    subprocess.run([ngrok_path, "http", "8501", "--authtoken", "2l1fEPpquuVEMoSsVeDUgF6IJnP_2dagCQmZquo2GmJCqdvmH"])
except Exception as e:
    print(f"Failed to start ngrok: {e}")

# Create .streamlit directory and config.toml to suppress warnings
streamlit_dir = os.path.expanduser("~/.streamlit")
os.makedirs(streamlit_dir, exist_ok=True)

config_path = os.path.join(streamlit_dir, 'config.toml')
with open(config_path, 'w') as f:
    f.write("[logger]\nlevel = 'error'\n")

# Run the Streamlit app
subprocess.run([sys.executable, "-m", "streamlit", "run", "C:\\cassava_web_app\\web1.py", "--logger.level", "error"])