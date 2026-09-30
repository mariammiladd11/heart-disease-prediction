# run_training.py
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_training():
    """Run the model training"""
    print("Starting model training...")
    subprocess.check_call([sys.executable, "train_model.py"])

if __name__ == "__main__":
    install_requirements()
    run_training()