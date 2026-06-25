import subprocess, sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "krishi_mitra.py"], check=True)

