# Entry point for Streamlit Cloud
if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", "krishi_mitra.py"], check=True)
