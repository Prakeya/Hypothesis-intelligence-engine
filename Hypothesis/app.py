import os
import sys
import subprocess
from pathlib import Path

def main():
    """ shim to run the refactored server package """
    app_path = Path(__file__).parent / "server" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    main()
