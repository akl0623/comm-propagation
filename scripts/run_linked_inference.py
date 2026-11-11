from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    script = Path(__file__).resolve().parents[1] / 'src' / 'inference' / 'portable_inference.py'
    subprocess.run([sys.executable, '-B', str(script), *sys.argv[1:]], check=True)
