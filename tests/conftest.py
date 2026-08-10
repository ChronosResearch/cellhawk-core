import sys
import pathlib

# Make the repo root importable so `from python.cortex...` works in tests
sys.path.insert(0, str(pathlib.Path(__file__).parent))
