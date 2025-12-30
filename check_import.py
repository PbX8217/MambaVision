
import sys
import os
sys.path.append(os.getcwd())
try:
    from mambavision.models import mamba_vision_smt
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
