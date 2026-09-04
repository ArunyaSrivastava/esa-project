"""
Single-Command Launcher for Personal Safety Monitor.
Usage:
    python run.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from backend.config import HOST, PORT

if __name__ == "__main__":
    print("==================================================")
    print("   MULTIMODAL THREAT & DISTRESS DETECTION SYSTEM   ")
    print("==================================================")
    print(f"Launching Live Web Dashboard at: http://localhost:{PORT}")
    print("Press Ctrl+C to terminate.")
    print("==================================================")
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False, log_level="info")
