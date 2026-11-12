"""
run_ml_service.py
=================
CLI Launcher for the Petrophysical AI GPU Reasoning & Inference Service.
"""

import sys
import uvicorn
from welllogqc.service.server import print_startup_banner

if __name__ == "__main__":
    print_startup_banner()
    try:
        uvicorn.run("welllogqc.service.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\n[GPU-SERVICE] Gracefully shutting down...")
        sys.exit(0)
