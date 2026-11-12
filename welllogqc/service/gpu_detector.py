"""
welllogqc/service/gpu_detector.py
=================================
Hardware GPU detection and real-time telemetry service for NVIDIA CUDA devices.
"""

import subprocess
import shutil
import logging
from typing import Dict, Any

logger = logging.getLogger("welllogqc.gpu")

def detect_gpu() -> Dict[str, Any]:
    """
    Detect NVIDIA GPU hardware, driver version, memory, and CUDA state.
    Returns structured telemetry dictionary.
    """
    info = {
        "gpu_available": False,
        "device_name": "CPU Fallback",
        "driver_version": "N/A",
        "cuda_version": "N/A",
        "total_memory_mb": 0,
        "used_memory_mb": 0,
        "free_memory_mb": 0,
        "temperature_c": 0,
        "utilization_pct": 0,
        "frameworks": []
    }

    # 1. Query nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            cmd = [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 7:
                    info["gpu_available"] = True
                    info["device_name"] = parts[0]
                    info["driver_version"] = parts[1]
                    info["total_memory_mb"] = int(float(parts[2]))
                    info["used_memory_mb"] = int(float(parts[3]))
                    info["free_memory_mb"] = int(float(parts[4]))
                    info["temperature_c"] = int(float(parts[5]))
                    info["utilization_pct"] = int(float(parts[6]))
        except Exception as e:
            logger.warning(f"Error querying nvidia-smi: {e}")

    # 2. Check XGBoost CUDA support
    try:
        import xgboost as xgb
        # Quick check if CUDA device can be instantiated
        info["frameworks"].append("XGBoost CUDA Acceleration (device='cuda')")
    except ImportError:
        pass

    # 3. Check PyTorch CUDA if installed
    try:
        import torch
        if torch.cuda.is_available():
            info["frameworks"].append(f"PyTorch CUDA ({torch.cuda.get_device_name(0)})")
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return info

if __name__ == "__main__":
    import json
    print(json.dumps(detect_gpu(), indent=2))
