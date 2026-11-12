"""
welllogqc/service/client.py
===========================
Client connector for the GPU ML Microservice.
Provides methods for Streamlit pages to query GPU telemetry, execute GPU inference,
and retrieve Ideal Reservoir benchmarks and petrophysical decisions.
"""

from typing import Dict, Any, List, Optional
import requests
import logging

logger = logging.getLogger("welllogqc.client")

SERVICE_URL = "http://127.0.0.1:8000"

def check_service_connection(timeout: float = 1.0) -> Dict[str, Any]:
    """Check if the GPU ML Service is running in CMD."""
    try:
        r = requests.get(f"{SERVICE_URL}/health", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return {
                "online": True,
                "gpu": data.get("gpu", {}),
                "message": "Connected to GPU Service"
            }
    except Exception:
        pass
    return {
        "online": False,
        "gpu": {"device_name": "Offline", "gpu_available": False},
        "message": "GPU Service is not running. Launch 'run_ml_service.bat' in CMD."
    }

def get_live_gpu_metrics(timeout: float = 1.0) -> Dict[str, Any]:
    """Retrieve live GPU utilization, temperature, and memory from service."""
    try:
        r = requests.get(f"{SERVICE_URL}/gpu/metrics", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"gpu_available": False, "device_name": "N/A"}

def get_service_status(timeout: float = 1.0) -> Dict[str, Any]:
    """Inspect active loaded model, device placement, and memory state."""
    try:
        r = requests.get(f"{SERVICE_URL}/status", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"active_model_loaded": False, "device_mode": "Offline"}

def load_service_model(model_path: Optional[str] = None, timeout: float = 5.0) -> Dict[str, Any]:
    """Instruct the GPU service to hot-reload a trained model checkpoint."""
    try:
        payload = {"model_path": model_path} if model_path else {}
        r = requests.post(f"{SERVICE_URL}/load_model", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.error(f"Failed to load model on GPU service: {e}")
    return {"success": False, "error": "GPU Service unreachable"}

def run_service_inference(
    well_name: str,
    feature_df: Any,
    threshold: float = 0.5,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """Send feature matrix to GPU Service for CUDA inference."""
    try:
        clean_features = feature_df.fillna(0.0).to_dict(orient="records") if hasattr(feature_df, "fillna") else feature_df
        payload = {
            "well_name": well_name,
            "features": clean_features,
            "threshold": threshold
        }
        r = requests.post(f"{SERVICE_URL}/predict", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.error(f"GPU inference request failed: {e}")
    return {"risk_scores": [], "risk_labels": [], "elapsed_ms": 0, "device_used": "CPU"}

def run_service_analysis(
    well_name: str,
    df: Any,
    vsh: Optional[Any] = None,
    phid: Optional[Any] = None,
    sw: Optional[Any] = None,
    qc_flags: Optional[Any] = None,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """Request multi-sensor reasoning, ideal reservoir proximity, and advisories."""
    try:
        clean_df = df.fillna(0.0) if hasattr(df, "fillna") else df
        payload = {
            "well_name": well_name,
            "data": clean_df.reset_index().to_dict(orient="records") if "DEPT" not in clean_df.columns else clean_df.to_dict(orient="records"),
            "vsh": vsh.fillna(0.0).tolist() if (vsh is not None and hasattr(vsh, "fillna")) else (vsh.tolist() if vsh is not None else None),
            "phid": phid.fillna(0.0).tolist() if (phid is not None and hasattr(phid, "fillna")) else (phid.tolist() if phid is not None else None),
            "sw": sw.fillna(0.0).tolist() if (sw is not None and hasattr(sw, "fillna")) else (sw.tolist() if sw is not None else None),
            "qc_flags": qc_flags.fillna("").to_dict(orient="records") if qc_flags is not None and hasattr(qc_flags, "to_dict") else None
        }
        r = requests.post(f"{SERVICE_URL}/analyze", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.error(f"Analysis request failed: {e}")
    return {}
