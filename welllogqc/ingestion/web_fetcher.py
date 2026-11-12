"""
welllogqc/ingestion/web_fetcher.py
==================================
Web Data Ingestion and Online Benchmark Fetcher.
Allows retrieving open-source well-log data and benchmark datasets from public URLs.
"""

from pathlib import Path
from typing import Dict, Any, List
import urllib.request
import urllib.error
import logging

logger = logging.getLogger("welllogqc.web_fetcher")

# Curated open-access well logs & public benchmark feeds
PUBLIC_BENCHMARK_SOURCES = [
    {
        "name": "USGS Standard Test Well (Public Petrophysical Reference)",
        "url": "https://raw.githubusercontent.com/petropy/petropy/master/data/raw/test.las",
        "description": "Standard clastic reference well containing GR, CALI, RHOB, NPHI, ILD, DT curves.",
        "default_filename": "USGS_REFERENCE_WELL.las",
        "role": "test"
    },
    {
        "name": "Kansas Geological Survey - Open Petrophysical Log",
        "url": "https://chasm.kgs.ku.edu/ords/oil/ogdata.las_web.dl_las?id=1043810143",
        "description": "Mid-continent Paleozoic carbonate and sandstone sequence with complete wireline suite.",
        "default_filename": "KGS_OPEN_BENCHMARK.las",
        "role": "training"
    }
]

def fetch_las_from_url(
    url: str,
    output_dir: Path,
    filename: str = None,
    timeout: int = 15
) -> Dict[str, Any]:
    """
    Download a LAS file from a remote web URL and save it to the specified directory.
    Returns status dict with path, byte size, and success flag.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not filename:
        filename = url.split("/")[-1].split("?")[0]
        if not filename.endswith(".las"):
            filename = f"web_log_{abs(hash(url)) % 10000}.las"

    dest_path = output_dir / filename
    
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "WellLogQC-Professional/2.0 (Petroleum Engineering AI Research)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()
            
        with open(dest_path, "wb") as f:
            f.write(content)
            
        return {
            "success": True,
            "path": str(dest_path),
            "filename": filename,
            "bytes_downloaded": len(content),
            "message": f"Successfully retrieved {len(content):,} bytes from {url}"
        }
    except Exception as e:
        logger.error(f"Failed to fetch LAS from {url}: {e}")
        return {
            "success": False,
            "path": None,
            "filename": filename,
            "bytes_downloaded": 0,
            "message": f"Download failed: {str(e)}"
        }

def list_public_benchmark_sources() -> List[Dict[str, str]]:
    """Return list of pre-configured public open data well sources."""
    return PUBLIC_BENCHMARK_SOURCES
