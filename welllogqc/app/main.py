"""
welllogqc/app/main.py
=====================
Enterprise Petrophysical Evaluation & Machine Learning Platform.
Two-Interface Architecture:
1. Training: Paired wireline log and ground-truth result files for AI model training.
2. Analysis: Single wireline log input, automated QC, petrophysics, and interactive depth-linked review highlighting.
Strictly zero emojis.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="WellLogQC Enterprise AI Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

from welllogqc.app.ui_theme import inject_custom_theme
inject_custom_theme()

# Two-Interface Navigation Router
pages = [
    st.Page(str(APP_DIR / "pages" / "1_Model_Training.py"), title="Training", default=False),
    st.Page(str(APP_DIR / "pages" / "2_Well_Analysis.py"), title="Analysis", default=True),
]

pg = st.navigation(pages)
pg.run()
