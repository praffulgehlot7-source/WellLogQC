# WellLogQC Enterprise: AI-Assisted Petrophysical Quality Control Platform

> **Status:** Fully functional enterprise platform with dedicated GPU reasoning microservice and multi-page dashboard. Strictly professional interface without emojis.

---

## 1. System Overview

Automates petrophysical well-log quality control and preliminary formation evaluation from Log ASCII Standard (LAS) files. The platform combines deterministic engineering physics with GPU-accelerated machine learning to prioritize anomalous intervals and deliver actionable drilling, logging, and evaluation recommendations.

### Key Capabilities
- **Agnostic Curve Sequencing:** Ingests LAS 1.2 and 2.0 files with arbitrary curve column ordering; mnemonics are mapped dynamically via [`curve_aliases.yaml`](file:///e:/Reservoir/welllogqc/config/curve_aliases.yaml).
- **Strict Depth Validation:** Validates strictly ascending depth sequences and detects non-uniform acquisition sampling steps.
- **Dedicated GPU Microservice:** Standalone FastAPI service running in Command Prompt utilizes your NVIDIA CUDA GPU (GeForce GTX 1650) for accelerated feature matrices, XGBoost inference, and reasoning.
- **World-Class Ideal Reservoir Benchmarking:** Formulates multi-parameter proximity metrics comparing target wells against standard prolific clastic reservoirs (Wilcox / Brent clean sandstone pay).
- **Automated Interval Decisions:** Issues definitive operational dispositions (`APPROVED_PAY`, `REJECTED_WASHOUT`, `RE_LOG_RECOMMENDED`, `QUESTIONABLE_TIGHT`).
- **Domain Advisory Engine:** Synthesizes actionable improvements spanning drilling mud geomechanics, wireline cable dynamics, and core-calibrated petrophysical volumetrics.
- **Open Internet Data Ingestion:** Fetches public petrophysical reference logs (e.g., USGS, Kansas Geological Survey) directly through web feeds.

---

## 2. Operational Workflow

The application workflow operates via two connected components:

```
+------------------------------------+           REST API            +---------------------------------------+
|  Command Prompt (CMD)             |  -------------------------->  |  Streamlit Enterprise Dashboard       |
|  run_ml_service.bat                |  http://127.0.0.1:8000        |  http://localhost:8501                |
|  - NVIDIA GeForce GTX 1650 (CUDA) |  <--------------------------  |  - Ingestion, QC, Training, Petro,    |
|  - Real-Time VRAM & Load Telemetry|         JSON Payload           |    Ideal Benchmarking, & Review Hub   |
+------------------------------------+                               +---------------------------------------+
```

### Step 1: Launch the GPU ML Service in Command Prompt (CMD)
Open your Command Prompt or double-click the batch launcher:

```cmd
run_ml_service.bat
```

Or from PowerShell:
```powershell
.\.venv\Scripts\python.exe run_ml_service.py
```

The CMD console displays an executive banner with hardware telemetry:
```
========================================================================================
     PETROPHYSICAL AI REASONING & INFERENCE SERVICE (GPU-ACCELERATED)
========================================================================================
  Host Device        : NVIDIA GeForce GTX 1650
  Hardware Mode      : NVIDIA CUDA (Active)
  Total VRAM         : 4096 MB (Free: 3935 MB)
  Driver Version     : 592.82
  Service Endpoint   : http://127.0.0.1:8000
========================================================================================
```

### Step 2: Launch the Enterprise Dashboard
In a separate terminal, launch the dashboard:

```powershell
.\.venv\Scripts\streamlit.exe run welllogqc/app/main.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser. The dashboard automatically connects to the GPU service and displays live device metrics.

---

## 3. Dashboard Modules

1. **`1_Data_Ingestion`**:
   - Ingest local `.las` files or disk archives.
   - Internet Data Ingestion tab: one-click retrieval of public open-access logs (USGS, KGS) or custom remote URLs.
   - Monotonic depth validation and null sentinel standardization.
2. **`2_Quality_Control`**:
   - Caliper breakout (washout) and tight-hole tolerances.
   - Modified Z-Score spike filter and tool sticking detection.
   - Multi-track vertical well-log visualization with color-coded QC bands.
3. **`3_Model_Training`**:
   - Ground-truth interval manager (`Acceptable`, `Questionable`, `Unreliable`).
   - Fit GPU-accelerated XGBoost (`device='cuda'`), Random Forest, or Gradient Boosting with well-level cross-validation.
4. **`4_GPU_Inference`**:
   - Dispatches inference requests to the CUDA GPU microservice with sub-millisecond execution.
   - Continuous QC-risk curve, risk histograms, and feature importance rankings.
5. **`5_Petrophysical_Analysis`**:
   - Computes $V_{sh}$ (Linear IGR), density porosity ($\phi_D$), Archie water saturation ($S_w$), and Net-to-Gross ($NTG$).
   - **Ideal Reservoir Benchmark Gap Analysis**: Multi-parameter proximity radar scores evaluating Gauge Stability, Matrix Cleanliness, Porosity Quality, and Hydrocarbon Saturation.
6. **`6_Executive_Review`**:
   - Consolidated KPI cards (Net Pay, NTG, Mean Porosity, Proximity Score).
   - Automated interval decision table with petrophysicist disposition overrides.
   - Actionable engineering improvements (Mud Weight Geomechanics, Cable Dynamics, Reserve Volumetrics).
   - Export CSV deliverables and executive HTML audit reports.

---

## 4. Verification & Testing

To execute all unit, GPU telemetry, and advisory tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

To run the end-to-end integration pipeline:
```powershell
.\.venv\Scripts\python.exe integration_test.py
```
