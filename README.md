# Physics-Informed Wellbore Flow Prediction from DTS Telemetry

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Pipeline](https://github.com/aliyev-toghrul/DTS-Thermodynamic-ML-Modeling/actions/workflows/python-app.yml/badge.svg)](https://github.com/aliyev-toghrul/DTS-Thermodynamic-ML-Modeling/actions)

An end-to-end physics-informed deep learning pipeline that reconstructs continuous downhole wellbore flow profiles from Distributed Temperature Sensing (DTS) fiber-optic telemetry and Production Logging Tool (PLT) calibration data.

---

## Executive Summary

Predicting zonal flow-rate contributions along a petroleum borehole from fiber-optic temperature logs is a challenging inverse problem. Standard data-driven models often fail when extrapolating across unseen depths because they ignore physical wellbore boundaries. This project embeds thermodynamic feature engineering and physical boundary constraints directly into a lightweight **ConvLSTM** architecture.

Tested on **1GB+ (~26 million rows)** of high-noise downhole telemetry (June 2017 wellbore campaign), the pipeline translates raw geothermal anomalies into accurate, continuous flow profiles:

| Benchmark Mode | Validation Strategy | Model | Scaled MSE | R² Score | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Active Producing Zone** | Spatial Cross-Validation | **ConvLSTM + AdamW** | **0.0304** | **0.9412** | **Validated (High Fidelity)** |
| **Full Wellbore Profile** | Physics Boundary Enforcement | **PI-ConvLSTM** | **0.0321** | **0.9240** | **Physics-Grounded (Flow → 0)** |
| *Unconstrained Baseline* | *Naive Toe Extrapolation* | *Raw ConvLSTM* | *579,758* | *-92.66\** | *Unconstrained OOD Artifact\** |

> **\*Why did the initial unconstrained Colab baseline show R² = -92.66?**
> In the early unconstrained experiment, the test split was placed at the very bottom of the well (**1648 m – 1685 m**, below the perforations) where actual fluid flow is nearly zero (**actual flow ≈ 0**, meaning test variance is near zero). Because the unconstrained neural network had only seen active upper-wellbore flow during training, it predicted ~700 BFPD in this dead zone. Dividing that prediction error by a near-zero test variance caused the R² formula (`1 - MSE / Variance`) to blow up to **-92.66**. Adding physical zero-flow boundary conditions at the perforation base resolves this domain shift and aligns full-wellbore behavior with the **R² = 0.9412** achieved across the active production interval.

---

## Visual Results

![Model Results](result.png)

1. **DTS Temperature Profile (Top-Left):** Continuous temperature log down to the perforation base (**1648 m**), capturing the geothermal gradient (**~0.015 °C/m**) and fluid entry cooling/heating signatures.
2. **Wellbore Flow Profile (Top-Right):** Predicted continuous flow curve (**0 → 5,200 BFPD**) tracking PLT ground-truth measurements from the upper wellbore down to zero-flow at the bottom toe.
3. **Training Convergence (Bottom-Left):** Smooth AdamW optimization with Cosine Annealing learning rate schedule over 200 epochs.
4. **Actual vs. Predicted Parity (Bottom-Right):** Strong correlation along the ideal `y = x` line across the active producing zone (**R² = 0.9412**).

---

## Thermodynamic Feature Engineering

Raw fiber-optic DTS data captures subtle temperature shifts caused by fluid entry and convective heat transport. To guide the model with physical context, raw temperature readings are expanded into five core features:

- **Spatial Temperature Gradient (`dT/dz`):** First derivative of temperature with respect to depth, pinpointing fluid entry zones.
- **Thermal Curvature (`d²T/dz²`):** Second derivative along the depth axis, highlighting boundary transitions and rate-of-change inflections.
- **Rolling Spatial Statistics (`T_roll_mean`, `T_roll_std`):** 11-point centered moving windows that smooth out high-frequency optical noise while keeping true thermal anomalies intact.

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dT_dz"] = np.gradient(df["Temp"].values, df["Depth"].values)
    df["d2T_dz2"] = np.gradient(df["dT_dz"].values, df["Depth"].values)
    df["T_roll_mean"] = df["Temp"].rolling(11, min_periods=1, center=True).mean()
    df["T_roll_std"] = df["Temp"].rolling(11, min_periods=1, center=True).std().fillna(0)
    return df
```

---

## Architecture Evolution

```text
Input Window [Batch, Seq=8, Features=5]
               │
               ▼
   [ Conv1D (16 ch, k=3) + BatchNorm1d + GELU ]
               │
               ▼
   [ Conv1D (32 ch, k=3) + BatchNorm1d + GELU ]
               │
               ▼
   [ 2-Layer LSTM (Hidden=32, Dropout=0.3) ]
               │
               ▼
   [ Dense (32 → 16) + GELU + Dropout(0.3) ]
               │
               ▼
      [ Linear (16 → 1) ]
               │
               ▼
   [ Physics Boundary Enforcement: Flow(depth >= 1648 m) = 0 ]
```

1. **Baseline (`Conv1D-BiLSTM` + Lion):** Suffered from gradient instability on small depth windows.
2. **Intermediate (`ConvLSTM` without boundary rules):** Converged well on training depths but over-predicted flow (~700 BFPD) in the stagnant bottom zone.
3. **Final Verified Model (`ConvLSTM` + Physics Constraints + AdamW):** Compact **19,361-parameter** network trained with gradient clipping (`≤ 1.0`), weight decay (`1e-4`), Cosine Annealing down to `1e-7`, and physical zero-flow enforcement below the perforation base.

---

## Repository Structure

```text
├── main.py                          # Production & synthetic demo training pipeline
├── requirements.txt                 # Python package dependencies
├── result.png                       # 4-panel model evaluation summary figure
├── DTS_FlowRate_Results.png         # High-resolution supplementary plot
├── Research_Task.pdf                # Full Google Colab experimental log & debugging history
├── DTS_FlowRate_Full_Notebook.pdf   # Complete notebook report on flow profile characteristics
├── Project_Presentation.pdf         # Technical presentation slides (Waverity Assessment)
├── LICENSE                          # MIT License
└── .github/
    └── workflows/
        └── python-app.yml           # Automated syntax & end-to-end demo CI check
```

---

## Getting Started

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/aliyev-toghrul/DTS-Thermodynamic-ML-Modeling.git
cd DTS-Thermodynamic-ML-Modeling
pip install -r requirements.txt
```

### 2. Quick Demo Run (No External Data Needed)
Run the full pipeline immediately using the built-in synthetic wellbore generator:
```bash
python main.py --demo --epochs 100
```
This generates realistic June 25 wellbore telemetry, trains the ConvLSTM model, prints evaluation metrics, and outputs `result.png`.

### 3. Training on Field Telemetry
When running with the full field dataset (`Stage2_data.zip` extracted to `./data/extracted/`):
```bash
python main.py --data-dir ./data/extracted/ --epochs 200 --batch-size 32 --lr 0.001
```

---

## Acknowledgements

Developed in collaboration with and under the mentorship of **Artughrul Qayibov (Data Science & ML Instructor, AI Researcher & ML Innovator)** at Baku Engineering University.
