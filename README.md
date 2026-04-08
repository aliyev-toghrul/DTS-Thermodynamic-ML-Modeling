# Wellbore Flow Prediction via DTS Telemetry
## Technical Assessment: Algorithm Developer (Research Assistant)

### Executive Summary
This project implements an end-to-end Machine Learning pipeline to predict fluid flow-rate contributions using Distributed Temperature Sensing (DTS) data. The solution extrapolates stationary Production Logging Tool (PLT) calibration points across a 11,000m wellbore.

### Key Engineering Challenges Solved:
1. **Big Data Ingestion:** Developed a manual line-streaming parser to process a 1GB+ CSV, bypassing C-engine buffer overflows in standard Pandas.
2. **Legacy Data Extraction:** Built a Regex-based parser to handle malformed, space-separated strings in legacy Excel exports.
3. **Asynchronous Sensor Fusion:** Implemented a `merge_asof` nearest-neighbor join with a 5.0m spatial tolerance to synchronize DTS and PLT datasets.
4. **Physics-Informed Feature Engineering:** Derived a first-order numerical gradient ($dT/dz$) to capture the thermodynamic relationship between temperature and flow.

### Tech Stack
- **Language:** Python 3.12
- **Libraries:** Pandas, Scikit-Learn, Matplotlib, Regex
- **Model:** Random Forest Regressor
