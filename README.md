# DTS-Thermodynamic-ML-Modeling
# DTS Thermodynamic Flow Modeling
## Algorithm Developer Technical Task - eiGroup LLC

### Project Overview
This project involves predicting wellbore flow-rate contributions using Distributed Temperature Sensing (DTS) data. 

### Technical Challenges Overcome:
1. **Big Data Ingestion:** Developed a manual line-streaming parser in Python to process a 1GB+ CSV dataset, bypassing C-engine buffer overflows that occur in standard Pandas implementations.
2. **Legacy Data Parsing:** Implemented Regex-based extraction to clean malformed legacy Excel strings into structured numeric arrays.
3. **Asynchronous Sensor Alignment:** Utilized `merge_asof` with nearest-neighbor heuristics to synchronize high-frequency DTS telemetry with discrete PLT calibration points.
4. **Feature Engineering:** Derived a first-order numerical gradient ($dT/dz$) to capture thermodynamic signatures of fluid movement.

### Tech Stack:
- Python (Pandas, NumPy, Scikit-Learn)
- Machine Learning: Random Forest Regression
- Visualization: Matplotlib
