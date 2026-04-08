# DTS Wellbore Thermodynamic Flow Modeling

## Project Overview
This repository contains a physics-informed Machine Learning pipeline engineered to model wellbore thermodynamic flow using Distributed Temperature Sensing (DTS) and Production Logging Tool (PLT) data. The project tackles the complexities of handling massive, high-resolution telemetry datasets and extrapolating sparse, stationary calibration points into a continuous 11,000m thermodynamic flow profile.

## Technical Challenges & Engineering Solutions

1. **Big Data Challenge & Memory Management**
   - **Challenge:** Processing a massive 1GB+ DTS telemetry file containing over 26 million rows caused memory fragmentation and C-engine buffer overflows using standard Pandas loading methods.
   - **Solution:** Bypassed the C-engine limitations by building a manual Python line-streaming parser that reliably handled the data pipeline without overwhelming memory.

2. **Legacy Data Extraction**
   - **Challenge:** Critical ground-truth data was trapped in malformed, space-separated strings within legacy Excel files.
   - **Solution:** Engineered a robust, custom Regex-based parser to reliably recover and reconstruct the numeric target arrays from the raw strings.

3. **Asynchronous Sensor Fusion**
   - **Challenge:** DTS and PLT sensors sample at asynchronous rhythms and irregular depths.
   - **Solution:** Synchronized the telemetry data using the `merge_asof` technique, implementing a nearest-neighbor heuristic with a strict 5.0m spatial tolerance to guarantee high-fidelity data fusion.

## Methodology

1. **Data Pipeline Architecture & Physics-Informed ML**
   - Engineered a first-order numerical thermal gradient ($dT/dz$) as a primary feature. Incorporating this physics-based numerical gradient intrinsically grounds the statistical model in thermodynamic reality.

2. **Predictive Modeling**
   - Developed a **Random Forest Regressor** to predict continuous thermodynamic flow parameters. The non-linear capabilities of the Random Forest were instrumental in extrapolating isolated, stationary calibration points into a continuous 11,000m active wellbore flow profile.

## Results
- Successfully established a continuous flow profile across the entire 11,000m string.
- (Refer to `Research_Outcome.png` in the repository for a visual representation of the prediction constraints and modeled flow dynamics).
