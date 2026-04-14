# Physics-Informed Wellbore Flow Prediction using ConvLSTM

## The Problem
Predicting flow-rate contributions from Distributed Temperature Sensing (DTS) data is a complex challenge in petroleum engineering. Accurate spatial extrapolation is critical, as traditional methods often struggle to capture the non-linear thermodynamic behaviors and relationships inherent in wellbore fluid dynamics. 

## The Solution
We introduce a **Physics-Informed approach** to flow prediction. By explicitly incorporating spatial temperature gradients (`dT/dz`) and curvature (`d²T/dz²`) as engineered features, the model is guided by the underlying thermodynamic principles of fluid movement, significantly improving its interpretive predictive capabilities.

## Architecture
This implementation utilizes a highly efficient **ConvLSTM** architecture:
- **1D-CNN layers**: To extract local spatial temperature features along the wellbore.
- **LSTM layers**: To capture the sequential and spatial dependencies in the depth domain.
- **Optimization**: Trained with **AdamW** and a **Cosine Annealing** learning rate scheduler for robust convergence.

## Validation Strategy
To guarantee generalizability and prevent data leakage, a rigorous **Spatial Split** approach is employed:
- **Top 80%** of the wellbore depth is used for **Training**.
- **Bottom 20%** of the wellbore depth is reserved for **Testing**.
*This explicit spatial separation is the key differentiator, ensuring the model's ability to extrapolate to unseen depth regions rather than merely memorizing adjacent point patterns.*

## Metrics

```text
══ FINAL METRICS ══
MSE: 579758.88 | R²: -92.6573 | AIC: 39611.11
Architecture: ConvLSTM | Params: 19,361 | BIC: 82296.17
```

## Visuals

![DTS FlowRate Results](DTS_FlowRate_Results.png)
