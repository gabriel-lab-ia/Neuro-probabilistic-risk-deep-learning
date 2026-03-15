# Architecture

The current project is organized as a research prototype instead of a clinical product.

## Modeling stack

- `TabularBackbone`: deep MLP with `LayerNorm`, `GELU` and dropout for biomarker or clinical tables.
- `TemporalConvBranch`: lightweight 1D convolutional encoder for EEG-like or physiological sequences.
- `FusionModule`: simple concatenation-plus-projection block for future multimodal expansion.
- `NeuroRiskClassifier`: shared classifier head that outputs logits for multiclass risk stratification.

## Uncertainty and calibration

- Deterministic inference uses logits followed by `softmax`.
- MC Dropout keeps dropout layers active at inference time to sample stochastic subnetworks.
- The prototype computes mean predictive probabilities, class-wise variance, predictive entropy and mutual information.
- Temperature scaling is fit on the validation split as a lightweight post-hoc calibration stage.

## Data strategy

- The MVP uses synthetic multimodal placeholder data saved under `data/processed/neuro_risk_placeholder/`.
- Synthetic generation mixes tabular covariates and time-series patterns with overlapping class boundaries to make uncertainty visible.
- The pipeline is ready to swap in real datasets later without changing the main training and inference abstractions.
