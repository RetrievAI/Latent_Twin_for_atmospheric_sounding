# Latent Twins for Atmospheric Sounding

This repository provides a Python implementation of the **Latent Twins architecture** for data-driven atmospheric radiative transfer inversion using hyperspectral infrared observations.

The framework learns forward and inverse relationships between atmospheric states and radiative observations through coupled autoencoders and linear mappings between their latent spaces.

The implementation also includes auxiliary neural networks for correcting inconsistencies in retrieved liquid- and ice-cloud microphysical properties.

## Methodology

The framework consists of two main stages.

### 1. Latent Twins training

Two fully connected autoencoders learn compressed representations of:

- **Atmospheric states (X):** surface temperature, atmospheric profiles, surface emissivity, and cloud microphysical properties.
- **Observations (Y):** geolocation, temporal information, pressure profiles, and infrared radiance spectra.

Two linear mappings connect the learned latent spaces, enabling:

- Atmospheric-state reconstruction
- Observation reconstruction
- Forward prediction (X → Y)
- Inverse retrieval (Y → X)

The models are jointly trained using a weighted combination of four Mean Squared Error loss terms.

### 2. Cloud microphysical correction

After the main training stage, the Latent Twins models are frozen.

Two additional neural networks are trained to improve the consistency of retrieved cloud properties:

- **Liquid-cloud network:** corrects liquid water content (LWC) and liquid effective radius (REL).
- **Ice-cloud network:** corrects ice water content (IWC) and ice effective radius (REI).

During inference, corrections are applied only where inconsistencies between cloud water content and effective radius are detected.

## Repository Contents

The main Jupyter notebook implements:

1. Input data preparation and validation
2. Autoencoder and latent-space mapping architectures
3. Joint training of the main models
4. Freezing of pretrained models
5. Cloud correction architectures and loss functions
6. Training of liquid- and ice-cloud corrective networks
7. Inference, evaluation, and result saving

## Requirements

The implementation uses Python and the following libraries:

- PyTorch
- NumPy
- SciPy
- Jupyter Notebook

Install the main dependencies using:

```bash
pip install torch numpy scipy notebook
```

A CUDA-compatible GPU can be used for the main training stage. The auxiliary correction stage runs on CPU in the provided implementation.

## Input Data

The notebook requires four preprocessed and normalized arrays:

```python
X_train
Y_train
X_test
Y_test
```

All arrays must be two-dimensional, with samples arranged in rows.

### Atmospheric states (X)

The atmospheric state contains 722 variables:

| Variable | Dimension |
|---|---:|
| Skin temperature | 1 |
| Temperature profile | 60 |
| Water vapour profile | 60 |
| Ozone profile | 60 |
| Surface emissivity | 301 |
| Liquid water content | 60 |
| Liquid effective radius | 60 |
| Ice water content | 60 |
| Ice effective radius | 60 |

### Observations (Y)

The observation vector contains:

- Longitude and latitude
- Month and hour
- Atmospheric pressure at 60 levels
- Infrared radiance spectrum

All input quantities must be normalized before being passed to the networks.

**Note:** Data loading, preprocessing, and normalization are not included in the notebook. Users must provide the four arrays before running the training procedure.

## Usage

1. Install the required Python libraries.
2. Open the Jupyter notebook.
3. Prepare the normalized training and test datasets.
4. Define `X_train`, `Y_train`, `X_test`, and `Y_test` in the notebook environment.
5. Configure the training parameters in the initial code cell.
6. Execute the notebook cells sequentially.

The main training configuration includes the number of epochs, batch size, learning rate, latent dimensions, and loss weights.

Existing model checkpoints can be loaded when available.

### Outputs

The notebook creates two directories:

- `model/`: trained neural network parameters and training history.
- `results/`: inference results saved in MATLAB `.mat` format.

The inference results include original atmospheric states and observations, autoencoder reconstructions, forward predictions, and inverse retrievals before and after cloud correction.

## Publications

If you use this implementation in your research, please cite the relevant publications:

**1. Latent Twins for all-sky atmospheric sounding**

*Latent twins: a framework for scene recognition and fast radiative transfer inversion in FORUM all-sky observations.*

Scientific Reports (2026).

https://doi.org/10.1038/s41598-026-57822-6

**2. Latent Twins for clear-sky retrieval**

*A latent twins architecture for robust clear-sky retrieval from IASI acquisitions.*

Proceedings of SPIE (2026).

https://doi.org/10.1117/12.3119073

Citation metadata are also provided in the `CITATION.cff` file.

## Acknowledgements

The notebook implementation was developed by Cristina Sgattoni, building on the original Latent Twins framework by Matthias Chung (2024).

## License

This project is distributed under the terms specified in the [LICENSE](LICENSE) file.

