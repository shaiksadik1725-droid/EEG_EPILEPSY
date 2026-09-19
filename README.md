# EEG Epilepsy Feature Extraction & Signal Analysis

A research-oriented EEG analysis application for exploring signal preprocessing and feature extraction, with particular focus on seizure and epilepsy-related workflows.

## Overview

The project provides a Flask-based backend and browser interface for EEG signal analysis. It organizes features across time-domain, frequency-domain, time-frequency, nonlinear, and connectivity categories.

## Feature Groups

### Time Domain
- Hjorth parameters
- Mean and variance
- Skewness and kurtosis
- RMS
- Zero-crossing rate
- Entropy-based features
- Fractal-dimension features

### Frequency Domain
- Absolute and relative band power
- Welch PSD
- FFT-based PSD
- Spectral entropy
- Peak frequency
- Band-power ratios
- Spectral edge frequency

### Time-Frequency
- STFT
- DWT
- CWT
- Wavelet packet features
- EMD / IMF-related features

### Nonlinear & Connectivity
- Lyapunov-related measures
- Correlation dimension
- DFA
- Permutation entropy
- Hurst exponent
- Coherence
- Phase Locking Value
- Cross-correlation and graph-oriented metrics

## EEG Bands

The application works with common EEG bands including delta, theta, alpha, beta, gamma, and high-frequency activity.

## Technology Stack

- Python
- Flask
- HTML / CSS / JavaScript
- Numerical signal-processing logic

## Project Structure

```text
EEG_EPILEPSY/
├── app.py
├── index.html
└── dataset/
```

## Run Locally

```bash
pip install flask
python app.py
```

Then open the local address shown by the Flask server.

## Academic Use

This repository is intended as a signal-processing and research prototype. It is not a clinical diagnostic system.

## Future Improvements

- Add validated EEG datasets and reproducible experiments
- Integrate SciPy/MNE preprocessing pipelines
- Add trained seizure-classification models
- Add artifact-removal methods
- Add automated reports and exportable feature tables
- Validate features using clinically annotated recordings

## Author

**Sadik Shaik**

Computer Engineering / AI & Embedded Systems Projects
