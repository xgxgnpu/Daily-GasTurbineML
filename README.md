# Daily-GasTurbineML

> Daily tracking of Physics-Informed Neural Networks & Deep Learning papers for Gas Turbines and Turbomachinery.

[中文版](README_CN.md)

## Author

**Xiong Xiong (熊雄)**

- Northwestern Polytechnical University (NWPU)
- Research interests: AI4PDE, Physics-Informed Deep Learning, Data-Driven Discovery
- Email: xiongxiongnwpu@mail.nwpu.edu.cn
- [Google Scholar](https://scholar.google.com/citations?user=j1M9tkwAAAAJ&hl=zh-CN&oi=sra)
- [ResearchGate](https://www.researchgate.net/profile/Xiong-Xiong-19?ev=hdr_xprf)
- [Physics-informed-vibe-coding](https://github.com/xgxgnpu/Physics-informed-vibe-coding)

---


## Overview

This repository independently tracks the latest research on **machine learning and physics-informed methods applied to gas turbines and turbomachinery**. It covers **71 papers** across 6 categories.

Topics include: PINN for turbomachinery flows, surrogate models for performance prediction, deep learning for aerodynamic design, CFD-ML hybrid methods, combustion ML, and gas turbine health monitoring.

> **Quick Navigation:**
>
> [1. PINN-GasTurbine](#1-pinn-for-gas-turbines--turbomachinery) | [2. Surrogate](#2-surrogate-models-for-gas-turbine-performance) | [3. DL-Aero](#3-deep-learning-for-turbomachinery-aerodynamics) | [4. CFD-ML](#4-cfd-ml-for-turbomachinery-flows) | [5. Combustion](#5-ml-for-combustion--thermal-analysis) | [6. Health](#6-gas-turbine-health-monitoring--fault-diagnosis)

---

### 1. PINN for Gas Turbines & Turbomachinery

Physics-Informed Neural Networks applied directly to gas turbine and turbomachinery simulations:

- PINN for compressor and turbine blade aerodynamics
- Physics-constrained flow field prediction in blade passages
- Inverse design and parameter identification in turbomachinery
- Multi-fidelity PINN combining CFD data with physical laws

### 2. Surrogate Models for Gas Turbine Performance

Data-driven surrogate models replacing expensive simulations for performance prediction:

- Neural network surrogate for gas turbine cycle analysis
- Multi-point compressor/turbine map modeling
- Gaussian process and deep learning for off-design performance
- Reduced-order modeling for engine simulation acceleration

### 3. Deep Learning for Turbomachinery Aerodynamics

Deep learning applied to aerodynamic analysis and design in turbomachinery:

- CNN/Transformer for blade flow field prediction
- Generative models for turbine blade shape optimization
- Graph neural networks for cascade aerodynamic analysis
- Attention-based models for wake interaction and unsteady flows

### 4. CFD-ML for Turbomachinery Flows

Machine learning augmenting or replacing CFD in turbomachinery:

- ML-enhanced turbulence closure models (RANS, LES)
- Physics-informed data assimilation for turbomachinery CFD
- Neural operators for fast flow field prediction
- ML-corrected RANS for secondary flows in blade passages

### 5. ML for Combustion & Thermal Analysis

Machine learning for combustion chambers and thermal management in gas turbines:

- Neural network combustion models for ignition and flame dynamics
- ML-based NOx and emission prediction
- Deep learning for turbine cooling design and optimization
- Thermal field reconstruction from sparse sensor data

### 6. Gas Turbine Health Monitoring & Fault Diagnosis

Deep learning for condition monitoring, fault detection and prognostics in gas turbines:

- LSTM/CNN for gas turbine degradation prediction
- Anomaly detection in vibration and temperature sensor data
- Remaining useful life estimation for turbine components
- Transfer learning for cross-engine fault diagnosis

---

**Total papers**: 71

## Usage

```bash
pip install arxiv requests

# Fetch latest papers (10 per category)
python fetch_today.py

# Specify number per category
python fetch_today.py --per_category 15

# Append N new papers per category without overwriting
python fetch_today.py --append 10

# Skip code search for faster execution
python fetch_today.py --no_code_search
```

## License

MIT
