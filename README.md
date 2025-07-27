# Machine Learning-Assisted Analysis of the Catassembly Triad Framework

This repository contains the source code for the Bayesian Gaussian Mixture Model (BGMM) and Gaussian Process (GP) classification analysis described in the paper "Catassembly Triad: A Catalytic Framework for Enantioselective Chiral Molecular Assembly". The implementation quantitatively validates catalyst efficacy prediction through triad descriptors (attachability, controllability, detachability).

## Key Features
- Iterative BGMM clustering with automatic component selection
- ELBO convergence analysis for model stability
- 3D probability mapping with Gaussian Processes
- Visualization of cluster evolution and classification confidence
- Experimental validation

## Installation
1. Create virtual environment:
```bash
python -m venv triad-env
source triad-env/bin/activate  # Linux/Mac
triad-env\Scripts\activate    # Windows
```

2. Install dependencies:
pip install numpy pandas scikit-learn matplotlib plotly kaleido
or:
uv add numpy

## Project Structure
├── data/                    # Input datasets
│   ├── data.csv               # Primary triad descriptors
│   └── data - forplot.csv     # BGMM labeled data for GP classification
├── src/
│   ├── bgmm.py                # Bayesian GMM implementation
│   ├── elbo_plot.py           # ELBO convergence visualization
│   └── gp_plotly_3D.py        # 3D probability mapping
├── reports/                 # Generated analysis reports
├── output/                  # Visualizations and plots
└── pyproject.toml           # Dependency configuration

## Usage
1. Run BGMM Analysis
```bash
python src/bgmm.py
```
Generates:
    Cluster reports in reports/
    3D visualizations in output/

2. Generate ELBO Plot
```bash
python src/elbo_plot.py
```
Generates:
    ELBO visualizations in output/

3. Create 3D Probability Map
```bash
python src/gp_plotly_3D.py
```
Generates:
    GP 3D visualizations in output/
