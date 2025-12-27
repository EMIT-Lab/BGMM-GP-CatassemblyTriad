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
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows
```

2. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib plotly kaleido
```
or:
```bash
uv add numpy
```

## Project Structure
```bash
├── data/                    # Input datasets
│   └── data_sorted_by_No_in_article.csv
├── src/
│   ├── bgmm.py                # Bayesian GMM implementation
│   ├── elbo_plot.py           # ELBO convergence visualization
│   └── gp_plotly_3D.py        # 3D probability mapping
├── reports/                 # Generated analysis reports
├── output/                  # Visualizations and plots
└── pyproject.toml           # Dependency configuration
```

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

## Citation
If you find this code useful in your research, please cite the following paper:
```bibtex
@article{Li2025Catassembly,
    title = {Catassembly Triad: A Catalytic Framework for Enantioselective Chiral Molecular Assembly},
    author = {Li, Zhihao and Huang, Xuehai and Jiang, Yansong and Zeng, Wei and Zou, Ding and Dong, Xue and Yang, Liulin and Cao, Xiaoyu and Tian, Zhongqun and Wang, Yu},
    journal = {Journal of the American Chemical Society},
    year = {2025},
    doi = {10.1021/jacs.5c16840},
    url = {https://doi.org/10.1021/jacs.5c16840}
}
