# BayesianSSA
## Description
BayesianSSA is a Bayesian statistical model based on structural sensitivity analysis (SSA).
BayesianSSA is implemented as a python Class.
## Usage
You can easily reproduce the result of [the BayesianSSA paper]() using notebook/conduct_BayesianSSA.ipynb and notebook/reproduce_paper_figures_and_tables.ipynb.
data/SSDesignSuppTable.csv is from the following paper:
> Toya, Yoshihiro, Takanori Shiraki, and Hiroshi Shimizu. "Ssdesign: Computational metabolic pathway design based on flux variability using elementary flux modes." Biotechnology and bioengineering 112.4 (2015): 759-768.
This network dataset was corrected on the basis of communications with the authors.
## Requirements
- Python (3.7.9)
- Python library
    - tqdm (4.64.0)
    - numpy (1.21.6)
    - sympy (1.10.1)
    - scipy (1.7.3)
    - pandas (1.3.5)
    - matplotlib (3.5.2)
    - seaborn (0.11.2)
- All experiments were validated on Ubuntu 20.04.4.
