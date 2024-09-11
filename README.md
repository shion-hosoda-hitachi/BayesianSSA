# BayesianSSA
## Description
BayesianSSA is a Bayesian statistical model based on structural sensitivity analysis (SSA).
BayesianSSA is implemented as a python Class.
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
## Usage
### Reproducing paper
You can easily reproduce the result of [the BayesianSSA paper](https://doi.org/10.1186/s12859-024-05921-4) using notebook/conduct_BayesianSSA.ipynb and notebook/reproduce_paper_figures_and_tables.ipynb.
data/SSDesignSuppTable.csv is from the following paper:
> Toya, Yoshihiro, Takanori Shiraki, and Hiroshi Shimizu. "Ssdesign: Computational metabolic pathway design based on flux variability using elementary flux modes." Biotechnology and bioengineering 112.4 (2015): 759-768.
This metabolic network information was corrected on the basis of communications with the authors.
### Applying BayesianSSA to your dataset
If you have your own dataset that includes metabolic network information and perturbation data, you can easily apply BayesianSSA to them.
You parse the network information as follows:
```
import bayesianssa
import numpy as np
import pandas as pd
parser = bayesianssa.ReactionFormulaParser(
    '/path/to/your_metabolic_network.csv',
    ignored_reactions=['BIO'],
    ignored_metabolites=['Export'])
parser.parse()
```
See data/SSDesignSuppTable.csv for the format of the dataset.
Here, you can specify reactions/metabolites that should be ignored.
And then, you obtain its stoichiometric matrix as follows:
```
nu = parser.nu
nu = bayesianssa.remove_non_flow_metabolites(nu)
```
```bayesianssa.remove_non_flow_metabolites``` removes metabolites that are not used or produced. Such metabolites make BayesianSSA impossible to be applied.
You specify the hyperparameters and the number of iterations as follows:
```
n_iter = 10000
a = 3
b = 1
```
Specifying the dependence of reactions to metabolites as ```indices```, you can make ```BayesianSSA``` object and conduct an initial computation by ```run``` method as follows:
```
indices = np.argwhere(nu.T.to_numpy() < 0)
np.random.seed(0)
bssa = bayesianssa.BayesianSSA(nu, indices, a, b,
                               n_iter=n_iter, verbose=False)
bssa.run()
```
Here, this dependence is specified on the basis of substrate-reaction relationships.
You perform computation of the posterior distribution using your own perturbation dataset as follows:
```
data = pd.read_csv('/path/to/your_perturbation_data.csv')
ex_results = {}
ex_results['target'] = data[['Observation target', 'Perturbation target']].to_numpy()
ex_results['up/down'] = data['Qualitative response'].to_numpy()
bssa.update_distributions(ex_results)
```
See data/real_perturbation_data.csv for the format of the dataset.
Now, you have obtained the positivity confidence value
```
bssa.confidence['positive']
```
and the predictive probability
```
pred_prob = bssa.calculate_predictive_prob('SUCCt', 'CS', 1)
```
```pred_prob``` shows the predictive probability that the SUCCt flux increases when the CS reaction is up-regulated.
