# Hierarchical Bayesian Modeling of Numerosity Estimation in Parkinson's Diseases Patients with Hallucinations

This repository contains code for the decision making course exam (Cognitive Science Aarhus University 2026).
I constructed a bayesian model in PyMC and fit that to the data obtained in the study by Albert et al. (2024).

**Dataset**:
Albert, L. P. (2024). Numerosity estimation of virtual humans as a digital-robotic marker for hallucinations in Parkinson's disease [Data set]. 
In Nature Communications (Vol. 15, Number 1, p. 1905). Zenodo. https://doi.org/10.5281/zenodo.10511579

Code was run in jupyter notebooks. You find them in the directory /notebooks.
- fit_model_1.ipynb : Fits the main model, and computes all analysis plots and tables as seen in the paper.
- model_comp.ipynb : Fits the main model and a null model and uses LOO for model comparison.
- recovery.ipynb : Performs sensitivity analysis across a range of parameters.

/src contains the PyMC models in model.py and other main functions imported in notebooks.

## Setup .venv (Virtual Environment)

I use the pacakge manager **uv** to setup the virtual environment.
If you do not already have uv installed we refer to [https://docs.astral.sh/uv/getting-started/installation/]
For example try, ``curl -LsSf https://astral.sh/uv/install.sh | sh`` or ``brew install uv``.

You can then run,
```bash
uv sync
```
and activate .venv
```
source .venv/bin/activate
```
