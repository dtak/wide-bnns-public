# Wide Mean-Field Bayesian Neural Networks Ignore the Data

This repository contains the code necessary to reproduce the experiments in our paper, *Wide Mean-Field Bayesian Neural Networks Ignore the Data* (AISTATS 2022).

**Bayesian neural networks (BNNs)** combine the expressive power of deep learning with the advantages of Bayesian formalism. In recent years, the analysis of wide, deep BNNs has provided theoretical insight into their priors and posteriors. However, we have no analogous insight into their posteriors under approximate inference. In this work, **we show that mean-field variational inference entirely fails to model the data** when the network width is large and the activation function is odd (e.g., tanh). We also show this need not be true if the activation function is not odd (e.g., ReLU).

## Reproducibility 

To reproduce our results you'll first need to train BNNs. For more instructions, open `commands.txt` in each of the following folders, each corresponding to one or more figures:

 - `experiments/experiment_1/work`: Figures 1 and 4
 - `experiments/experiment_2/work`: Figure 2
 - `experiments/experiment_3/work`: Figures 5 and 6

After training the models, run `experiments/figures/work/make_figures.py`. 