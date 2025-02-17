# GBMAP (Gradient Boosting Mapping)
Gradient Boosting Mapping is and nonlinear dimensionality reduction and feature creation method. This repository contains the Python code for GBMAP and the experiments in the paper:

> Patron, A., Savvides, R., Franzon, L., Luu, H.P.H., Puolamäki, K. (2025). Fast and Understandable Nonlinear Supervised Dimensionality Reduction. In: Pedreschi, D., Monreale, A., Guidotti, R., Pellungrini, R., Naretto, F. (eds) Discovery Science. DS 2024. Lecture Notes in Computer Science(), vol 15243. Springer, Cham. https://doi.org/10.1007/978-3-031-78977-9_25

## Data
Most of the datasets are from [OpenML](https://www.openml.org/) and will be downloaded as needed, but GeckoQ has to be downloaded separately.

The GeckoQ data can be downloaded from [Fairdata repository](https://doi.org/10.23729/022475cc-e527-41a9-bbc0-0113923cf04c), you'll only have to download the `Dataframe.csv` and place it to `experiments/data`.


## Instructions for installing and running the experiments
Script `run.sh` contains explicit instructions how to install and run the experiments. The results for the experiments are placed to `experiments/results` and figures to `experiments/figures`.

# Citation
If you use the code, please cite the paper:
```
@incollection{patron2025Fast,
  title = {Fast and {{Understandable Nonlinear Supervised Dimensionality Reduction}}},
  booktitle = {Discovery {{Science}}},
  author = {Patron, Anri and Savvides, Rafael and Franzon, Lauri and Luu, Hoang Phuc Hau and Puolam{\"a}ki, Kai},
  year = {2025},
  volume = {15243},
  pages = {385--400},
  publisher = {Springer Nature Switzerland},
  address = {Cham},
  doi = {10.1007/978-3-031-78977-9_25},
}
```
