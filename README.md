# GBMAP (Gradient Boosting Mapping)
Gradient Boosting Mapping is and nonlinear dimensionality reduction and feature creation method. This repository contains the Python code for GBMAP and the Discovery Science experiments. The code has been tested on Python 3.10.

Accepted to Discovery Science 2024, meanwhile check out a previous preprint http://arxiv.org/abs/2405.08486.

## Installing

Here's how to install GBMAP as a python package (required for running the experiments) and the requirements.
```bash
cd gbmap
pip install -e .
pip install -r requirements.txt
```

## Data
Most of the datasets are from [OpenML](https://www.openml.org/), but GeckoQ has to be downloaded separately.

The GeckoQ data can be downloaded from [Fairdata repository](https://doi.org/10.23729/022475cc-e527-41a9-bbc0-0113923cf04c)
Head to the Fairdata repository (link above) and download the `Dataframe.csv` and place it to `gbmap/experiments/ds/data`.
