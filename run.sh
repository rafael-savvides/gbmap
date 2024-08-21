#!/bin/bash
# This script runs all of the experiments
# Note that this script is not designed to be run from start to finish, but to show how to reproduce the results

# create a conda enviroment
conda create -n gbmap python=3.10
conda activate gbmap

# install packages
pip install -r requirements.txt
# install gbmap
pip install .

cd experiments

# download data (GeckoQ needs to be downloaded by hand (see README.md))
python data.py

# scaling
for i in {0..9}; do
    python scaling.py --method GBMAP -r $i
    python scaling.py --method PCA -r $i
    python scaling.py --method LOL -r $i
    python scaling.py --method IVIS -r $i
done

# nfit
python run_nfit.py -r 100

# feature creation experiment
python feature_creation.py -c reg -p feature_ex -m LM
python feature_creation.py -c reg -p feature_ex -m DT
python feature_creation.py -c reg -p feature_ex -m KNN

python feature_creation.py -c cls -p feature_ex -m LM
python feature_creation.py -c cls -p feature_ex -m DT
python feature_creation.py -c cls -p feature_ex -m KNN

# low-data example
python feature_creation.py -d superconductor -p low_data_superconductor -m LM -n 6000

# make figures
python create_intro_figs.py

python create_scaling_figs.py

python run_nfit.py -r 100 -s

python create_feature_tab.py

python create_feature_figs.py -f results/features_ex -l

python create_gbmap_examples.py

python create_feature_figs.py -f results/low_data_superconductor/ -d low_data