from consts import DIR_DATA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


from data import (
    abalone,
    auto_mpg,
    airquality,
    california_housing,
    concrete,
    cpu_small,
    higgs,
    eeg_eye_state,
    superconductor,
    qm9,
    geckoq,
    musk,
    community_crime,
    diabetes,
)


FEATURE_CREATION_REG = {
    "results_filename": "features_reg",
    "embedding_dims": [1, 2, 3, 4],
    "repeats": 20,
    "embedding_config": {
        "Original": {},
        "GBMAP": {
            "penalty_ridge": 1e-3,
            "optim_maxiter": 400,
            "optim_tol": 1e-3,
            "softplus_scale": 10,
            "n_fit": 10,
        },
        "PCA": {},
        "CCA": {},
        "PLS": {},
        "IVIS": {
            "supervision_metric": "mae",
            "epochs": 400,
            "n_epochs_without_progress": 2,
            "verbose": 0,
        },
    },
    "model_configs": {
        "LM": {"model": LinearRegression, "params": {}},
        "DT": {"model": DecisionTreeRegressor, "params": {"max_depth": 4}},
        "KNN": {"model": KNeighborsRegressor, "params": {"n_neighbors": 10}},
    },
}


FEATURE_CREATION_CLS = {
    "results_filename": "features_cls",
    "embedding_dims": [1, 2, 3, 4],
    "repeats": 20,
    "embedding_config": {
        "Original": {},
        "GBMAP": {
            "penalty_ridge": 1e-3,
            "optim_maxiter": 400,
            "optim_tol": 1e-3,
            "softplus_scale": 10,
            "n_fit": 10,
        },
        "PCA": {},
        "LDA": {},
        "CCA": {},
        "PLS": {},
        "LOL": {"svd_solver": "full"},
        "IVIS": {
            "epochs": 400,
            "n_epochs_without_progress": 2,
            "verbose": 0,
        },
    },
    "model_configs": {
        "LM": {"model": LogisticRegression, "params": {"penalty": "l2", "C": 1000}},
        "DT": {"model": DecisionTreeClassifier, "params": {"max_depth": 4}},
        "KNN": {"model": KNeighborsClassifier, "params": {"n_neighbors": 10}},
    },
}

SCALING_N = {
    "n_sizes": [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
    "p_sizes": [25],
}

SCALING_P = {"n_sizes": [100_000], "p_sizes": [100, 200, 400, 800, 1_600, 3_200]}

XGBOOST_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 2,
    "subsample": 0.9,
}


def features_reg_datasets():

    datasets = {
        "california": {
            "loader": california_housing,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "concrete": {
            "loader": concrete,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "superconductor": {
            "loader": superconductor,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "qm9": {
            "loader": qm9,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "geckoq": {
            "loader": geckoq,
            "params": {"scale_data": False, "center_y": False, "data_home": DIR_DATA},
        },
        "cpu-small": {
            "loader": cpu_small,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "abalone": {
            "loader": abalone,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "autompg": {
            "loader": auto_mpg,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "airquality": {
            "loader": airquality,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "community-crime": {
            "loader": community_crime,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
    }
    return datasets


def features_cls_datasets():
    """Lazy loader dictionary for regression datasets

    Returns:
        Dictionary: Dict with data loaders and loader params
    """
    datasets = {
        "higgs": {
            "loader": higgs,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "musk": {
            "loader": musk,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "eeg-eye-state": {
            "loader": eeg_eye_state,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
        "diabetes": {
            "loader": diabetes,
            "params": {"scale_data": False, "data_home": DIR_DATA},
        },
    }
    return datasets


color_palette = {
    "Original": "grey",
    "XGBOOST": "grey",
    "GBMAP": "black",
    "PCA": "C0",
    "IVIS": "C1",
    "LOL": "C2",
}
