from consts import RANDOM_SEED

from common import results_in_subdir_path

from experiment_configs import (
    FEATURE_CREATION_REG,
    FEATURE_CREATION_CLS,
    features_reg_datasets,
    features_cls_datasets,
    XGBOOST_PARAMS,
)

from gbmap.gbmap import GBMAP
from gbmap.common import loss_logistic, loss_quadratic

import os
import argparse
import traceback

import numpy as np
import pandas as pd

import jax

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical

from lol import LOL
from ivis import Ivis
from xgboost import XGBRegressor
from xgboost import XGBClassifier


def make_argparser():

    help_txt = """Select the experiment config from ['reg', 'cls'],
    where 'reg' is regression and 'cls' is classification.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="reg", help=help_txt)
    parser.add_argument(
        "-n",
        "--datapoints",
        type=int,
        default=None,
        help="Subsample datasets to have n datapoints and add 10 noise variables.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="LM",
        help="Supervised model configuration ['LM', 'DT', 'KNN'].",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="",
        help="Dataset. If empty, run all.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="features_ex",
        help="Directory name for the features experiment results.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do a test run with only one embedding dimension.",
    )
    parser.add_argument(
        "-bb",
        "--blackbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eval blackbox (XGBOOST) as 'features'.",
    )
    return parser


def get_features(
    method, params, n_features, X_train, y_train, X_test, is_reg, gbmap_key
):
    """Compute n_features (embedding) using given method.

    Args:
        method (str): Embedding/feature generation model or 'Original' for dummy embedding.
        params (Dictionary): Model params.
        n_features (int): How many features/dims for the embedding.
        X_train (array-like): Training samples.
        y_train (array-like): Groundtruth label.
        X_test (array-like): Test samples.
        is_reg (bool): is regression task.
        gbmap_key (jax.PRNGKey): jax key for gbmap.

    Raises:
        ValueError: Raised if the embedding method is uknown.

    Returns:
        tuple: Embedded train and test samples.
    """
    if method == "Original":
        # no transformation baseline
        if n_features != 1:
            raise ValueError(
                "Original method can return only all features (requested features {})".format(
                    n_features
                )
            )
        return X_train, X_test
    elif method == "GBMAP":
        model = GBMAP(n_boosts=n_features, random_state=gbmap_key, **params)
        model.fit(X_train, y_train)
    elif method == "PCA":
        model = PCA(n_components=n_features, **params)
        model.fit(X_train)
    elif method == "RP":
        model = SparseRandomProjection(n_components=n_features, **params)
        model.fit(X_train, y_train)
    elif method == "LDA":
        model = LinearDiscriminantAnalysis(n_components=n_features, **params)
        model.fit(X_train, y_train)
    elif method == "CCA":
        model = CCA(n_components=n_features, **params)
        model.fit(X_train, y_train)
    elif method == "PLS":
        model = PLSCanonical(n_components=n_features, **params)
        model.fit(X_train, y_train)
    elif method == "LOL":
        model = LOL(n_components=n_features, **params)
        model.fit(X_train, y_train)

        # test that LOL gives correct number of features
        lol_dims = model.transform(X_train[:2, :]).shape[1]
        if lol_dims != n_features:
            print(
                "WARNING: LOL gave d={} embedding (was expecting d={}).".format(
                    lol_dims, n_features
                )
            )
            # try to recover by asking n_components=n_features + 1
            model = LOL(n_components=n_features + 1)
            model.fit(X_train, y_train)
            # assert
            lol_dims = model.transform(X_train[:2, :]).shape[1]
            if lol_dims != n_features:
                raise ValueError("Recovery with n_components=(n_features + 1) failed.")
    elif method == "IVIS":
        if is_reg:
            y_ivis = y_train
        else:
            # transform -1,1 to 0,1 labels
            y_ivis = (y_train + 1) / 2
        try:
            model = Ivis(embedding_dims=n_features, **params)
            model.fit(X_train, y_ivis)
        except Exception:
            traceback.print_exc()
            raise ValueError("IVIS broke again.")

    else:
        raise ValueError("Got an unknown embedding methdod '{}'".format(method))

    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)

    return Z_train, Z_test


def add_feature_predict(
    X_train,
    X_test,
    y_train,
    y_test,
    is_reg,
    method,
    params,
    n_features,
    dataset_name,
    fold_idx,
    eval_model,
    eval_model_name,
    data_points,
    gbmap_key=None,
):
    """Trains GBMAP (or only baseline performance when method=="Original") and computes n_features features
      and evaluates the performance of baseline predictor + new features.

    Args:
        X_train (array-like): Train samples.
        X_test (array-like): Test samples.
        y_train (array-like): Train groundtruth.
        y_test (array-like): Test groundtruth.
        is_reg (bool): is classification or regression.
        method (str): 'GBMAP' or 'Original'.
        params (Dictionary): GBMAP params in a dict.
        n_features (int): number of GBMAP features (m).
        dataset_name (str): Dataset name.
        fold_idx (int): Index for repeats.
        eval_model (class): Model to evaluate the features
        eval_model_name (str): Model name.
        data_points (int): Number of datapoints in a dataset.
        gbmap_key (jax.PRNGKey): jax key for gbmap.

    Raises:
        ValueError: Raised if unknown method is provided.

    Returns:
        Dictionary: Results in a dict.
    """

    if method == "XGBOOST":
        if n_features != 1:
            raise ValueError(
                "XGBOOST can provide only one feature (requested features {})".format(
                    n_features
                )
            )
        # use black-box model for predictions
        if is_reg:
            model = XGBRegressor(**params)
        else:
            # transform -1,1 to 0,1 labels
            y_train = (y_train + 1) / 2
            y_test = (y_test + 1) / 2
            model = XGBClassifier(**params)

        model.fit(X_train, y_train)

        # faux eval if we directly use XGBOOST features as a feature
        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)
    else:
        # compute new features
        Z_train, Z_test = get_features(
            method=method,
            params=params,
            n_features=n_features,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            is_reg=is_reg,
            gbmap_key=gbmap_key,
        )

        # train model with features
        eval_model.fit(Z_train, y_train)

        # predictions
        yhat_train = eval_model.predict(Z_train)
        yhat_test = eval_model.predict(Z_test)

    # feature params
    feature_params_str = ", ".join(f"{k}={v}" for k, v in params.items())

    # eval predictions
    if is_reg:
        # average quadratic loss
        train_loss = loss_quadratic(y_train, yhat_train)
        test_loss = loss_quadratic(y_test, yhat_test)

    else:
        # average logistic loss
        train_loss = loss_logistic(y_train, yhat_train)
        test_loss = loss_logistic(y_test, yhat_test)

    results = {
        "dataset": dataset_name,
        "datapoints": data_points,
        "features_method": method,
        "n_features": n_features,
        "fold": fold_idx,
        "test_loss": test_loss,
        "train_loss": train_loss,
        "eval_model": eval_model_name,
        "feature_params": feature_params_str,
    }

    return results


def evaluate_models(
    embedding_params,
    datasets,
    embedding_dims_base,
    is_reg,
    save_path,
    model_config,
    subsample,
    repeats=10,
):
    """Main loop for evaluating additional GBMAP features.
    Updates a CSV file (in save_path) each time a new result is computed.

    Args:
        embedding_params (Dictionary): Params for GBMAP features/embeddings.
        datasets (Dictionary): Dictionary with dataset loaders and params.
        embedding_dims_base (list): list of how many embedding features to use.
        is_reg (bool): is regression or classification.
        save_path (str): Path for a CSV results file.
        model_config (dict): Configuration for a experiment.
        subsample (int or None): Number of training samples.
        repeats (int, optional): How many times repeat randomly split data and eval. Defaults to 10.

    Raises:
        ValueError: Should not be raised.
    """

    embedding_methods = embedding_params.keys()

    iter = 1

    np.random.seed(RANDOM_SEED)

    eval_model_name = model_config["model"].__name__

    for dataset_name in datasets.keys():
        data = datasets[dataset_name]["loader"](**datasets[dataset_name]["params"])
        if len(data) == 2:
            # dataloader returns only covariates and a groundtruth target
            X, y = data
        elif len(data) == 3:
            # X covariates, y groundtruth, y0 baseline predictor
            X, y, _ = data
            # no baseline predictor in dataset

        # extend the base embedding dims to extend to p + 2 (interval of 2)
        embedding_dims = embedding_dims_base.copy()
        embedding_dims = embedding_dims + list(
            range(np.max(embedding_dims) + 2, X.shape[1] + 2, 2)
        )

        n_experiments = len(datasets) * len(embedding_methods) * len(embedding_dims)

        if subsample is not None:
            # add noise features in data and subsample train data
            X_iid = np.random.standard_normal((X.shape[0], 10))
            X = np.hstack([X, X_iid])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=subsample
            )

        N = X.shape[0]

        for embedding_method in embedding_methods:
            embedding_paramsi = embedding_params.get(embedding_method, None)

            gbmap_key = None
            if embedding_method == "GBMAP":
                gbmap_key = jax.random.PRNGKey(RANDOM_SEED)

            for n_features in embedding_dims:
                if embedding_method == "IVIS" and n_features > 10:
                    print("n_features > 10, skipping IVIS EVAL.")
                    continue
                print(
                    "[Progress {i}/{n}]: Evaluating: dataset={d}, features={e}, dims={k}, model={m}".format(
                        i=iter,
                        n=n_experiments,
                        d=dataset_name,
                        e=embedding_method,
                        k=n_features,
                        m=eval_model_name,
                    )
                )
                # repeat split eval
                for i in range(repeats):
                    # model for evaluating the embedding/features
                    eval_model = model_config["model"](**model_config["params"])

                    if subsample is None:
                        # split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2
                        )

                    # preprocess data
                    scaler_x = StandardScaler()
                    X_train = scaler_x.fit_transform(X_train)
                    X_test = scaler_x.transform(X_test)

                    if is_reg:
                        scaler_y = StandardScaler()

                        y_train = scaler_y.fit_transform(y_train[:, None]).ravel()
                        y_test = scaler_y.transform(y_test[:, None]).ravel()

                    if embedding_method == "GBMAP":
                        _, gbmap_key = jax.random.split(gbmap_key)
                    try:
                        # find features/embedding and evaluate boosted predictor
                        res = add_feature_predict(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            is_reg=is_reg,
                            method=embedding_method,
                            params=embedding_paramsi,
                            n_features=n_features,
                            dataset_name=dataset_name,
                            fold_idx=i,
                            eval_model=eval_model,
                            eval_model_name=eval_model_name,
                            data_points=N,
                            gbmap_key=gbmap_key,
                        )
                    except ValueError:
                        # Encountered a ValueError while evaluating features
                        # this can happen for example if CCA is run with n_components > min(n_samples, n_features, n_targets)
                        traceback.print_exc()
                        print("Skipping...")
                        break

                    df = pd.DataFrame([res])  # dataframe with one row
                    print("####RESULTS####")
                    print(df)
                    # append results to a file, creates a new file if one does not exists
                    df.to_csv(
                        save_path,
                        mode="a",
                        header=not os.path.exists(save_path),
                        index=False,
                    )
                # end of repeats
                iter += 1


def main():
    parser = make_argparser()
    args = parser.parse_args()

    # experiment params
    is_reg = False

    if args.config == "reg":
        config_dict = FEATURE_CREATION_REG
        is_reg = True
    elif args.config == "cls":
        config_dict = FEATURE_CREATION_CLS
    else:
        raise ValueError("Unknown experiment config {}".format(args.config))

    embedding_dims = config_dict["embedding_dims"]
    eval_repeats = config_dict["repeats"]
    embedding_params = config_dict["embedding_config"]
    data_suffix = "all" if args.dataset == "" else args.dataset
    results_filename = "{}_{}_{}.csv".format(
        config_dict["results_filename"], args.model, data_suffix
    )

    model_configs = config_dict["model_configs"]
    model_config = model_configs[args.model]

    if args.blackbox:
        # add blackbox "feature creator"
        embedding_params["XGBOOST"] = XGBOOST_PARAMS

    if args.test:
        embedding_dims = [1]  # [1, 2, 3, 4]
        eval_repeats = 1

    # load datasets
    if is_reg:
        dataset_loaders = features_reg_datasets()

        datasets = [
            "concrete",
            "qm9",
            "california",
            "geckoq",
            "superconductor",
            "abalone",
            "autompg",
            "airquality",
            "cpu-small",
        ]
    else:
        dataset_loaders = features_cls_datasets()
        datasets = ["higgs", "musk", "eeg-eye-state"]

    datasets = datasets if args.dataset == "" else [args.dataset]
    ex_datasets = {dataset: dataset_loaders[dataset] for dataset in datasets}

    # get a path for results.csv file
    save_path = results_in_subdir_path(args.path, results_filename)

    # evaluate embedding methods
    evaluate_models(
        embedding_params=embedding_params,
        datasets=ex_datasets,
        embedding_dims_base=embedding_dims,
        is_reg=is_reg,
        save_path=save_path,
        repeats=eval_repeats,
        model_config=model_config,
        subsample=args.datapoints,
    )

    print("Results saved to {}".format(save_path))


if __name__ == "__main__":
    main()
