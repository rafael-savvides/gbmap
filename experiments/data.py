import os
import requests
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml


def auto_mpg(scale_data=True, return_df=False, data_home=None):
    # Note: Does not contain carname.
    data = fetch_openml(data_id=196, parser="auto", data_home=data_home)
    df: pd.DataFrame = (
        data["data"]
        .rename(columns={"model": "year"})
        .astype({"cylinders": int, "year": int})
    )
    df["origin123"] = df["origin"]
    df["origin"] = df["origin"].cat.rename_categories(
        {"1": "USA", "2": "Europe", "3": "Japan"}
    )
    df = pd.get_dummies(df, columns=["origin"])
    not_na = ~df.isna().any(axis=1)
    X = df.values[not_na]
    y = data["target"].values[not_na]
    df = df.dropna()
    df["mpg"] = y

    if scale_data:
        X = scale(X)
        y = scale(y)

    if return_df:
        return X, y, df
    else:
        return X, y


def geckoq(
    return_df=False,
    center_y=True,
    scale_data=True,
    data_home=None,
):
    """The canonical geckoQ loader.

    Assumes data_home contains "Dataframe.csv".
    """
    df_published = pd.read_csv(
        os.path.join(data_home, "Dataframe.csv"),
    )

    # use vitus et al. featureset and y in pascal
    df = df_published.drop(
        columns=[
            "index",
            "InChIKey",
            "ChemPot_kJmol",
            "FreeEnergy_kJmol",
            "HeatOfVap_kJmol",
        ]
    )
    variables = [
        "MW",
        "NumOfAtoms",
        "NumOfC",
        "NumOfO",
        "NumOfN",
        "NumHBondDonors",
        "NumOfConf",
        "NumOfConfUsed",
        "C=C (non-aromatic)",
        "C=C-C=O in non-aromatic ring",
        "hydroxyl (alkyl)",
        "aldehyde",
        "ketone",
        "carboxylic acid",
        "ester",
        "ether (alicyclic)",
        "nitrate",
        "nitro",
        "aromatic hydroxyl",
        "carbonylperoxynitrate",
        "peroxide",
        "hydroperoxide",
        "carbonylperoxyacid",
        "nitroester",
    ]

    X = df[variables].values
    y = np.log10(df["pSat_Pa"].values)

    if scale_data:
        X = scale(X)
    if center_y:
        y = y - y.mean()

    if return_df:
        return X, y, df
    else:
        return X, y


def breast_cancer():
    # data source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    x, y = load_breast_cancer(return_X_y=True)
    # malignant == -1, benign == 1
    y[y == 0] = -1
    x = scale(x)
    return x, y


def diabetes(scale_data=True, return_df=False, data_home=None):
    data = fetch_openml(data_id=37, parser="auto", data_home=data_home)
    df = data["data"]
    target = data["target"]

    x = df.values
    y = np.where(target == "tested_positive", 1, -1)

    if scale_data:
        x = scale(x)

    if return_df:
        return x, y, df

    return x, y


def abalone(scale_data=True, data_home=None):
    # source: https://archive.ics.uci.edu/dataset/1/abalone
    df = fetch_openml(data_id=183, parser="auto", data_home=data_home)

    features: pd.DataFrame = df["data"]
    y = df["target"]

    sex_one_hot = np.eye(2)[(features["Sex"] == "F").values.astype(int)]
    real_vars = features[
        [
            "Length",
            "Diameter",
            "Height",
            "Whole_weight",
            "Shucked_weight",
            "Viscera_weight",
            "Shell_weight",
        ]
    ].values

    x = np.concatenate([sex_one_hot, real_vars], axis=1)

    if scale_data:
        x = scale(x)
        y = scale(y)

    return x, y


def german_credit(data_home=None):
    # source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
    data = fetch_openml(data_id=31, parser="auto", data_home=data_home)

    df = data["data"]
    y = data["target"]

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    preprocessing = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(), categorical_features),
            ("scaler", StandardScaler(), numerical_features),
        ]
    )

    x = preprocessing.fit_transform(df)
    y = np.where(y == "good", 1, -1)
    return x, y


def concrete(scale_data=True, data_home=None):
    # source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
    df = fetch_openml(data_id=4353, parser="auto", data_home=data_home)
    data = df.data
    data = data.dropna()
    x = data[
        [
            "Cement (component 1)(kg in a m^3 mixture)",
            "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
            "Fly Ash (component 3)(kg in a m^3 mixture)",
            "Water  (component 4)(kg in a m^3 mixture)",
            "Superplasticizer (component 5)(kg in a m^3 mixture)",
            "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
            "Fine Aggregate (component 7)(kg in a m^3 mixture)",
            "Age (day)",
        ]
    ].values

    y = data["Concrete compressive strength(MPa. megapascals)"].values

    if scale_data:
        x = scale(x)
        y = scale(y)

    return x, y


def cpu_small(scale_data=True, data_home=None):
    # source https://www.openml.org/search?type=data&status=active&id=227
    x, y = fetch_openml(
        data_id=227, parser="auto", return_X_y=True, data_home=data_home
    )
    x = np.array(x)
    y = np.array(y)

    if scale_data:
        x = scale(x)
        y = scale(y)
    return x, y


def higgs(size=None, scale_data=True, return_df=False, data_home=None):
    # source https://www.openml.org/search?type=data&sort=runs&id=23512&status=active
    data = fetch_openml(data_id=23512, parser="auto", data_home=data_home)
    mask = ~data["data"].isnull().any(axis=1)
    x = data["data"][mask].values
    y = data["target"][mask].values.astype(int)
    y = np.where(y == 1, 1, -1)

    df = data.data.iloc[mask.values, :]

    if scale_data:
        x = scale(x)

    if size is not None:
        x = x[0:size]
        y = y[0:size]
        df = df.iloc[:size, :]

    if return_df:
        return x, y, df
    else:
        return x, y


def airquality(scale_data=True, data_home=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    filename = os.path.join(data_home, "AirQualityUCI.zip")
    csv_name = "AirQualityUCI.csv"

    if not os.path.isfile(filename):
        r = requests.get(url)
        if r.status_code == 200:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, "wb") as f:
                f.write(r.content)

    with ZipFile(filename, "r") as zip:
        with zip.open(csv_name) as f:
            data = pd.read_csv(f, sep=";", decimal=",")

    # Cleaning copied from
    # https://bitbucket.org/edahelsinki/regressionfunction/src/master/python_notebooks/regressionfunction_notebook.ipynb
    data = data.replace(-200, np.nan)
    # impute cases where only 1 hour of data is missing by the mean of its successor and predessor
    for j in range(data.shape[1]):
        for i in range(1, data.shape[0]):
            if (
                (pd.isna(data.iloc[i, j]))
                and not pd.isna(data.iloc[i - 1, j])
                and not pd.isna(data.iloc[i + 1, j])
            ):
                data.iloc[i, j] = (data.iloc[i - 1, j] + data.iloc[i + 1, j]) / 2
    data = data.drop(columns=["NMHC(GT)"])  # Mostly NA.
    data = data.dropna(axis=1, how="all").dropna(axis=0)
    covariates = [
        "PT08.S1(CO)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
    ]
    target = "CO(GT)"

    X = data[covariates].values
    y = data[target].values

    if scale_data:
        X, y = scale(X), scale(y)

    return X, y


def qm9(size=None, scale_data=True, data_home=None):
    """Get the QM9 dataset as used in the "Slisemap application" paper: http://arxiv.org/abs/2310.15610"""
    path = os.path.join(data_home, "slisemap_phys.zip")
    if not os.path.exists(path):
        url = "https://www.edahelsinki.fi/papers/SI_slisemap_phys.zip"
        urlretrieve(url, path)
    df = pd.read_feather(ZipFile(path).open("SI/data/qm9_interpretable.feather"))
    df.drop(columns="index", inplace=True)
    X = df.to_numpy(np.float32)
    y = pd.read_feather(ZipFile(path).open("SI/data/qm9_label.feather"))
    y = y["homo"].to_numpy()

    if size is not None:
        heavies_idx = (
            df.sort_values(by="MW", ascending=False).iloc[:size, :].index.values
        )
        X = X[heavies_idx, :]
        y = y[heavies_idx]
    if scale_data:
        X, y = scale(X), scale(y)
    # else:
    #    max_norm = np.max(np.sqrt(np.sum(X**2, axis=1)))
    #    X = X / max_norm
    #    y = y - y.mean()

    return X, y


def eeg_eye_state(scale_data=True, data_home=None):
    # source https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=1471
    x, y = fetch_openml(
        data_id=1471, parser="auto", return_X_y=True, data_home=data_home
    )
    x = np.array(x)
    y = np.array(y)
    if scale_data:
        x = scale(x)
    y = np.where(y == "1", 1, -1).astype(int)  # Convert 0 to -1
    return x, y


def superconductor(size=None, data_home=None, scale_data=True):
    # https://www.openml.org/search?type=data&status=active&id=43174
    # https://archive.ics.uci.edu/dataset/464/superconductivty+data
    x, y = fetch_openml(
        data_id=43174, parser="auto", return_X_y=True, data_home=data_home
    )
    x = x.values
    y = y.values
    if size is not None:
        np.random.seed(42)
        indices = np.random.choice(x.shape[0], size, replace=False)
        x = x[indices, :]
        y = y[indices]
    if scale_data:
        x = scale(x)
        y = scale(y)
    return x, y


def musk(data_home=None, scale_data=True):
    """
    Predict whether new molecules will be musks or non-musks.
    Found in: https://www.openml.org/d/1116
    See also https://archive.ics.uci.edu/dataset/74/musk+version+1.
    """
    data = fetch_openml(data_id=1116, parser="auto", data_home=data_home)
    df = data["data"]
    y = data["target"].astype(int).values
    y = 2 * y - 1  # -1, +1
    varnames = data["feature_names"][1:]
    X = df.iloc[:, 1:].values
    if scale_data:
        X = scale(X)
    return X, y


def community_crime(data_home=None, scale_data=True):
    """Predict per capita violent crimes.
    https://archive.ics.uci.edu/dataset/183/communities+and+crime"""
    zip_name = Path(data_home) / "communities+and+crime.zip"
    csv_name = "communities.data"
    varnames_name = "communities.names"

    if not zip_name.exists():
        url = "https://archive.ics.uci.edu/static/public/183/communities+and+crime.zip"
        urlretrieve(url, zip_name)

    with ZipFile(zip_name, "r") as zip:
        with zip.open(csv_name) as f:
            df = pd.read_csv(f, sep=",", header=None, na_values="?")
    # Keep columns with less than pct_na percent missing, then filter rows with at least one NA value.
    pct_na = 0.3
    few_na_cols = df.isna().mean(axis=0) < pct_na
    df = df.loc[:, few_na_cols].select_dtypes(exclude=["O"])
    df = df.loc[~df.isna().any(axis=1), :]
    y = df.values[:, -1]
    X = df.values[:, :-1]

    if scale_data:
        X = scale(X)
        y = scale(y)
    return X, y


def california_housing(scale_data=True, return_df=False, data_home=None):
    """
    California housing dataset processed to match sklearn's fetch_california_housing
    Found in: https://www.openml.org/d/44977
    """
    data = fetch_openml(data_id=44977, parser="auto", data_home=data_home)
    df = data["data"]
    y = data["target"].values / 100_000
    df = df.rename(
        columns={
            "longitude": "Longitude",
            "latitude": "Latitude",
            "medianIncome": "MedInc",
            "population": "Population",
            "housingMedianAge": "HouseAge",
        }
    )

    df["AveRooms"] = df["totalRooms"] / df["households"]
    df["AveBedrms"] = df["totalBedrooms"] / df["households"]
    df["AveOccup"] = df["Population"] / df["households"]

    df = df[
        [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
    ]

    X = df.values
    if scale_data:
        X = scale(X)
        y = scale(y)

    if return_df:
        return X, y, df

    return X, y


if __name__ == "__main__":

    DIR_DATA = Path(__file__).parent / "data"
    Path(DIR_DATA).mkdir(exist_ok=True, parents=True)
    auto_mpg(data_home=DIR_DATA)
    geckoq(data_home=DIR_DATA)
    california_housing(data_home=DIR_DATA)
    breast_cancer()
    diabetes(data_home=DIR_DATA)
    abalone(data_home=DIR_DATA)
    german_credit(data_home=DIR_DATA)
    concrete(data_home=DIR_DATA)
    cpu_small(data_home=DIR_DATA)
    higgs(data_home=DIR_DATA)
    airquality(data_home=DIR_DATA)
    qm9(data_home=DIR_DATA)
    eeg_eye_state(data_home=DIR_DATA)
    superconductor(data_home=DIR_DATA)
    musk(data_home=DIR_DATA)
    community_crime(data_home=DIR_DATA)
