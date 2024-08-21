from pathlib import Path
import glob

import pandas as pd


def average_results(df):
    metric = "test_loss"

    df = df.copy()
    df["n_features"] = pd.to_numeric(df["n_features"], errors="coerce")

    # Average the results over the cross-validation folds
    averaged_res = []
    for model in df["eval_model"].unique():
        for dataset_name in df["dataset"].unique():
            for embedding_method in df["features_method"].unique():
                for n_features in df["n_features"].unique():
                    mask = (
                        (df["dataset"] == dataset_name)
                        & (df["features_method"] == embedding_method)
                        & (df["n_features"] == n_features)
                        & (df["eval_model"] == model)
                    )

                    # mean = std = np.nan
                    # the algorithm has results on all folds, calculate mean, std
                    median = df[mask][metric].median()
                    std = df[mask][metric].std()

                    averaged_res.append(
                        {
                            "eval_model": model,
                            "dataset": dataset_name,
                            "features_method": embedding_method,
                            "n_features": n_features,
                            "median": median,
                            "std": std,
                            "n_folds": len(df[mask][metric]),
                        }
                    )
    df = pd.DataFrame(averaged_res)
    df["n_features"] = df["n_features"].astype(int)
    df = df.rename(columns={"n_features": "m"})
    return df


WRITE_DIR = "latex_tables"
results_dir = "results/features_ex"
methods = ["GBMAP", "CCA", "PLS", "LDA"]

OUT_DIR = Path(WRITE_DIR)
OUT_DIR.mkdir(exist_ok=True, parents=True)
csvs = glob.glob(results_dir + "/*.csv")

dfs = [pd.read_csv(csv_file) for csv_file in csvs]
df = pd.concat(dfs)

df = df[df["features_method"].isin(methods)]
df = df[df["n_features"] == 1]
# replace the model names with a shorter one
# This will make sure we have just 3 tables with both regression and classification datasets
df["eval_model"].replace("KNeighborsRegressor", "KNN", inplace=True)
df["eval_model"].replace("KNeighborsClassifier", "KNN", inplace=True)
df["eval_model"].replace("LinearRegression", "LM", inplace=True)
df["eval_model"].replace("LogisticRegression", "LM", inplace=True)
df["eval_model"].replace("DecisionTreeRegressor", "DT", inplace=True)
df["eval_model"].replace("DecisionTreeClassifier", "DT", inplace=True)

avg_df = average_results(df)

frames = []
for model in avg_df["eval_model"].unique():
    subset = avg_df[avg_df["eval_model"] == model].copy()

    subset["features_method"] = subset["features_method"].apply(
        lambda x: "{}-{}".format(x, model)
    )
    frames.append(subset)

combined_df = pd.concat(frames)
piv = combined_df.pivot(
    index=["dataset", "m"], columns="features_method", values="median"
)
cols_reg = [
    "GBMAP-LM",
    "CCA-LM",
    "PLS-LM",
    # "GBMAP-DT",
    "CCA-DT",
    "PLS-DT",
    # "GBMAP-KNN",
    "CCA-KNN",
    "PLS-KNN",
]
piv[cols_reg]
piv[cols_reg].to_latex(
    Path(OUT_DIR / "feature_reg_tab.tex"),
    na_rep="--",
    float_format="$%.2f$",
    escape=False,
)

cols_cls = [
    "GBMAP-LM",
    "LDA-LM",
    # "GBMAP-DT",
    "LDA-DT",
    # "GBMAP-KNN",
    "LDA-KNN",
]
piv[cols_cls]
piv[cols_cls].to_latex(
    Path(OUT_DIR / "feature_cls_tab.tex"),
    na_rep="--",
    float_format="$%.2f$",
    escape=False,
)
