from pathlib import Path
import warnings
import glob
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import StrMethodFormatter


def main(csv_file, out_dir, palette=None, interval=None, plot_ceil=None, legend=False):
    """Make figures from csv_file and save them to out_dir"""
    print("Making figures from:", csv_file)

    df = pd.read_csv(csv_file)
    datasets = df["dataset"].unique()
    # replace the model names with a shorter one
    df["eval_model"].replace("KNeighborsRegressor", "KNN", inplace=True)
    df["eval_model"].replace("KNeighborsClassifier", "KNN", inplace=True)
    df["eval_model"].replace("LinearRegression", "LM", inplace=True)
    df["eval_model"].replace("LogisticRegression", "LM", inplace=True)
    df["eval_model"].replace("DecisionTreeRegressor", "DT", inplace=True)
    df["eval_model"].replace("DecisionTreeClassifier", "DT", inplace=True)
    eval_models = df["eval_model"].unique()
    features_methods = ["Original", "XGBOOST", "PCA", "IVIS", "LOL", "GBMAP"]
    dashes = {
        "Original": (1, 1),
        "XGBOOST": (1, 1),
        "GBMAP": "",
        "IVIS": (3, 1),
        "PCA": (5, 5),
        "LOL": (1, 3),
    }

    for dataset in datasets:
        for eval_model in eval_models:
            df_agg = process_results(
                df,
                dataset=dataset,
                eval_model=eval_model,
                features_methods=features_methods,
            )
            axs = plot_feature_construction(
                df_agg,
                palette=palette,
                dashes=dashes,
                interval=interval,
                plot_ceil=plot_ceil,
                legend=legend,
            )

            plt.title(f"{dataset.upper()}-{eval_model}")
            # requires bbox_inches='tight' to correctly show legend in saved fig
            filename = Path(out_dir) / f"features_{eval_model}_{dataset}.pdf"
            plt.savefig(
                filename,
                bbox_inches="tight",
            )
            plt.close()
            print(f"Saved to {filename}.")


def process_results(
    df, dataset, eval_model, features_methods=None, q_low=0.1, q_high=0.9
):
    """
    Process raw results dataframe for plotting

    - select results for a given dataset and model (and feature construction methods)
    - compute medians and quantiles over repeats
    """
    features_methods = (
        df["features_method"].unique() if features_methods is None else features_methods
    )
    df0 = (
        df.query(
            "dataset == @dataset and "
            "eval_model == @eval_model and "
            "features_method in @features_methods"
        )
        .groupby(["dataset", "features_method", "n_features", "eval_model"])
        .agg(
            test_loss_median=pd.NamedAgg(column="test_loss", aggfunc="median"),
            test_loss_q_low=pd.NamedAgg(
                column="test_loss", aggfunc=lambda x: np.quantile(x, q=q_low)
            ),
            test_loss_q_high=pd.NamedAgg(
                column="test_loss", aggfunc=lambda x: np.quantile(x, q=q_high)
            ),
            test_loss_min=pd.NamedAgg(column="test_loss", aggfunc=lambda x: np.min(x)),
        )
        .reset_index()
    )
    # Sort data frame by the given order of feature_methods. Affects legend order.
    df0["features_method"] = pd.Categorical(
        df0["features_method"], categories=features_methods, ordered=True
    )
    df0 = df0.sort_values(by=["features_method"])
    return df0


def plot_feature_construction(
    df, palette=None, dashes=None, ax=None, interval=None, plot_ceil=None, legend=False
):
    """Lineplot of feature construction experiment results

    - x-axis: number of features
    - y-axis: test loss (with error bars)
    - color: feature construction method

    Args:
        df: Data frame with columns = ["n_features", "test_loss_median", "test_loss_q_low", "test_loss_q_high", "features_method"].
        palette: Color palette. Defaults to None.
        dashes: Linestyles. Defaults to None.
        ax: Existing Axes to plot to. Defaults to None.

    Returns:
        Axes
    """
    columns = [
        "features_method",
        "n_features",
        "test_loss_median",
        "test_loss_q_low",
        "test_loss_q_high",
    ]
    if any([col not in df.columns for col in columns]):
        raise ValueError(f"df should have columns {columns}")

    if interval is not None:
        # filter some points
        df = df[(df["n_features"] % interval == 0) | (df["n_features"] < 10)]
    if plot_ceil is not None:
        # cut above m > 100
        df = df[df["n_features"] < plot_ceil]

    xticks = np.sort(df["n_features"].unique())
    df_orig = df[df["features_method"] == "Original"]
    df_bbox = df[df["features_method"] == "XGBOOST"]
    df_others = df[
        (df["features_method"] != "Original") & (df["features_method"] != "XGBOOST")
    ]
    features_methods = df_others["features_method"].unique()

    with warnings.catch_warnings():
        # Ignore pandas FutureWarning in seaborn 0.12.2 ("FutureWarning: use_inf_as_na option is deprecated")
        warnings.simplefilter(action="ignore", category=FutureWarning)
        plt.figure(figsize=(4, 3))
        axs = sns.lineplot(
            df_others,
            x="n_features",
            y="test_loss_median",
            hue="features_method",
            style="features_method",
            hue_order=features_methods,
            style_order=features_methods,
            palette=palette,
            dashes=dashes,
            ax=ax,
        )

    if "test_loss_q_low" in df.columns and "test_loss_q_high" in df.columns:
        # black-box line
        if len(df_bbox) > 0:
            plt.hlines(
                y=df_bbox["test_loss_min"],  # y=df_bbox["test_loss_median"],
                xmin=xticks.min(),
                xmax=xticks.max(),
                linestyle=":",
                colors="black",
                label="XGBOOST",
            )

        # Error bars
        for i, features_method in enumerate(features_methods):
            hspace = 0.01 * i
            df_i = df_others.query("features_method == @features_method")
            plt.errorbar(
                x=df_i["n_features"] + hspace,
                y=df_i["test_loss_median"],
                yerr=[
                    np.abs(df_i["test_loss_q_low"] - df_i["test_loss_median"]),
                    np.abs(df_i["test_loss_q_high"] - df_i["test_loss_median"]),
                ],
                c=palette[features_method],
                fmt=".",
                linewidth=1,
            )
        # Original
        if len(df_orig) > 0:
            qlow = df_orig["test_loss_q_low"].item()
            qhigh = df_orig["test_loss_q_high"].item()
            if np.abs(qhigh - qlow) < 0.001:
                # if the quantiles are too close, the Original bar might not be visible
                print(
                    "Warning: Original quantiles are too close, raising the upper quantile by 0.002"
                )
                qhigh += 0.002
            plt.fill_between(
                x=xticks,
                y1=np.repeat(
                    qlow,
                    len(xticks),
                ),
                y2=np.repeat(
                    qhigh,
                    len(xticks),
                ),
                color=palette["Original"] if palette is not None else "grey",
                alpha=0.3,
                edgecolor=None,
            )
    plt.legend(title="")
    if not legend:
        plt.legend([], [], frameon=False)  # remove legend
    plt.gca().xaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )  # No decimal places
    # plt.xticks(xticks)
    plt.xlabel("Embedding features ($m$)")
    plt.ylabel("Test loss")
    return axs


def make_parser():
    parser = argparse.ArgumentParser(
        description="Plot number of features vs. test error."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the CSV output file from feature_creation.py",
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="features",
        help="Subdirectory name for the figures.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=None,
        help="Plotting interval, e.g., plot every 4:th point.",
    )
    parser.add_argument(
        "-c",
        "--ceil",
        type=int,
        default=None,
        help="Highest m datapoint to plot (ceiling).",
    )
    parser.add_argument(
        "-l",
        "--legend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add legend or not.",
    )
    return parser


if __name__ == "__main__":
    from consts import FIGURE_WRITE_DIR
    from experiment_configs import color_palette

    parser = make_parser()
    args = parser.parse_args()
    DIR_FIGURES = Path(FIGURE_WRITE_DIR)
    figs_path = Path(DIR_FIGURES) / args.dir
    figs_path.mkdir(exist_ok=True, parents=True)

    if ".csv" not in args.file:
        csvs = glob.glob(args.file + "/*.csv")
        for csv_file in csvs:
            main(
                csv_file=csv_file,
                out_dir=figs_path,
                palette=color_palette,
                interval=args.interval,
                plot_ceil=args.ceil,
                legend=args.legend,
            )
    else:
        main(
            csv_file=args.file,
            out_dir=figs_path,
            palette=color_palette,
            interval=args.interval,
            plot_ceil=args.ceil,
            legend=args.legend,
        )
