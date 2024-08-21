from consts import FIGURE_WRITE_DIR
from experiment_configs import color_palette

from pathlib import Path
import glob
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

methods = ["GBMAP", "PCA", "LOL", "IVIS"]


def process_results(df, q_low=0.1, q_high=0.9):
    """
    Process raw results dataframe for plotting

    - select results for a given dataset and model (and feature construction methods)
    - compute medians and quantiles over repeats
    """
    method = df["method"].unique()
    df0 = (
        df.query("method in @method")
        .groupby(["method", "n", "p"])
        .agg(
            runtime_median=pd.NamedAgg(column="runtime", aggfunc="median"),
            runtime_q_low=pd.NamedAgg(
                column="runtime", aggfunc=lambda x: np.quantile(x, q=q_low)
            ),
            runtime_q_high=pd.NamedAgg(
                column="runtime", aggfunc=lambda x: np.quantile(x, q=q_high)
            ),
        )
        .reset_index()
    )
    # Sort data frame by the given order of feature_methods. Affects legend order.
    df0["method"] = pd.Categorical(df0["method"], categories=method, ordered=True)
    df0 = df0.sort_values(by=["method"])
    return df0


def scaling_plot(df, xvar, palette=None, dashes=None, ax=None):
    """Creates scaling plot"""

    method = df["method"].unique()
    xticks = np.sort(df[xvar].unique())
    plt.figure(figsize=(4, 3))
    with warnings.catch_warnings():
        # Ignore pandas FutureWarning in seaborn 0.12.2 ("FutureWarning: use_inf_as_na option is deprecated")
        warnings.simplefilter(action="ignore", category=FutureWarning)
        plt.figure(figsize=(4, 3))
        axs = sns.lineplot(
            df,
            x=xvar,
            y="runtime_median",
            hue="method",
            style="method",
            hue_order=method,
            style_order=method,
            palette=palette,
            dashes=dashes,
            ax=ax,
        )

        # Error bars
        for i, method in enumerate(method):
            hspace = 0.01 * i
            df_i = df.query("method == @method")
            plt.errorbar(
                x=df_i[xvar] + hspace,
                y=df_i["runtime_median"],
                yerr=[
                    np.abs(df_i["runtime_q_low"] - df_i["runtime_median"]),
                    np.abs(df_i["runtime_q_high"] - df_i["runtime_median"]),
                ],
                c=palette[method],
                fmt=".",
                linewidth=1,
            )

    plt.legend(title="", loc="lower right")
    plt.xticks(xticks)
    plt.xlabel("${}$".format(xvar))
    plt.ylabel("Runtime (s)")
    return axs


DIR_FIGURES = Path(FIGURE_WRITE_DIR)
DIR_FIGURES.mkdir(exist_ok=True, parents=True)

dashes = {
    "GBMAP": "",
    "IVIS": (3, 1),
    "PCA": (5, 5),
    "LOL": (1, 3),
}

# datapoints scaling
res_dfs = []
for method_name in methods:
    result_paths = glob.glob("results/scaling/scaling_{}-n*.csv".format(method_name))
    dfs = [pd.read_csv(path) for path in result_paths]
    df = pd.concat(dfs)
    # res_dfs.append(df)
    res_dfs.append(df[df["n"] != 50_000_000])

df = pd.concat(res_dfs)
df_agg = process_results(df)
axs = scaling_plot(df_agg, xvar="n", dashes=dashes, palette=color_palette)
axs.set(xscale="log", yscale="log")
filename = Path(DIR_FIGURES) / "scaling_n.pdf"
plt.savefig(
    filename,
    bbox_inches="tight",
)
plt.close()
print(f"Saved to {filename}.")

# features scaling
res_dfs = []
for method_name in methods:
    result_paths = glob.glob("results/scaling/scaling_{}-p*.csv".format(method_name))
    dfs = [pd.read_csv(path) for path in result_paths]
    df = pd.concat(dfs)
    res_dfs.append(df)

df = pd.concat(res_dfs)
df_agg = process_results(df)
axs = scaling_plot(df_agg, xvar="p", dashes=dashes, palette=color_palette)
axs.set(xscale="log", yscale="log")
filename = Path(DIR_FIGURES) / "scaling_p.pdf"
plt.savefig(
    filename,
    bbox_inches="tight",
)
plt.close()
print(f"Saved to {filename}.")
