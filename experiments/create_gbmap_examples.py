from consts import FIGURE_WRITE_DIR, DIR_DATA

from gbmap.common import mse, loss_logistic
from gbmap.gbmap import GBMAP

from data import california_housing
from data import diabetes

from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# This script creates the examples diabetes and california housing


def plot_vars(X, i1, i2=None, names=None, outliers=False, whis=(1, 99)):
    """Paired boxplot for variables in array X comparing two subsets i1, i2.
    - boxes: [q25, q75],
    - whiskers:
        - if whis is float, then [q25 - whis * (q75-q25), q75 + whis * (q75-q25)]
        - if whis = (a, b), then [q_a, q_b]
    """
    positions = np.arange(X.shape[1])
    if names is None:
        names = positions
    width = 0.15
    width_box = 0.2
    positions1, positions2 = positions - width, positions + width
    bx1 = plt.boxplot(
        X[i1],
        positions=positions1,
        widths=width_box,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", linewidth=0),
        whiskerprops=dict(linewidth=0),
        flierprops=dict(markersize=1),
        vert=False,
        sym="o" if outliers else "",
        capwidths=0,
        whis=whis,
        showcaps=False,
    )
    for median in bx1["medians"]:
        median.set_color("black")
    bx2 = None
    if i2 is not None:
        bx2 = plt.boxplot(
            X[i2],
            positions=positions2,
            widths=width_box,
            patch_artist=True,
            boxprops=dict(facecolor="gray", linewidth=0),
            whiskerprops=dict(linewidth=0),
            flierprops=dict(markersize=1),
            vert=False,
            sym="o" if outliers else "",
            capwidths=0,
            whis=whis,
            showcaps=False,
        )
    for median in bx2["medians"]:
        median.set_color("black")
    plt.yticks(positions, names)
    plt.xlabel("Feature value")
    return bx1, bx2


DIR_FIGURES = Path(FIGURE_WRITE_DIR)
DIR_FIGURES.mkdir(exist_ok=True, parents=True)

pal = sns.color_palette(palette="Greys")


#### California ####

X, y, df = california_housing(data_home=DIR_DATA, return_df=True, scale_data=False)
feature_names = list(df.columns)
df["med.value"] = y


scaler = StandardScaler()
X = scaler.fit_transform(X)

lr = LinearRegression()
lr.fit(X, y)
y0 = lr.predict(X)
print("linear reg mse", mse(y, y0))


gb = GBMAP(n_boosts=1, softplus_scale=1000, n_fit=10, random_state=0)
gb.fit(x=X, y=y, y0=y0)
print("mse after gbmap correction", mse(y, gb.predict(X, y0=y0)))
print(Counter(gb.get_activated_hyperplane(X)))
gb.print_params()

# Plot LM/GBMAP coefs
gb_coefs = np.array(gb.params[0][1] * gb.params[0][-1])[:-1]
lm_sort_idx = np.flip(np.argsort(lr.coef_))
plot_df = pd.DataFrame(
    [lr.coef_[lm_sort_idx], gb_coefs[lm_sort_idx]],
    columns=np.array(feature_names)[lm_sort_idx],
)
plot_df["coefs"] = ["LM", "GBMAP correction"]

color_palette = {"LM": pal[2], "GBMAP correction": pal[4]}

plt.figure(figsize=(4, 3))
long_df = plot_df.melt(id_vars="coefs").rename(columns=str.title)
sns.barplot(x="Value", y="Variable", hue="Coefs", data=long_df, palette=color_palette)
plt.xlabel("Coefficient")
plt.ylabel("")
plt.legend(title="")
filename = Path(DIR_FIGURES) / "cal_coefs.pdf"
plt.savefig(filename, bbox_inches="tight")
plt.close()
print(f"Saved to {filename}.")


# plot regions box plot
plt.figure(figsize=(4, 3))
bplot_idx = np.flip(lm_sort_idx)
plot_vars(
    X[:, bplot_idx],
    [r == () for r in gb.get_activated_hyperplane(X)],
    [r == (0,) for r in gb.get_activated_hyperplane(X)],
    names=np.array(feature_names)[bplot_idx],
    whis=(25, 75),
)
filename = Path(DIR_FIGURES) / "cal_bplot.pdf"
plt.savefig(filename, bbox_inches="tight")
plt.close()
print(f"Saved to {filename}.")


#### diabetes ####
X, y, df = diabetes(scale_data=False, return_df=True)
df = df.rename(columns={"mass": "BMI"})
df = df.drop(columns=["pedi"])
feature_names = list(df.columns)

X = X[:, [0, 1, 2, 3, 4, 5, 7]]
scaler = StandardScaler()
X = scaler.fit_transform(X)

gb = GBMAP(
    n_boosts=1, softplus_scale=1000, n_fit=10, is_classifier=True, random_state=0
)
gb.fit(x=X, y=y)
print("gbmap logistic loss", loss_logistic(y, gb.predict(X)))
print(Counter(gb.get_activated_hyperplane(X)))
gb.print_params()

features = np.array(feature_names)
coef = (gb.params[0][1] * gb.params[0][-1])[:-1]
sort_idx = np.flip(np.argsort(coef))
plot_df = pd.DataFrame({"feature": features[sort_idx], "weight": coef[sort_idx]})

color_palette = {varname: pal[4] for varname in feature_names}
plt.figure(figsize=(4, 3))
sns.barplot(data=plot_df, x="weight", y="feature", palette=color_palette, width=0.5)
plt.xlabel("Coefficient")
plt.ylabel("")
# ax1.set_xlabel("Coefficient")
# ax1.set_ylabel("")
filename = Path(DIR_FIGURES) / "diabetes_coefs.pdf"
plt.savefig(filename, bbox_inches="tight")
plt.close()
print(f"Saved to {filename}.")

# plot regions box plot
plt.figure(figsize=(4, 3))
bplot_idx = np.flip(sort_idx)
plot_vars(
    X[:, bplot_idx],
    [r == () for r in gb.get_activated_hyperplane(X)],
    [r == (0,) for r in gb.get_activated_hyperplane(X)],
    names=features[bplot_idx],
    whis=(25, 75),
)
filename = Path(DIR_FIGURES) / "diabetes_bplot.pdf"
plt.savefig(filename, bbox_inches="tight")
plt.close()
print(f"Saved to {filename}.")
