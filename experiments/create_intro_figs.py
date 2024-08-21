from gbmap.gbmap import GBMAP

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_relu_softplus(file=None, width=2, height=2):
    xgrid = np.linspace(-1, 1, num=100)
    fig, axs = plt.subplots(figsize=(width, height))
    plt.plot(xgrid, relu(xgrid), c="black", label="ReLU")
    plt.plot(xgrid, softplus(xgrid, beta=8), c="grey", ls="dashed", label="Softplus")
    plt.gca().axis("off")
    plt.xticks([0], labels=[0])
    plt.yticks([])
    plt.annotate("Active", (0.1, -0.1), annotation_clip=False)
    plt.annotate(
        "Inactive",
        (-0.1, -0.1),
        annotation_clip=False,
        horizontalalignment="right",
    )
    plt.vlines(0, ymin=-0.1, ymax=0, ls="dashed", color="grey", lw=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    if file is not None:
        plt.savefig(file)
        print(f"Saved to {file}.")
    return fig, axs


def plot_gbmap_iterations(file=None, width=5, height=2, seed=2024):
    def get_intercept(w):
        return -w[1] / w[0]

    rng = np.random.default_rng(seed=seed)
    X = rng.uniform(-1, 1, (100, 1))
    y = 1 - X[:, 0] ** 2 + 0.01 * rng.standard_normal(len(X))
    gb = GBMAP(n_boosts=3, softplus_scale=30, n_fit=10, random_state=123).fit(X, y)
    xgrid = np.linspace(X.min(), X.max(), num=100).reshape(-1, 1)
    yhat1 = gb.predict(xgrid, n_boosts=1)
    yhat12 = gb.predict(xgrid, n_boosts=2)
    yhat2 = yhat12 - yhat1

    fig, axs = plt.subplots(ncols=2, figsize=(width, height))

    ax = axs[0]
    ax.scatter(X, y, s=1, c="black")
    ax.plot(xgrid, yhat1, c="C0", lw=3, alpha=0.7)
    ax.set_title("$m=1$")
    ax.set_axis_off()
    intercept = get_intercept(gb.get_params(0)["w"])
    ax.axvline(intercept, ls="dashed", c="grey", lw=0.5)
    ax.annotate(
        "1 active",
        (intercept, -0.1),
        horizontalalignment="right",
        annotation_clip=False,
        bbox={"fc": "white", "ec": "white"},
    )

    ax = axs[1]
    ax.scatter(X, y, s=1, c="black")
    # ax.plot(xgrid, yhat1, c="blue", lw=1, alpha=0.7)
    ax.plot(xgrid, yhat12, c="C0", lw=3, alpha=0.7)
    ax.set_title("$m=2$")
    ax.set_axis_off()
    intercept = get_intercept(gb.get_params(0)["w"])
    ax.axvline(intercept, ls="dashed", c="grey", lw=0.5)
    ax.annotate(
        f"{0+1} active",
        (intercept, -0.1),
        horizontalalignment="right",
        annotation_clip=False,
        bbox={"fc": "white", "ec": "white"},
    )
    intercept = get_intercept(gb.get_params(1)["w"])
    ax.axvline(intercept, ls="dashed", c="grey", lw=0.5)
    ax.annotate(
        f"{1+1} active",
        (intercept, -0.1),
        horizontalalignment="left",
        annotation_clip=False,
        bbox={"fc": "white", "ec": "white"},
    )
    plt.tight_layout()
    if file is not None:
        plt.savefig(file)
        print(f"Saved to {file}.")
    return fig, axs


def plot_gbmap_regions_2d(file=None, width=3, height=3, seed=2024):
    clrs = {(): "grey", (0,): "blue", (1,): "darkred", (0, 1): "purple"}
    markers = {(): ".", (0,): "o", (1,): "x", (0, 1): "*"}
    bbox = dict(boxstyle="round,pad=0.1", fc="white", ec="white", lw=1)

    rng = np.random.default_rng(seed=seed)
    X2 = rng.standard_normal((200, 2))
    y2 = (
        1
        + 0.5 * X2[:, 0] ** 2
        - 0.5 * X2[:, 1] ** 2
        + 0.01 * rng.standard_normal(len(X2))
    )
    gb2 = GBMAP(n_boosts=2, softplus_scale=10, n_fit=10, random_state=seed).fit(X2, y2)
    regions = gb2.get_activated_hyperplane(X2)

    fig, axs = plt.subplots(figsize=(width, height))
    for region in set(regions):
        idx = [r == region for r in regions]
        label = (
            f"{','.join([f'{r + 1}' for r in region])} active"
            if region != ()
            else "Inactive"
        )
        plt.scatter(
            X2[idx, 0],
            X2[idx, 1],
            c=clrs[region],
            s=15,
            marker=markers[region],
        )
        plt.annotate(
            label,
            xy=np.mean(np.quantile(X2[idx, :], q=[0, 1], axis=0), axis=0),
            horizontalalignment="center",
            c=clrs[region],
            bbox=bbox,
        )
    for i in range(2):
        w1, w2, w0 = gb2.get_params(i)["w"]
        intercept, slope = -w0 / w2, -w1 / w2
        plt.axline((0, intercept), slope=slope, c=clrs[(i,)])
    plt.xlim((X2[:, 0].min(), X2[:, 0].max()))
    plt.ylim((X2[:, 1].min(), X2[:, 1].max()))
    plt.gca().axis("off")
    plt.tight_layout()
    if file is not None:
        plt.savefig(file)
        print(f"Saved to {file}.")
    return fig, axs


def relu(x):
    return np.maximum(x, 0)


def softplus(x, beta=1):
    return (1 / beta) * np.logaddexp(beta * x, 0)


if __name__ == "__main__":
    from consts import FIGURE_WRITE_DIR

    dir_figures = Path(FIGURE_WRITE_DIR)
    dir_figures.mkdir(parents=True, exist_ok=True)

    plot_relu_softplus(dir_figures / "relu_softplus.pdf", width=2, height=2)
    plot_gbmap_iterations(
        dir_figures / "gbmap_iterations.pdf", width=5, height=2, seed=2024
    )
    plot_gbmap_regions_2d(
        dir_figures / "gbmap_regions.pdf", width=3, height=3, seed=2024
    )
