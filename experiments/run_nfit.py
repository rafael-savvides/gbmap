from gbmap.gbmap import GBMAP

import argparse
from datetime import datetime
from functools import partial
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from data import (
    abalone,
    auto_mpg,
    concrete,
    geckoq,
    airquality,
    cpu_small,
    qm9,
    california_housing,
    superconductor,
)

# Force flushing when running on the cluster (large output buffer).
print = partial(print, flush=True)


def main(
    dir_results,
    dir_data,
    dir_figures,
    datasets,
    summarize,
    nfit_max,
):
    fname_prefix = "nfits"
    t_start = datetime.now()

    if not summarize:
        print(
            f"Running nfit experiment. {t_start.strftime('%Y-%m-%d %T')}\n"
            f"nfit_max={nfit_max}, datasets=[{','.join(datasets)}]"
        )
        for i, dataset in enumerate(datasets, start=1):
            print(f"{i}/{len(datasets)} {dataset}")
            X, y = get_dataset(dataset, data_home=dir_data)
            rng = np.random.default_rng(seed=2024 + sum(map(ord, dataset)))
            random_states = rng.integers(0, 2**32, size=nfit_max)
            res = run_nfit(
                X,
                y,
                dataset,
                input_params={"n_boosts": 10, "softplus_scale": 1},
                random_states=random_states,
                progress=True,
            )
            read_write_result(Path(dir_results) / f"{fname_prefix}-{dataset}.pkl", res)
    else:
        nfits = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]
        loss_dict = dict()
        for file in Path(dir_results).glob(f"{fname_prefix}*.pkl"):
            res = read_write_result(file)
            loss_dict[res["dataset"]] = process_nfit(
                losses=res["losses_tr"], nfits=nfits, seed=0, n_rep=100
            )

        with plt.rc_context({"font.size": 18}):
            nfits_ticks = [1] + list(range(10, max(nfits), 10)) + [max(nfits)]
            plt.figure(figsize=(5, 3))
            for dataset, loss_min in loss_dict.items():
                plt.plot(nfits, loss_min, label=dataset)
            # plt.legend()
            plt.xlabel("$n_{fit}$")
            plt.ylabel("$\\frac{L - L_{min}}{L_{min}}$")

            plt.xticks(
                ticks=nfits_ticks,
                labels=[nfit if nfit in [1, 50, 100] else "" for nfit in nfits_ticks],
            )
            plt.tight_layout()
            plt.savefig(Path(dir_figures) / "nfits.png")
    t_end = datetime.now()
    print(
        f"Done. ({t_end.strftime('%Y-%m-%d %T')}, {(t_end - t_start).total_seconds(): .0f} sec)"
    )


def run_nfit(X, y, dataset, random_states, input_params, progress=False):
    gbmaps = [
        GBMAP(**input_params, random_state=random_state).fit(X, y)
        for random_state in tqdm(random_states, disable=not progress)
    ]
    losses_tr = [np.mean((gb.predict(X) - y) ** 2) for gb in gbmaps]
    return {"dataset": dataset, "losses_tr": losses_tr, "random_states": random_states}


def process_nfit(losses, nfits=(1, 5, 10, 20, 50, 75, 100), seed=None, n_rep=100):
    rng = np.random.default_rng(seed)
    # l is the average minimum loss with nfit (min over nfit losses, avg over n_rep random sets of nfit losses).
    l = np.array(
        [
            np.mean(
                [
                    np.min(rng.choice(losses, size=nfit, replace=False))
                    for _ in range(n_rep)
                ]
            )
            for nfit in nfits
        ]
    )
    return (l - l.min()) / l.min()


def process_nfit_q(
    losses,
    nfits=(1, 5, 10, 20, 50, 75, 100),
    seed=None,
    n_rep=100,
    q_low=0.25,
    q_high=0.75,
):
    rng = np.random.default_rng(seed)
    # l is the median minimum loss with nfit (min over nfit losses, median over n_rep random sets of nfit losses).
    l, l_low, l_high = [], [], []
    for nfit in nfits:
        l_nfit = [
            np.min(rng.choice(losses, size=nfit, replace=False)) for _ in range(n_rep)
        ]
        l.append(np.median(l_nfit))
        l_low.append(np.quantile(l_nfit, q_low))
        l_high.append(np.quantile(l_nfit, q_high))
    l, l_low, l_high = np.array(l), np.array(l_low), np.array(l_high)
    s = lambda x, a=l: (x - np.min(a)) / np.min(a)
    return s(l), s(l_low), s(l_high)


def read_write_result(file: Path, obj=None):
    """Read or write a result to a file

    Args:
        file: File (inc. extension) to save to or load from, such as "result.pkl".
        obj: A python object to save to file. If None, file is read.
    """
    file = Path(file)
    suffix = "".join(file.suffixes)
    mode = "read" if obj is None else "write"
    if suffix == ".pkl":
        if mode == "read":
            with open(file, mode="rb") as f:
                obj = pickle.load(f)
            return obj
        elif mode == "write":
            with open(file, mode="wb") as f:
                pickle.dump(obj, f)
            return None
    else:
        raise NotImplementedError("file should end in .pkl")


def get_dataset(dataset, data_home=None):
    match dataset:
        case "auto_mpg":
            X, y = auto_mpg(data_home=data_home)
        case "california_housing":
            X, y = california_housing(data_home=data_home)
        case "abalone":
            X, y = abalone(data_home=data_home)
        case "concrete":
            X, y = concrete(data_home=data_home)
        case "cpu_small":
            X, y = cpu_small(data_home=data_home)
        case "airquality":
            X, y = airquality(data_home=data_home)
        case "superconductor":
            X, y = superconductor(data_home=data_home)
        case "qm9":
            X, y = qm9(data_home=data_home)
        case "geckoq":
            X, y, y0 = geckoq(data_home=data_home)
    return X, y


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--summarize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Summarize results.",
    )
    parser.add_argument(
        "-p",
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print progress bar.",
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=2,
        help="Number of repetitions.",
    )
    parser.add_argument(
        "-j",
        "--job-index",
        type=int,
        default=0,
        help="Job index. Used for naming files and setting seed when parallelizing.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="",
        help="Dataset. If empty, run all.",
    )
    return parser


if __name__ == "__main__":
    from consts import WRITE_DIR, FIGURE_WRITE_DIR

    datasets = [
        "california_housing",
        "concrete",
        "cpu_small",
        "geckoq",
        "qm9",
        "superconductor",
    ]

    parser = make_argparser()
    args = parser.parse_args()

    DIR_DATA = Path(__file__).parent / "data"
    DIR_RESULTS = Path(WRITE_DIR) / "nfits"
    DIR_FIGURES = Path(FIGURE_WRITE_DIR)
    DIR_RESULTS.mkdir(exist_ok=True, parents=True)
    DIR_FIGURES.mkdir(exist_ok=True, parents=True)
    main(
        dir_results=DIR_RESULTS,
        dir_data=DIR_DATA,
        dir_figures=DIR_FIGURES,
        nfit_max=100,
        datasets=datasets if args.dataset == "" else [args.dataset],
        summarize=args.summarize,
    )
