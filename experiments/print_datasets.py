from consts import FIGURE_WRITE_DIR
from experiment_configs import features_reg_datasets, features_cls_datasets

from pathlib import Path

import pandas as pd


dir_tables = Path(FIGURE_WRITE_DIR)
dir_tables.mkdir(parents=True, exist_ok=True)

datasets = []
for dataset, dataset_dict in features_reg_datasets().items():
    X, y = dataset_dict["loader"](**dataset_dict["params"])
    datasets.append(
        {"dataset": dataset, "N": X.shape[0], "p": X.shape[1], "target": "regression"}
    )

for dataset, dataset_dict in features_cls_datasets().items():
    X, y = dataset_dict["loader"](**dataset_dict["params"])
    datasets.append(
        {
            "dataset": dataset,
            "N": X.shape[0],
            "p": X.shape[1],
            "target": "classification",
        }
    )

df = pd.DataFrame(datasets)
print(df)


def column_formatter(x):
    d = {
        "dataset": "Dataset",
        "N": "$n$",
        "p": "$p$",
        "target": "Target",
    }
    try:
        return d[x]
    except:
        return x


path_to_tex = dir_tables / "datasets.tex"
df.replace(
    {"target": {"regression": r"$\mathbb{R}$", "classification": r"$\pm 1$"}}
).style.hide(axis=0).format_index(
    axis=1,
    formatter=column_formatter,
    escape="latex",
).format(
    escape="latex"
).to_latex(
    path_to_tex, hrules=True
)
print(f"Saved table to {path_to_tex}.")
