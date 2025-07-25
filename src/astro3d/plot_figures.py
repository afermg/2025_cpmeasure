#!/usr/bin/env python
from itertools import chain
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
import shap
import xgboost
from cp_measure.bulk import get_core_measurements
from joblib import Parallel, delayed
from pooch import Untar, retrieve
from skimage.io import imread
from sklearn.model_selection import train_test_split
from util_plot import generate_label
from xgboost import XGBClassifier

MEASUREMENTS = get_core_measurements()

overwrite = False
figs_path = Path(".") / ".." / ".." / "figs"
data_path = Path("/") / "datastore" / "alan" / "cp_measure"
figs_path.parent.mkdir(parents=True, exist_ok=True)
data_path.parent.mkdir(parents=True, exist_ok=True)
profiles_fpath = data_path / Path("astro3d_profiles.parquet")

target_feature = "day"

cell_count_col = "Nuclei Count"
seed = 42


def fix_non_continuous_labels(labels_redz: np.ndarray) -> np.ndarray:
    """
    Enforce continuous labels from 1 to `n_unique_labels`.

    Some measurements assume that labels continuous in 1->n.
    This function removes any gaps (e.g., [0,1,3,4] -> [0,1,2,3]).
    """
    max_val = labels_redz.max()
    uniq = np.unique(labels_redz)
    missing_ix = np.setdiff1d(
        np.arange(max_val),
        uniq,
        assume_unique=True,
    )
    if len(missing_ix):
        new_ranges = np.arange(1, len(uniq))
        for i, ix in enumerate(uniq[1:]):
            labels_redz[labels_redz == ix] = new_ranges[i]

    return labels_redz


def apply_measurements(mask_path: str, img_path: str) -> pl.DataFrame:
    meta = Path(img_path).parent.name.split("_")[:2]
    meta = (*meta, Path(img_path).stem)

    labels = imread(mask_path)
    labels_redz = labels.max(axis=0)
    # Cover case where the reduction on z removes an entire item
    # Unlike in other cases, here we assume that the input mask can have gaps
    labels_redz = fix_non_continuous_labels(labels_redz)

    img = imread(img_path)
    flat_img = img.max(axis=0)
    flat_img = flat_img / flat_img.max()
    d = {}

    for meas_name, meas_f in MEASUREMENTS.items():
        measurements = meas_f(labels_redz, flat_img)
        # Unpack output dictionaries
        for k, v in measurements.items():
            d[k] = v
        for k, v in zip(("pert", "day", "stem"), meta):
            d[k] = v

    df = pl.from_dict(d)

    return df


def map_mask_to_image(fpath: Path) -> tuple[str, str] or None:
    if fpath.endswith("mask.tif"):
        img_path = (
            str(fpath).replace("_masks", "_deconv").removesuffix("_mask.tif") + ".tif"
        )
        return (fpath, str(img_path))


url_hash = {
    "https://zenodo.org/api/records/15594999/files/nha_day3_deconv.tar.gz/content": "6d1fd24d7f9381e1aa4f95dbc9c2c351c7633a3a4500d95b4a6736d40b4100af",
    "https://zenodo.org/api/records/15594999/files/nha_day3_masks.tar.gz/content": "d428491b3bc8207bedf77f33afdde5e6542ea6c4f9f63e0cf920f46b6f68f43b",
    "https://zenodo.org/api/records/15594999/files/nha_day5_deconv.tar.gz/content": "f7c7f93d1a173441f8b116ca8411df524678d926ed30d490d580d01edd782503",
    "https://zenodo.org/api/records/15594999/files/nha_day5_masks.tar.gz/content": "ddbf48a883f03ab89c3adf11cbf4b1e07905f8deb734502f38b684bb009e900b",
    "https://zenodo.org/api/records/15594999/files/nha_day7_deconv.tar.gz/content": "9dd69934f306cab31e94df427ee7618d21c32038a62a14ccab15b7d3517eadf7",
    "https://zenodo.org/api/records/15594999/files/nha_day7_masks.tar.gz/content": "55aafc7b49f904f711187a85ef0bf5e410d3e6aef4b0b7bf69ef17fce51ee8b6",
    "https://zenodo.org/api/records/15594999/files/vpa_day3_deconv.tar.gz/content": "2e1637fd0f240fda6b19ea7d57668ce6b82ae35f697fa2af9b30fc6915d609b5",
    "https://zenodo.org/api/records/15594999/files/vpa_day3_masks.tar.gz/content": "e47c7b476f1328f79f4d6825c74706fc7e5504ca3eed7d41caa7199b3d001690",
    "https://zenodo.org/api/records/15594999/files/vpa_day5_deconv.tar.gz/content": "08c7278e4c1a61fa77b8c196e79bcd5e222241ff05b492d13a18a9f976c81cf3",
    "https://zenodo.org/api/records/15594999/files/vpa_day5_masks.tar.gz/content": "366293f5e68247e927628b6b2e8a8c33338de9a56db9f269e7fb6a0500dbb55b",
    "https://zenodo.org/api/records/15594999/files/vpa_day7_deconv.tar.gz/content": "022e8d8d086e65551d1fabf8048bbf0fe0a6fbe9015bdc6a6191227730ad8408",
    "https://zenodo.org/api/records/15594999/files/vpa_day7_masks.tar.gz/content": "63bf6d123ee9505a51fb0ef099ba77223a800babab51756d577cd7cfd8dd182d",
}

# Original locations:
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day3_deconv.tar.gz": "6d1fd24d7f9381e1aa4f95dbc9c2c351c7633a3a4500d95b4a6736d40b4100af",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day3_masks.tar.gz": "d428491b3bc8207bedf77f33afdde5e6542ea6c4f9f63e0cf920f46b6f68f43b",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day5_deconv.tar.gz": "f7c7f93d1a173441f8b116ca8411df524678d926ed30d490d580d01edd782503",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day5_masks.tar.gz": "ddbf48a883f03ab89c3adf11cbf4b1e07905f8deb734502f38b684bb009e900b",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day7_deconv.tar.gz": "9dd69934f306cab31e94df427ee7618d21c32038a62a14ccab15b7d3517eadf7",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day7_masks.tar.gz": "55aafc7b49f904f711187a85ef0bf5e410d3e6aef4b0b7bf69ef17fce51ee8b6",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day3_deconv.tar.gz": "2e1637fd0f240fda6b19ea7d57668ce6b82ae35f697fa2af9b30fc6915d609b5",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day3_masks.tar.gz": "e47c7b476f1328f79f4d6825c74706fc7e5504ca3eed7d41caa7199b3d001690",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day5_deconv.tar.gz": "08c7278e4c1a61fa77b8c196e79bcd5e222241ff05b492d13a18a9f976c81cf3",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day5_masks.tar.gz": "366293f5e68247e927628b6b2e8a8c33338de9a56db9f269e7fb6a0500dbb55b",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day7_deconv.tar.gz": "022e8d8d086e65551d1fabf8048bbf0fe0a6fbe9015bdc6a6191227730ad8408",
# "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day7_masks.tar.gz": "63bf6d123ee9505a51fb0ef099ba77223a800babab51756d577cd7cfd8dd182d",

retrieved = list(
    Parallel(n_jobs=-1)(
        delayed(retrieve)(
            url,
            processor=Untar(
                extract_dir=Path(url.removesuffix("/content")).name.split(".")[0]
            ),
            known_hash=h,
            progressbar=True,
        )
        for url, h in url_hash.items()
    )
)

dir_files = {
    x: y
    for x, y in zip(
        (Path(url.removesuffix("/content")).name.split(".")[0] for url in url_hash),
        retrieved,
    )
    if x.endswith("_masks")  # Optional but makes the next step faster
}

pairs = [
    x
    for x in Parallel(n_jobs=-1)(
        delayed(map_mask_to_image)(x) for x in chain(*dir_files.values())
    )
    if x is not None  # Ignore pngs
]

# %%
if not profiles_fpath.exists() or overwrite:
    dfs = list(
        Parallel(n_jobs=-1)(
            delayed(apply_measurements)(m_path, i_path) for m_path, i_path in pairs
        )
    )
    profiles = pl.concat(dfs)
    profiles.write_parquet(profiles_fpath)
else:
    profiles = pl.read_parquet(profiles_fpath)

# %%
filtered = profiles.filter(pl.col("day") == "day7")
core_df = profiles
meta = core_df.select(cs.by_dtype(pl.String))
cc = meta.group_by("stem").agg(
    pl.len().alias(cell_count_col), pl.first("pert"), pl.first("day")
)


with_cc = (
    core_df.select(~cs.by_dtype(pl.String), "stem").join(cc, on="stem").sort(by="stem")
)
mapper = {k: i for i, k in enumerate(with_cc[target_feature].unique().sort())}
target = with_cc.with_columns(pl.col(target_feature).replace_strict(mapper))[
    target_feature
]
data = with_cc.select(~cs.by_dtype(pl.String))

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.1, random_state=seed
)

# %% Train and run an XGB Classifier
from xgboost import DMatrix

bst = XGBClassifier(
    n_estimators=5,
    max_depth=3,
    learning_rate=1,
    objective="binary:logistic",
    seed=seed,
)
Xd = DMatrix(X_train, label=y_train)
Xd_test = DMatrix(X_test, label=y_test)
model = xgboost.train({"eta": 1, "max_depth": 3, "base_score": 0, "lambda": 0}, Xd, 1)

print("Model error =", np.linalg.norm(y_train - model.predict(Xd)))
print(
    f"Training prediction accuracy {np.round((y_train - np.round(model.predict(Xd)) == 0).sum() / len(y_train), 2)}"
)
acc = f"Accuracy= {np.round((y_test - np.round(model.predict(Xd_test) == 0)).abs().sum() / len(y_test), 2)}"
print(f"Testing accuracy {acc}")
# %% Plot
plt.close()
sns.violinplot(
    data=with_cc,
    y=cell_count_col,
    x="day",
    split=True,
    hue="pert",
    palette="Set2",
)
plt.gca().spines[["right", "top"]].set_visible(False)
plt.savefig(figs_path / "astro3d.svg")


pred = model.predict(Xd, output_margin=True)
explainer = shap.TreeExplainer(
    model, feature_names=[x.replace("_", " ") for x in data.columns]
)
explanation = explainer(Xd)
shap_values = explanation.values
# make sure the SHAP values add up to marginal predictions
# np.abs(shap_values.sum(axis=1) + explanation.base_values - pred).max)
plt.close()
# %% Figure 2: Examples and SHAP values
font = {"family": "sans-serif", "size": 14}

matplotlib.rc("font", **font)

axd = plt.figure(layout="constrained").subplot_mosaic(
    """
    AB
    CC
    """,
)
i = 1
img = imread(pairs[i][1]).max(axis=0)
labels = imread(pairs[i][0]).max(axis=0)

shap.plots.beeswarm(
    explanation,
    max_display=6,
    ax=axd["C"],
    plot_size=None,
    show=False,
    # color_bar=False,
)
axd["C"].set_yticklabels(
    axd["C"].get_yticklabels(),
    # rotation=-30,
    ha="right",
    rotation_mode="anchor",
)
axd["A"].imshow(img)
axd["A"].axis("off")
axd["A"].set_title("Image")
axd["B"].imshow(labels)
axd["B"].axis("off")
axd["B"].set_title("Labels")
# plt.text(0.15, -0.5, acc, fontweight="bold")
from matplotlib.offsetbox import AnchoredText

text_box = AnchoredText(
    acc,
    frameon=False,
    loc=4,
    prop=dict(fontweight="bold", fontsize=10),
    borderpad=0,
)

axd["C"].add_artist(text_box)

axd["C"].set_aspect(0.15)
for ax_id in "AB":
    axd[ax_id].add_artist(
        generate_label(
            ax_id,
            bbox_to_anchor=(-0.2, 1),
            bbox_transform=axd[ax_id].transAxes,
        )
    )
axd["C"].add_artist(
    generate_label("C", bbox_to_anchor=(-0.98, 1.04), bbox_transform=axd["C"].transAxes)
)
font = {"family": "sans-serif", "size": 15}
axd["C"].set_yticklabels(axd["C"].get_yticklabels(), fontdict=font)
axd["C"].tick_params(axis="y", which="major", pad=-16)
axd["C"].set_xlabel(axd["C"].get_xlabel(), fontdict=font)
plt.savefig(figs_path / "example_shap.svg")
plt.close()
