#!/usr/bin/env python
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from cp_measure.bulk import get_core_measurements
from joblib import Parallel, delayed
from pooch import Untar, retrieve
from skimage.io import imread
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

MEASUREMENTS = get_core_measurements()

overwrite = False
profiles_fpath = Path("tmp.parquet")
fig_fpath = Path(".") / ".." / ".." / "figs" / "axon3d.svg"
fig_fpath.parent.mkdir(parents=True, exist_ok=True)

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
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day3_deconv.tar.gz": "6d1fd24d7f9381e1aa4f95dbc9c2c351c7633a3a4500d95b4a6736d40b4100af",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day3_masks.tar.gz": "d428491b3bc8207bedf77f33afdde5e6542ea6c4f9f63e0cf920f46b6f68f43b",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day5_deconv.tar.gz": "f7c7f93d1a173441f8b116ca8411df524678d926ed30d490d580d01edd782503",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day5_masks.tar.gz": "ddbf48a883f03ab89c3adf11cbf4b1e07905f8deb734502f38b684bb009e900b",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day7_deconv.tar.gz": "9dd69934f306cab31e94df427ee7618d21c32038a62a14ccab15b7d3517eadf7",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/nha_day7_masks.tar.gz": "55aafc7b49f904f711187a85ef0bf5e410d3e6aef4b0b7bf69ef17fce51ee8b6",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day3_deconv.tar.gz": "2e1637fd0f240fda6b19ea7d57668ce6b82ae35f697fa2af9b30fc6915d609b5",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day3_masks.tar.gz": "e47c7b476f1328f79f4d6825c74706fc7e5504ca3eed7d41caa7199b3d001690",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day5_deconv.tar.gz": "08c7278e4c1a61fa77b8c196e79bcd5e222241ff05b492d13a18a9f976c81cf3",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day5_masks.tar.gz": "366293f5e68247e927628b6b2e8a8c33338de9a56db9f269e7fb6a0500dbb55b",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day7_deconv.tar.gz": "022e8d8d086e65551d1fabf8048bbf0fe0a6fbe9015bdc6a6191227730ad8408",
    "https://www.socr.umich.edu/data/3d-cell-morphometry-data/vpa_day7_masks.tar.gz": "63bf6d123ee9505a51fb0ef099ba77223a800babab51756d577cd7cfd8dd182d",
}

retrieved = list(
    Parallel(n_jobs=-1)(
        delayed(retrieve)(
            url,
            processor=Untar(extract_dir=Path(url).name.split(".")[0]),
            known_hash=h,
            progressbar=True,
        )
        for url, h in url_hash.items()
    )
)

if not profiles_fpath.exists() or overwrite:
    dir_files = {
        x: y
        for x, y in zip((Path(url).name.split(".")[0] for url in url_hash), retrieved)
        if x.endswith("_masks")  # Optional but makes the next step faster
    }

    pairs = [
        x
        for x in Parallel(n_jobs=-1)(
            delayed(map_mask_to_image)(x) for x in chain(*dir_files.values())
        )
        if x is not None  # Ignore pngs
    ]

    dfs = list(
        Parallel(n_jobs=-1)(
            delayed(apply_measurements)(m_path, i_path) for m_path, i_path in pairs
        )
    )

    profiles = pl.concat(dfs)
    profiles.write_parquet(profiles_fpath)
else:
    profiles = pl.read_parquet(profiles_fpath)

# %% Train and run an XGB Classifier
target_feature = "day"

bst = XGBClassifier(
    n_estimators=5,
    max_depth=3,
    learning_rate=1,
    objective="binary:logistic",
    seed=seed,
)


meta = profiles.select(cs.by_dtype(pl.String))
cc = meta.group_by("stem").agg(
    pl.len().alias("ncells"), pl.first("pert"), pl.first("day")
)


with_cc = (
    profiles.select(~cs.by_dtype(pl.String), "stem").join(cc, on="stem").sort(by="stem")
)
mapper = {k: i for i, k in enumerate(with_cc[target_feature].unique().sort())}
target = with_cc.with_columns(pl.col(target_feature).replace_strict(mapper))[
    target_feature
]
data = with_cc.select(~cs.by_dtype(pl.String))

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.1, random_state=seed
)
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
test_acc = (preds & np.array(y_test)).sum() / len(y_test)
print(test_acc)

# %% Plot
plt.close()
sns.violinplot(
    data=with_cc,
    y="ncells",
    x=("pert", "day")[~("pert", "day").index(target_feature)],
    split=True,
    hue=target_feature,
    palette="RdBu",
)
plt.savefig(fig_fpath)
