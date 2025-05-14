"""
Compare CellProfiler and cp_measure metrics
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import chain, groupby, product
from pathlib import Path

import numpy as np
import polars as pl
import torch
from cellpose import models
from cp_measure.bulk import (
    get_core_measurements,
    get_correlation_measurements,
    get_multimask_measurements,
)
from joblib import Parallel, delayed
from skimage.io import imread
from tqdm import tqdm

img_dir = Path("/datastore/alan/cp_measure/jump_subset")
out_dir = Path("/datastore/alan/cp_measure/jump_masks")
out_dir.mkdir(exist_ok=True, parents=True)
img_dir.mkdir(exist_ok=True, parents=True)


MEASUREMENTS = get_core_measurements()


def get_keys(fpath: Path, n: int = 2) -> tuple[str, str]:
    return tuple(fpath.stem.split("_")[:n])


def segment_save(mask: np.ndarray, name: str) -> str:
    fpath = out_dir / f"{name}.npz"
    np.savez(fpath, mask)
    return fpath


channels = ["DNA", "AGP", "Mito", "RNA", "ER"]
from itertools import combinations

segmentable_channels = ("DNA", "AGP")
mask_dir = None  # Path("")
model = models.CellposeModel(gpu=True)


# cellprof_profiles = pl.read_parquet("")
# AASDHPPT_AGP_01__source_4__2021_05_10_Batch3__BR00123616__N19__4.tif
def try_imread(x):
    try:
        return imread(x)
    except:
        return str(x)


n_errors = len([
    x
    for x in Parallel()(delayed(try_imread)(x) for x in img_dir.rglob("*.tif"))
    if isinstance(x, str)
])
assert not n_errors
img_paths = sorted(list(img_dir.rglob("*.tif")))

segmentable_files = [
    x
    for x in img_paths
    if Path(x).name.split("_")[2] in segmentable_channels
    # and not (Path(x).name.startswith("AASDHPPT") and str(x)[45:47] == "01")
]


def apply_measurements(mask_path: Path, img_path: Path) -> pl.DataFrame:
    # meta = Path(img_path).parent.name.split("_")[:2]
    # meta = (*meta, Path(img_path).stem)

    if mask_path.endswith("npz"):
        labels = np.load(mask_path)["arr_0"]
    else:
        labels = imread(mask_path)
    # if gaps:
    #     labels = labels.max(axis=0)
    # Cover case where the reduction on z removes an entire item
    # Unlike in other cases, here we assume that the input mask can have gaps
    # labels = fix_non_continuous_labels(labels_redz)

    img = imread(img_path)
    if img.ndim == 3:
        flat_img = img.max(axis=0)
        flat_img = flat_img / flat_img.max()
    d = {}
    # mask = imread(mask_path)
    for meas_name, meas_f in MEASUREMENTS.items():
        measurements = meas_f(labels, img)
        # Unpack output dictionaries
        for k, v in measurements.items():
            d[k] = v
            # for k, v in zip(("pert", "day", "stem"), meta):
            #     d[k] = v
            d["object"] = mask_path.stem.split("_")[2]
            gene, site, channel = img_path.stem.split("_")[:3]
            d["gene"] = gene
            d["site"] = site
            d["channel"] = channel

    df = pl.from_dict(d)

    return df


# %%
d = dict(
    sorted(
        [(k, list(v)) for k, v in groupby(segmentable_files, get_keys)],
        key=lambda x: x[0],
    )
)
flat_segmentable_files = list(chain(*d.values()))
loaded_imgs = list(
    Parallel(n_jobs=-1)(delayed(imread)(x) for x in flat_segmentable_files)
)


# Divide and conquer (using both GPUs)
ngpus = torch.cuda.device_count()
model = [models.CellposeModel(device=torch.device(i)) for i in range(ngpus)]

with ThreadPoolExecutor(ngpus) as ex:
    masks = list(
        ex.map(
            lambda x: model[x].eval(
                loaded_imgs[x : (x + 1) * (len(loaded_imgs) // ngpus)]
            )[0],
            range(ngpus),
        )
    )
# Organize masks
mask_ids = list(
    Parallel(n_jobs=-1)(
        delayed(segment_save)(mask, "_".join(get_keys(fpath, 3)))
        for mask, fpath in tqdm(zip(masks, flat_segmentable_files))
    )
)

for mask, fpath in zip(masks, flat_segmentable_files):
    segment_save(mask, "_".join(get_keys(fpath, 3)))

pairs = [
    (
        [
            Path(out_dir) / ("_".join(k) + f"_{suffix}.npz")
            for suffix in segmentable_channels
        ],
        sorted(v),
    )
    for k, v in groupby(img_paths, get_keys)
]
combs = list(chain(*[list(product(*xy)) for xy in pairs]))

for mask, img in combs:
    assert np.load(mask)["arr_0"].shape == imread(img).shape, "error"

measure_results = Parallel(n_jobs=-1)(
    delayed(apply_measurements)(m, img) for m, img in tqdm(combs)
)


# Do the combinatorials
# Get features
# Bring together
# Get pairs of channels
# Get multichannel features
# Bring together
