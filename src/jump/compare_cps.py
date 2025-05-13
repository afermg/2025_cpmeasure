"""
Compare CellProfiler and cp_measure metrics
"""

from pathlib import Path

import numpy as np
import polars as pl
from cellpose import models
from cp_measure.bulk import (
    get_core_measurements,
    get_correlation_measurements,
    get_multimask_measurements,
)
from joblib import Parallel, delayed
from skimage.io import imread
from tqdm import tqdm

img_dir = Path("/datastore/alan/cp_measure/jump_subset/images")
out_dir = Path("/datastore/alan/cp_measure/jump_masks")
out_dir.mkdir(exist_ok=True, parents=True)


def segment_save(mask: np.ndarray, name: str) -> str:
    fpath = out_dir / f"{name}.npz"
    np.savez(fpath, mask)
    return fpath.stem.split("_")[:3]


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
    for x in Parallel()(delayed(try_imread)(x) for x in img_dir.glob("*.tif"))
    if isinstance(x, str)
])
assert not n_errors

segmentable_files = [
    str(x)
    for x in img_dir.glob("*.tif")
    if Path(x).name.split("_")[2] in segmentable_channels
    # and not (Path(x).name.startswith("AASDHPPT") and str(x)[45:47] == "01")
]

loaded_imgs = list(
    Parallel(n_jobs=-1)(delayed(imread)(x) for x in tqdm(segmentable_files))
)

masks_flows = model.eval(loaded_imgs)

mask_ids = list(
    Parallel(n_jobs=-1)(
        delayed(segment_save)(mask, Path(filepath).stem)
        for mask, filepath in tqdm(zip(masks_flows[0], segmentable_files))
    )
)


def apply_measurements(mask_path: str, img_path: str) -> pl.DataFrame:
    meta = Path(img_path).parent.name.split("_")[:2]
    meta = (*meta, Path(img_path).stem)

    # labels = imread(mask_path)
    # labels_redz = labels.max(axis=0)
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
