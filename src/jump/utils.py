from pathlib import Path

import numpy as np
import polars as pl
from cp_measure.bulk import (
    get_core_measurements,
    # get_correlation_measurements,
    # get_multimask_measurements,
)
from skimage.io import imread

MEASUREMENTS = get_core_measurements()


def read_labels(mask_path: Path):
    if str(mask_path).endswith("npz"):
        labels = np.load(mask_path)["arr_0"]
    else:
        labels = imread(mask_path)
    return labels


def apply_measurements(
    mask_path: Path, img_path: Path, object_name: str = None
) -> pl.DataFrame:
    gene, site, channel = img_path.stem.split("_")[:3]
    if object_name is None:
        object_name = mask_path.stem.split("_")[2]

    labels = read_labels(mask_path)

    img = imread(img_path)
    if img.ndim == 3:
        flat_img = img.max(axis=0)
        flat_img = flat_img / flat_img.max()
    d = {}
    for meas_name, meas_f in MEASUREMENTS.items():
        measurements = meas_f(labels, img)
        # Unpack output dictionaries
        for k, v in measurements.items():
            d[k] = v
            d["object"] = object_name
            d["gene"] = gene
            d["site"] = site
            d["channel"] = channel

    df = pl.from_dict(d)

    return df


def get_keys(fpath: Path, n: int = 2) -> tuple[str]:
    return tuple(fpath.stem.split("_")[:n])


def apply_measurements_type2(
    mask_path: Path, pixels1_path: Path, pixels2_path: Path
) -> pl.DataFrame:
    labels = read_labels(mask_path)
    pixels1 = imread(pixels1_path)
    pixels2 = imread(pixels2_path)

    for meas_name, meas_f in MEASUREMENTS_TYPE2.items():
        measurements = meas_f(labels, pixels1, pixels1)
        # Unpack output dictionaries
        for k, v in measurements.items():
            d[k] = v
            # for k, v in zip(("pert", "day", "stem"), meta):
            #     d[k] = v
            d["object"] = mask_path.stem.split("_")[2]
            gene, site, channel1 = pixels1_path.stem.split("_")[:3]
            d["gene"] = gene
            d["site"] = site
            channel2 = pixels2_path.stem.split("_")[2]
            d["channel"] = (channel1, channel2)
