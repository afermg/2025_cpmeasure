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

output_dir = Path("/datastore/alan/cp_measure/")
img_dir = output_dir / "jump_subset"
mask_dir = output_dir / "jump_masks"
profiles_dir = output_dir / "profiles_via_masks"
for folder in (img_dir, mask_dir, profiles_dir):
    folder.mkdir(exist_ok=True, parents=True)


MEASUREMENTS = get_core_measurements()


def get_keys(fpath: Path, n: int = 2) -> tuple[str]:
    return tuple(fpath.stem.split("_")[:n])


def segment_save(mask: np.ndarray, name: str) -> str:
    fpath = out_dir / f"shape_{mask.shape[1]}" / f"{name}.npz"
    fpath.parent.mkdir(parents=True, exist_ok=True)
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


def read_labels(mask_path: Path):
    if str(mask_path).endswith("npz"):
        labels = np.load(mask_path)["arr_0"]
    else:
        labels = imread(mask_path)
    return labels


def apply_measurements(mask_path: Path, img_path: Path) -> pl.DataFrame:
    gene, site, channel = img_path.stem.split("_")[:3]
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
            d["object"] = mask_path.stem.split("_")[2]
            d["gene"] = gene
            d["site"] = site
            d["channel"] = channel

    df = pl.from_dict(d)

    return df


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

batch_size = len(loaded_imgs) // ngpus
with ThreadPoolExecutor(ngpus) as ex:
    masks = list(
        chain(
            *ex.map(
                lambda x: model[x].eval(
                    loaded_imgs[x * batch_size : (x + 1) * batch_size]
                )[0],
                range(ngpus),
            )
        )
    )
# masks = list(chain(*masks))
# Organize masks
mask_ids = list(
    Parallel(n_jobs=-1)(
        delayed(segment_save)(mask, "_".join(get_keys(fpath, 3)))
        for mask, fpath in tqdm(zip(masks, flat_segmentable_files))
    )
)


# We pair the mask names with their associated images
pairs = [
    (
        [("_".join(k) + f"_{suffix}.npz") for suffix in segmentable_channels],
        sorted(v),
    )
    for k, v in groupby(img_paths, get_keys)
]

# Check that all image sets are the same size
assert len(set([len(x[1]) for x in pairs])) == 1, "Heterogeneous number of channels"

# Get the product of all the mask-image combinations
combs = list(chain(*[list(product(*xy)) for xy in pairs]))

# Check that shapes match
for mask, img in combs:
    assert (
        np.load(out_dir / img.parent.name / mask)["arr_0"].shape == imread(img).shape
    ), "error"
# table = table.append_column(
#     "object",
#     pa.array([step_name.split("_")[-1]] * len(table), pa.string()),
# )
# table = table.append_column(
#     "object",
type2_pairs = [list(product((k, combinations(v, 2)))) for k, v in pairs]
type2_pairs = [list(product((k, combinations(v, 2)))) for k, v in pairs]
type2_pairs = [list(product((k, combinations(v, 2)))) for k, v in pairs]
measure_results = Parallel(n_jobs=-1)(
    delayed(apply_measurements)(out_dir / img.parent.name / m, img)
    for m, img in tqdm(combs)
)
first_set = pl.concat(measure_results)
first_set.write_parquet(profiles_dir / "first_set.parquet")
# multichannel

MEASUREMENTS_TYPE2 = get_correlation_measurements()

# multimask
