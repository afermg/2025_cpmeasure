#!/usr/bin/env jupyter
"""
Compare CellProfiler and cp_measure metrics
"""

from itertools import chain, combinations, groupby, product
from pathlib import Path

import polars as pl
import pooch
from joblib import Parallel, delayed
from pooch import Unzip
from skimage.io import imread
from tqdm import tqdm
from utils import apply_measurements, apply_measurements_2, get_keys, read_labels


def get_mask_name(x):
    return x.parent.name.split("_")[0]


cp_data = pooch.retrieve(
    "https://zenodo.org/api/records/15480187/files/cellprofiler_analysis.zip/content",
    known_hash="42eb17d4b96863815f52764ac6cef5e03f5cd8020697ced7eecca1db7ff25482",
    processor=Unzip(),
)
mask_files = [Path(x) for x in cp_data if "masks" in x]
# %%

out_dir = Path("/datastore/alan/cp_measure/")
img_dir = out_dir / "jump_subset"
profiles_dir = out_dir / "profiles_via_masks"
for folder in (img_dir, profiles_dir):
    folder.mkdir(exist_ok=True, parents=True)


channels = ["DNA", "AGP", "Mito", "RNA", "ER"]


def try_imread(x):
    try:
        return imread(x)
    except:
        return str(x)


img_paths = sorted(list(img_dir.rglob("*.tif")))


# %% Get type 1 measurements

# We pair the mask names with their associated images

sorted_mask_files = sorted(
    mask_files, key=lambda x: "_".join(x.name.split("_")[:3]) + x.parent.name
)
grouped_masks = {k: list(v) for k, v in groupby(sorted_mask_files, get_keys)}
pairs = [(grouped_masks[k], sorted(v)) for k, v in groupby(sorted(img_paths), get_keys)]

# Check that all image sets are the same size
assert len(set([len(x[1]) for x in pairs])) == 1, "Heterogeneous number of channels"

# Get the product of all the mask-image combinations
combs = list(chain(*[list(product(*xy)) for xy in pairs]))

# Check that shapes match
for mask, img in combs:
    assert read_labels(mask).shape == imread(img).shape, "error"

measure_results = Parallel(n_jobs=100)(
    delayed(apply_measurements)(m, img, get_mask_name(m)) for m, img in tqdm(combs)
)
first_set_rad = pl.concat(measure_results)
first_set_rad.write_parquet(profiles_dir / "first_set.parquet")

# %% Type 2
triads = [  # Triad with mask, img1, img2
    (mask_ind, *pair_)
    for masks, channels in pairs
    for mask_ind, pair_ in product(masks, combinations(channels, 2))
]
for m, img1, img2 in triads:
    shape = try_imread(m).shape
    assert shape == try_imread(m).shape and shape == try_imread(img2).shape, "error"
# measure_results = Parallel(n_jobs=100)(
#     delayed(apply_measurements_2)(m, img1, img2, get_mask_name(m))
#     for m, img1, img2 in tqdm(triads[:5])
# )
# %%
for m, img1, img2 in tqdm(triads[:5]):
    tmp = apply_measurements_2(m, img1, img2, get_mask_name(m))
# measure_results = Parallel(n_jobs=100)(
#     delayed(apply_measurements_2)(m, img1, img2, get_mask_name(m))
# )
