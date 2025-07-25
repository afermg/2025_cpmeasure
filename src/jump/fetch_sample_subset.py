# /// script
# dependencies = [
#   "jump_portrait<3",
# ]
# ///
"""
Pull images from JUMP based on Shata's selection of significant features.
"""

from pathlib import Path

import polars as pl
from joblib import Parallel, delayed
from jump_portrait.fetch import get_item_location_info
from skimage.io import imread
from tqdm import tqdm

out_dir = Path("/datastore/alan/cp_measure/jump_subset")
out_dir.mkdir(parents=True, exist_ok=True)
source_table = (
    "https://zenodo.org/api/records/15359196/files/original_selection.csv/content"
)
id_col = "Perturbation"
max_samples_per_group = 2
seed = 1

items = pl.scan_csv(source_table).select(id_col).collect().to_series().to_list()

locations = list(
    Parallel(n_jobs=-1)(delayed(get_item_location_info)(x) for x in tqdm(items))
)  # May take a minute
locations_df = pl.concat(locations, how="diagonal")

subsample = locations_df.filter(
    pl.int_range(pl.len()).shuffle(seed=seed).over("standard_key")
    < max_samples_per_group
)

# The regex below filters out bright field
meta_cols = (
    "standard_key",
    *[f"Metadata_{x}" for x in ("Source", "Batch", "Plate", "Well", "Site")],
)
orig = subsample.select(pl.col("^URL_Orig[DARME].*$", *meta_cols))
images_df = orig.unpivot(
    index=("standard_key", *[x for x in orig.columns if x.startswith("Metadata")]),
    variable_name="channel",
).with_columns(
    pl.col("channel").str.strip_prefix("URL_Orig"),
    pl.int_range(1, pl.len() + 1)
    .over("standard_key", "channel")
    .cast(str)
    .str.pad_start(2, "0")
    .alias("rep"),
)

n_meta_cols = len(meta_cols)

# Save index of images that maps to original files
images_df.write_parquet(out_dir / ".." / "to_upload" / "image_index.parquet")


def download_save(row: tuple[str, str, str, str]):
    pert_name, *meta_vals = row[:n_meta_cols]


# Validate


for shape in (1080, 1280):
    fpath = Path(f"/datastore/alan/cp_measure/jump_subset/shape_{shape}/")
    for file in fpath.glob("*.tif"):
        tmp = imread(file).shape[1]
        assert tmp == shape, "va"
