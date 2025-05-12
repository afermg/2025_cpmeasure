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
from jump_portrait.s3 import get_image_from_s3uri
from PIL import Image
from tqdm import tqdm

out_dir = Path("/datastore/alan/cp_measure/jump")

source_table = (
    "https://zenodo.org/api/records/15359196/files/original_selection.csv/content"
)
id_col = "Perturbation"
max_samples_per_group = 5

items = pl.scan_csv(source_table).select(id_col).collect().to_series().to_list()

locations = list(
    Parallel(n_jobs=-1)(delayed(get_item_location_info)(x) for x in tqdm(items))
)  # May take a minute
locations_df = pl.concat(locations, how="diagonal")

subsample = locations_df.filter(
    pl.int_range(pl.len()).shuffle().over("standard_key") < max_samples_per_group
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

images_df.write_parquet(out_dir / ".." / "image_index.parquet")


def download_save(row: tuple[str, str, str, str]):
    pert_name, *meta_vals = row[:n_meta_cols]
    channel, uri, rep = row[n_meta_cols:]
    # Format: "PERTNAME_CHANNEL_REPLICATE__SOURCE__PLATE__BATCH__WELL__SITE.tif"
    # Note that the double underscore (__) splits the original ids AND the id set for this subset of the data.
    filename = out_dir / f"{pert_name}_{channel}_{rep}__{'__'.join(meta_vals)}.tif"

    img = get_image_from_s3uri(uri)
    pil_img = Image.fromarray(img)

    pil_img.save(filename)  # , compression="lzma")

    return 1


out_dir.mkdir(parents=True, exist_ok=True)

responses = list(
    Parallel(n_jobs=-1)(delayed(download_save)(item) for item in tqdm(images_df.rows()))
)

from skimage.io import imread

imread(
    "../../../../../../../../../datastore/alan/cp_measure/test/jump/AGMAT_ER_01__source_13__20221109_Run5__CP-CC9-R5-10__D20__2.tif"
)
