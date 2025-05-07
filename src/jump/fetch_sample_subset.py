# /// script
# dependencies = [
#   "jump_portrait<3",
# ]
# ///
"""
Pull images from JUMP based on Shata's selection of significant features.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from jump_portrait.fetch import get_item_location_info
from jump_portrait.s3 import get_image_from_s3uri
from PIL import Image

out_dir = Path("/datastore/alan/cp_measure/jump_subset")

source_table = "~/Documents/broad/drafts/papers/2025_cp_measure/src/jump/test_set.csv"
id_col = "Perturbation"
max_samples_per_group = 5

items = tuple(pl.scan_csv(source_table).select(id_col).collect().to_series().to_list())

locations = [get_item_location_info(x) for x in items]  # May take a couple of minutes
locations_df = pl.concat(locations, how="diagonal")

subsample = locations_df.filter(
    pl.int_range(pl.len()).shuffle().over("standard_key") < max_samples_per_group
)

# The regex below filters out bright field
meta_cols = ( "standard_key", *[f"Metadata_{x}" for x in ("Source","Batch","Plate","Well","Site")])
orig = subsample.select(pl.col("^URL_Orig[DARME].*$",*meta_cols))
images_df = orig.unpivot(index=("standard_key", *[x for x in orig.columns if x.startswith("Metadata")]), variable_name="channel").with_columns(
    pl.col("channel").str.strip_prefix("URL_Orig"),
    pl.int_range(1, pl.len() + 1)
    .over("standard_key", "channel")
    .cast(str)
    .str.pad_start(2, "0")
    .alias("rep"),
)

n_meta_cols = len(meta_cols)
def download_save(row: tuple[str, str, str, str]):
    pert_name, *meta_vals = row[:n_meta_cols]
    channel, uri, rep = row[n_meta_cols:]
    # Format: "PERTNAME_CHANNEL_REPLICATE__SOURCE__PLATE__BATCH__WELL__SITE.tif"
    # Note that the double underscore (__) splits the original ids AND the id set for this subset of the data.
    filename = out_dir / f"{pert_name}_{channel}_{rep}__{'__'.join(meta_vals)}.tif"
    
    img = get_image_from_s3uri(uri)
    pil_img = Image.fromarray(img)
    
    pil_img.save(filename, compression= "lzma")
    
    return 1


out_dir.mkdir(parents=True,exist_ok=True)
def run():
    with ThreadPoolExecutor() as ex: # 
        responses = list(ex.map(download_save, images_df.rows()))

%timeit  run()
# /// script
# dependencies = [
#   "jump_portrait<3",
# ]
# ///
"""
Pull images from JUMP based on Shata's selection of significant features.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from jump_portrait.fetch import get_item_location_info
from jump_portrait.s3 import get_image_from_s3uri
from PIL import Image

out_dir = Path("/datastore/alan/cp_measure/jump_subset")

source_table = "~/Documents/broad/drafts/papers/2025_cp_measure/src/jump/test_set.csv"
id_col = "Perturbation"
max_samples_per_group = 5

items = tuple(pl.scan_csv(source_table).select(id_col).collect().to_series().to_list())

locations = [get_item_location_info(x) for x in items]  # May take a couple of minutes
locations_df = pl.concat(locations, how="diagonal")

subsample = locations_df.filter(
    pl.int_range(pl.len()).shuffle().over("standard_key") < max_samples_per_group
)

# The regex below filters out bright field
meta_cols = ( "standard_key", *[f"Metadata_{x}" for x in ("Source","Batch","Plate","Well","Site")])
orig = subsample.select(pl.col("^URL_Orig[DARME].*$",*meta_cols))
images_df = orig.unpivot(index=("standard_key", *[x for x in orig.columns if x.startswith("Metadata")]), variable_name="channel").with_columns(
    pl.col("channel").str.strip_prefix("URL_Orig"),
    pl.int_range(1, pl.len() + 1)
    .over("standard_key", "channel")
    .cast(str)
    .str.pad_start(2, "0")
    .alias("rep"),
)

n_meta_cols = len(meta_cols)
def download_save(row: tuple[str, str, str, str]):
    pert_name, *meta_vals = row[:n_meta_cols]
    channel, uri, rep = row[n_meta_cols:]
    # Format: "PERTNAME_CHANNEL_REPLICATE__SOURCE__PLATE__BATCH__WELL__SITE.tif"
    # Note that the double underscore (__) splits the original ids AND the id set for this subset of the data.
    filename = out_dir / f"{pert_name}_{channel}_{rep}__{'__'.join(meta_vals)}.tif"
    
    img = get_image_from_s3uri(uri)
    pil_img = Image.fromarray(img)
    
    pil_img.save(filename, compression= "lzma")
    
    return 1


out_dir.mkdir(parents=True,exist_ok=True)
with ThreadPoolExecutor() as ex: # 
    responses = list(ex.map(download_save, images_df.rows()))

