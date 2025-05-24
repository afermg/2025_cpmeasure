#!/usr/bin/env jupyter
"""
Compare Cellprofiler and cp_measure data.
Plot the comparison.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import polars as pl
import polars_ds as pds
import pooch
import seaborn as sns
from duckdb.typing import VARCHAR
from parse_features import get_feature_groups
from polars import selectors as cs
from pooch import Unzip
from util_names import get_cpm_to_measurement_mapper
from util_plot import generate_label

figs_dir = Path("..") / ".." / "figs"

cpmeasure_parquet = "/datastore/alan/cp_measure/profiles_via_masks/first_set.parquet"


# Turns out it is easier to trim down the CellProfiler measurements
# into cp_measure than anything else
def trim_features(name: str) -> str:
    """
    Trim object and channel information from CellProfiler
    """
    replacement_ch = [f"_Orig{x}" for x in ("DNA", "AGP", "ER", "Mito", "RNA")]
    replacement_mea = ["AreaShape_", "Texture_"]

    for ch in replacement_ch:
        name = name.replace(ch, "")
    for mea in replacement_mea:
        name = name.replace(mea, "")
    return name


con = duckdb.connect()
con.create_function("trim_features", trim_features, [VARCHAR], VARCHAR)

cp_data = pooch.retrieve(
    "https://zenodo.org/api/records/15505477/files/cellprofiler_analysis.zip/content",
    known_hash="1ca6c08955336d15832fc6dc5c15000990f4dd4733e47a06030a416e7ac7a3e9",
    processor=Unzip(),
)

all_csv_files = [x for x in cp_data if x.endswith("csv")]
csv_files = {
    Path(x).stem: x
    for x in cp_data
    if Path(x).stem.split("_")[-1] in ("Cells", "Nuclei", "Image")
}

original_tables_d = {
    k: con.sql(f"SELECT * FROM '{v}'").pl().with_columns(pl.lit(k).alias("object"))
    for k, v in csv_files.items()
}
# %%
feature_tables = [v for k, v in original_tables_d.items() if not k.endswith("Image")]
common = set(feature_tables[0].columns)
for table in feature_tables[1:]:
    common &= set(table.columns)
orig_profiles = pl.concat([
    x.select(common).with_columns(pl.col("object").str.strip_prefix("cp_measure_"))
    for x in feature_tables
])
orig_consensus = (
    orig_profiles.group_by("ImageNumber", "object")
    .median()
    .select(pl.exclude("ObjectNumber"))
)
cols = (
    orig_consensus.select(~cs.by_dtype(pl.String))
    .select(pl.exclude("object", "^Metadata.*$"))
    .columns
)
parsed = get_feature_groups(cols, ("feature", "channel", "suffix")).with_columns(
    pl.col("channel").str.strip_prefix("Orig")
)
# %%
orig_unpivot = orig_consensus.unpivot(
    index=("ImageNumber", "object"), variable_name="fullname"
)
with_parsed = con.sql("SELECT * FROM orig_unpivot NATURAL JOIN parsed")

with_trimmed = con.sql("SELECT *,trim_features(fullname) AS cpm_id FROM with_parsed")
image_table = original_tables_d["cp_measure_Image"]
imageid_mapper = con.sql(
    "(SELECT ImageNumber,split_part(FileName_OrigDNA, '_', 2) AS site,split_part(FileName_OrigDNA, '_', 1) AS gene FROM image_table)"
)
cellprof_tidy = con.sql(
    "SELECT * EXCLUDE(value), CAST(value AS DOUBLE) as CellProfiler FROM with_trimmed NATURAL JOIN imageid_mapper"
)

# %% cp_measure

mapper = get_cpm_to_measurement_mapper()

cp_df = pl.read_parquet(cpmeasure_parquet)

meta_cols = ("object", "gene", "channel", "site")
new_consensus = cp_df.group_by(meta_cols).median()
# %%
chless_feats = [
    x
    for x in new_consensus.select(~cs.by_dtype(pl.String)).columns
    if "Intensity" not in x
    and "Zernike" not in x
    and "Granularity" not in x
    # and "Radial" not in x
    and "Difference" not in x
    and "InfoMeas" not in x
    and not x.startswith("RadialDistribution")
    and not x.startswith("Sum")
    and not x.startswith("Entropy")
    and not x.startswith("Contrast")
    and not x.startswith("AngularSecondMoment")
    and not x.startswith("Correlation")
    and not x.startswith("Variance")
]
new_comp = new_consensus.with_columns(
    pl.col("object")
    .replace({"DNA": "Nuclei", "AGP": "Cells", "cell": "Cells"})
    .str.to_titlecase()
)
unpivoted = new_comp.unpivot(index=meta_cols, variable_name="cpm_id")
dups = unpivoted.with_columns(
    pl.when(pl.col("cpm_id").is_in(chless_feats))
    .then(pl.lit(""))
    .otherwise(pl.col("channel"))
    .alias("channel")
)
uniq = dups.unique()
# %% combine
cols_to_sort = (*meta_cols, "cpm_id")
cols_to_print = (*cols_to_sort, "CellProfiler", "cp_measure")
merged = con.sql(
    "SELECT *, value AS cp_measure FROM uniq NATURAL JOIN cellprof_tidy ORDER BY object, gene, channel, site, cpm_id"
).pl()
# %% Plot

mpd = merged
plt.close()
g = sns.FacetGrid(
    mpd.to_pandas(),
    col="cpm_id",
    col_wrap=4,
    hue="object",
    sharex=False,
    sharey=False,
    legend_out=False,
    hue_kws={"markers": "channel"},
)
g.map(sns.scatterplot, "CellProfiler", "cp_measure", alpha=0.05)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.tight_layout()
plt.savefig(figs_dir / "grid_cp_vs_cpm.svg")
plt.savefig(figs_dir / "grid_cp_vs_cpm.png")

# %%
partitioned = {
    k: v.filter(~pl.col("cp_measure").is_nan())
    for k, v in merged.partition_by("cpm_id", as_dict=True).items()
}
dfs = []
for feat_name, x in partitioned.items():
    if len(x):
        dfs.append(
            x.select(
                pds.lin_reg_report(
                    *["CellProfiler"],
                    target="cp_measure",
                ).alias("result")
            )
            .unnest("result")
            .with_columns(
                pl.lit(feat_name[0]).alias("Feature")
            )  # , pl.lit(obj).alias("object"))
            .with_columns(pl.col("Feature").replace(mapper).alias("Measurement"))
        )
res = pl.concat(dfs)
# %%
plt.close()
feats_to_show = [
    "Area",
    "Intensity_MedianIntensity",
    "RadialDistribution_RadialCV_1of4",
]
axd = plt.figure(layout="constrained").subplot_mosaic(
    """
    AD
    BD
    CD
    """
)
g = sns.swarmplot(
    data=res.sort("Measurement").to_pandas(),
    x="Measurement",
    y="r2",
    ax=axd["D"],
    alpha=0.8,
    hue="Measurement",
)

# g.set_ylim(0.989, 1.002)
g.set_ylim(0, 1)
g.set_title("Linear fit")
# g.set_xlabel("Individual features")
g.set_ylabel("R squared")

pad = 0.25
axd["D"].add_artist(generate_label("D", pad=pad))
g.set_xticklabels(g.get_xticklabels(), rotation=15, rotation_mode="anchor", ha="right")
for ax_id, featname in zip("ABC", feats_to_show):
    ax = axd[ax_id]
    h = sns.scatterplot(
        merged.filter(pl.col("cpm_id") == featname).to_pandas(),
        x="CellProfiler",
        y="cp_measure",
        hue="object",
        ax=ax,
        alpha=0.1,
        palette=sns.color_palette("husl", 8),
        legend=None if ax_id != "C" else True,
    )
    # ax.text(0, 0, ax_id, fontsize=9)
    h.set_yticklabels(
        h.get_yticklabels(), rotation=30, ha="right", rotation_mode="anchor"
    )
    featname = featname.removeprefix("RadialDistribution_")
    featname = featname.removeprefix("Intensity_")
    ax.set_title(featname.replace("_", " "))
    sns.despine()
    ax.add_artist(generate_label(ax_id, pad=pad))
    if ax_id == "C":
        for lh in ax.get_legend().legend_handles:
            lh.set_alpha(1)

        sns.move_legend(ax, loc="lower right", bbox_to_anchor=(1.12, 0))
plt.savefig(figs_dir / "jump_r2_examples.svg")
plt.savefig(figs_dir / "jump_r2_examples.png")
