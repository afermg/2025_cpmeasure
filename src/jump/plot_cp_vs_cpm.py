#!/usr/bin/env jupyter
"""
Compare Cellprofiler and cp_measure data.
Plot the comparison.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import polars as pl
import polars as pls
import pooch
import seaborn as sns
from duckdb.typing import VARCHAR
from parse_features import get_feature_groups
from polars import selectors as cs
from pooch import Unzip
from util_plot import generate_label

figs_dir = Path("..") / ".." / "figs"

# cpmeasure_parquet = "/datastore/alan/cp_measure/profiles/first_set.parquet"
cpmeasure_parquet = "/datastore/alan/cp_measure/profiles_via_masks/first_set.parquet"


# Turns out it is easier to trim down the CellProfiler measurements
# into cp_measure than anything else
def trim_features(name: str) -> str:
    """
    Trim object and channel information from CellProfiler
    """
    replacement_obj = ("Cells_", "Image_", "Nuclei_", "Cytoplasm_")
    replacement_ch = [f"_Orig{x}" for x in ("DNA", "AGP", "ER", "Mito", "RNA")]
    replacement_mea = ["AreaShape_"]

    for obj in replacement_obj:
        name = name.replace(obj, "")
    for ch in replacement_ch:
        name = name.replace(ch, "")
    for mea in replacement_mea:
        name = name.replace(mea, "")
    return name


con = duckdb.connect()
con.create_function("trim_features", trim_features, [VARCHAR], VARCHAR)

cp_data = pooch.retrieve(
    "https://zenodo.org/api/records/15426610/files/cellprofiler_analysis.zip/content",
    known_hash="aef261cf87e5f138ef2dd91b9ed57add1a4d6f997ab14f139c17024f13a610d9",
    processor=Unzip(),
)

real_files = [x for x in cp_data if not Path(x).name.startswith(".")]

# %%

sql_files = [(Path(x).parent.parent.name, x) for x in cp_data if x.endswith("db")]
con.sql("INSTALL sqlite;")
con.sql("LOAD sqlite;")
for name, sql_file in sql_files:
    con.sql(f"ATTACH '{sql_file}' AS {name} (TYPE sqlite);")

con.sql("SHOW DATABASES;")
con.sql("SHOW TABLES;")
# %%
original_tables = []
for name, _ in sql_files:
    con.sql(f"USE {name}")
    orig_profiles = con.sql("SELECT * FROM MyExpt_Per_Object").pl()
    orig_consensus = (
        orig_profiles.group_by("ImageNumber")
        .median()
        .select(pl.exclude("ObjectNumber"))
    )
    # %%
    parsed = get_feature_groups(
        orig_consensus.select(~cs.by_dtype(pl.String)).columns,
        ("object", "feature", "channel", "suffix"),
    ).with_columns(pl.col("channel").str.strip_prefix("Orig"))

    orig_unpivot = orig_consensus.unpivot(index="ImageNumber", variable_name="fullname")
    with_parsed = con.sql("SELECT * FROM orig_unpivot NATURAL JOIN parsed")

    with_trimmed = con.sql(
        "SELECT *,trim_features(fullname) AS cpm_id FROM with_parsed"
    )
    imageid_mapper = con.sql(
        "(SELECT ImageNumber,split_part(Image_FileName_OrigDNA, '_', 2) AS site,split_part(Image_FileName_OrigDNA, '_', 1) AS gene  FROM MyExpt_Per_Image)"
    )
    with_chsite = con.sql(
        "SELECT * EXCLUDE(value), value as CellProfiler FROM with_trimmed NATURAL JOIN imageid_mapper"
    ).pl()
    original_tables.append(with_chsite.with_columns(batch=pl.lit(name)))
merged_cellprof_tables = pl.concat(original_tables)

# %% cp_measure
from util_names import get_cpm_to_measurement_mapper

mapper = get_cpm_to_measurement_mapper()

cp_df = pl.read_parquet(cpmeasure_parquet)
meta_cols = ("object", "gene", "site", "channel")
new_consensus = cp_df.group_by(meta_cols).median()
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
new_compartment = new_consensus.with_columns(
    pl.col("object")
    .replace({"DNA": "Nuclei", "AGP": "Cells", "cell": "Cells"})
    .str.to_titlecase()
)
new_unpivoted = new_compartment.unpivot(
    index=cs.by_dtype(pl.String), variable_name="cpm_id"
)
dups = new_unpivoted.with_columns(
    pl.when(pl.col("cpm_id").is_in(chless_feats))
    .then(pl.lit(""))
    .otherwise(pl.col("channel"))
    .alias("channel")
)
uniq = dups.unique()
# Remove channels for channel-less measurements
# %% combine
merged = (
    con.sql(
        "SELECT *, value AS cp_measure FROM uniq NATURAL JOIN merged_cellprof_tables WHERE (cp_measure+CellProfiler)!=0 ORDER BY cpm_id"
    )
    .pl()
    .with_columns(pl.col("cpm_id").alias(""))
)
# %% Plot


plt.close()

g = sns.FacetGrid(
    merged.to_pandas(),
    col="cpm_id",
    col_wrap=4,
    hue="object",
    sharex=False,
    sharey=False,
    legend_out=False,
    hue_kws={"markers": "channel"},
)
g.map(sns.scatterplot, "CellProfiler", "cp_measure", alpha=0.01)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.tight_layout()
plt.savefig(figs_dir / "grid_cp_vs_cpm.svg")

# %%
import polars_ds as pds
from polars_ds.modeling.transforms import polynomial_features
from sklearn import datasets, linear_model

# If you want the underlying computation to be done in f32, set pds.config.LIN_REG_EXPR_F64 = res = r
res = pl.concat([
    x.select(
        pds.lin_reg_report(
            *(
                ["CellProfiler"]
                # + polynomial_features(["x1", "x2", "x3"], degree=2, interaction_only=True)
            ),
            target="cp_measure",
        ).alias("result")
    )
    .unnest("result")
    .with_columns(pl.lit(feat_name).alias("Feature"), pl.lit(obj).alias("object"))
    .with_columns(pl.col("Feature").replace(mapper).alias("Measurement"))
    for (feat_name, obj), x in merged.partition_by(
        ("cpm_id", "object"), as_dict=True
    ).items()
])
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
    data=res.to_pandas(),
    y="r2",
    ax=axd["D"],
    alpha=0.8,
    hue="Measurement",
)
import matplotlib.ticker as ticker

# g.set_ylim(0.86, 1.01)
g.set_title("Linear fit")
g.set_xlabel("Individual features")
g.set_ylabel("R squared")

pad = 0.25
axd["D"].add_artist(generate_label("D", pad=pad))
for ax_id, featname in zip("ABC", feats_to_show):
    ax = axd[ax_id]
    h = sns.scatterplot(
        merged.filter(pl.col("cpm_id") == featname).to_pandas(),
        x="CellProfiler",
        y="cp_measure",
        hue="object",
        ax=ax,
        alpha=0.01,
        palette="Set2",
        legend=None,
    )
    # ax.text(0, 0, ax_id, fontsize=9)
    h.set_yticklabels(h.get_yticklabels(), rotation=30)
    featname = featname.removeprefix("RadialDistribution_")
    featname = featname.removeprefix("Intensity_")
    ax.set_title(featname.replace("_", " "))
    sns.despine()
    ax.add_artist(generate_label(ax_id, pad=pad))
    sns.move_legend(axd["D"], loc="upper left", bbox_to_anchor=(1, 1))
# plt.tight_layout)
plt.savefig(figs_dir / "jump_r2_examples.svg")
plt.close()
