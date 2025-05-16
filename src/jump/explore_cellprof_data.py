"""
Compare Cellprofiler and cp_measure data.
Plot the comparison.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import polars as pl
import pooch
from duckdb.typing import VARCHAR
from parse_features import get_feature_groups
from polars import selectors as cs
from pooch import Unzip
from skimage.io import imread

figs_dir = Path("../../") / "figs"


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
# sample_img = imread(real_files[0])
# plt.imshow(sample_img)
# plt.savefig("delme.png", dpi=300)

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
cp_df = pl.read_parquet("/datastore/alan/cp_measure/profiles/first_set.parquet")
# new_unpivoted.with_columns(pl.col("channel").when(pl.col("cpm_id").is_in(cpm_chless)
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
    pl.col("object").replace({"DNA": "Nuclei", "AGP": "Cells"})
)
new_unpivoted = new_compartment.unpivot(
    index=cs.by_dtype(pl.String), variable_name="cpm_id"
)
dups = new_unpivoted.with_columns(
    pl.when(pl.col("cpm_id").is_in(chless_feats))
    .then(pl.lit("NA"))
    .otherwise(pl.col("channel"))
    .alias("channel")
)
uniq = dups.unique()
# Remove channels for channel-less measurements
# %% combine
merged = con.sql(
    "SELECT *, value AS cp_measure FROM uniq NATURAL JOIN merged_cellprof_tables"
).pl()
# %% Plot

import seaborn as sns

g = sns.FacetGrid(
    merged.to_pandas(),
    col="cpm_id",
    col_wrap=4,
    hue="object",
    sharex=False,
    sharey=False,
    legend_out=False,
    hue_kws={"markers": "channel"},
    # margin_titles=True,
)
g.map(sns.scatterplot, "CellProfiler", "cp_measure", alpha=0.5)
plt.tight_layout()
plt.savefig(figs_dir / "grid_cp_vs_cpm.png", dpi=300)
