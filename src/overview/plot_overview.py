from pathlib import Path

import matplotlib.colors as mpl  # noqa: CPY001
import numpy as np
import polars as pl
from jump_portrait.fetch import get_item_location_info
from jump_portrait.s3 import get_image_from_s3uri
from matplotlib import pyplot as plt

figs_path = Path("") / ".." / ".." / "figs"
df_raw = get_item_location_info("JCP2022_011844", input_column="JCP2022")
orig = df_raw.select(pl.col("^URL_Orig[DARME].*$"))
channel_rgb = {
    "AGP": "#FF7F00",  # Orange
    "DNA": "#0000FF",  # Blue
    "ER": "#00FF00",  # Green
    "Mito": "#FF0000",  # Red
    "RNA": "#FFFF00",  # Yellow
}
for colname, url in orig.sample(1).unpivot().rows():
    plt.close()
    plt.axis("off")
    channel = colname.split("_")[1].removeprefix("Orig")
    print(channel)
    img = get_image_from_s3uri(url)
    cmap = mpl.LinearSegmentedColormap.from_list(
        channel, ("#000", channel_rgb[channel])
    )
    plt.imshow(img, vmin=0, vmax=np.percentile(img, 99.5), cmap=cmap)
    # plt.imshow(img)
    plt.savefig(figs_path / f"{channel}.png", dpi=300)
    plt.close()
