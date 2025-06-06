import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData
import os
import numpy as np


def plot_qc_feature(cell_by_gene, cell_meta, path):
    """
    Plot quality control features

    Parameters:
    cell_by_gene (pd.DataFrame): A DataFrame containing the cell by gene matrix
    cell_meta (pd.DataFrame): A DataFrame containing the cell metadata
    path (str): The path to save the plot
    control_probes (bool): Whether or not to plot control probes

    Returns:
    None
    """

    qc_info = cell_meta

    colors = ["#98FB98"]
    metrics = [
        "total_transcripts",
    ]

    plt.figure(
        figsize=(15, 3), dpi=200
    )
    sns.violinplot(y=qc_info[metrics[0]], color=colors[0])
    sns.stripplot(y=qc_info[metrics[0]], jitter=True, color="black", size=0.1)
    plt.title(metrics[0])  # Set title for the plot
    try:
        os.mkdir(os.path.join(path, "figures", "quality"))
    except:
        print("quality directory already made")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "figures", "quality", "qc.png"))
    plt.show()


def qc_before_clustering(
    adata,
    min_transcript_threshold=20,
    max_transcript_threshold=1500,
    guides = None
):
    """
    Perform quality control filtering

    Parameters:
    adata (AnnData): An AnnData object containing the original data

    Returns:
    adata (AnnData): An AnnData object containing the filtered data
    """
    print(f"{len(adata.obs.index)} cells before QC filtering")

    guide_ids = np.where(adata.var.index.isin(guides))[0]
    guide_counts = adata.X.A[:, guide_ids].sum(axis = 1).flatten()

    adata = adata[
        ((adata.obs["total_transcripts"] > min_transcript_threshold)
        & (adata.obs["total_transcripts"] < max_transcript_threshold)) | (guide_counts > 0),
        :,
    ]

    print(f"{len(adata.obs.index)} cells after QC filtering")

    return adata
