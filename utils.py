"""
This file contains utility functions for creating 2D and 3D embeddings, along with concatenating all of the ISMIR papers data, and other useful functions
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import pdb

GOOGLE_SHEET_ID = "1bxYfVBG-12H41vdVkWMth9rKn9JhUWAw2GwWflfVots"
GOOGLE_SHEET_TAB_NAME = "UN%20categorization"


def create_embeddings(data_path="data/ismir_all_papers.csv", pca_variance=0.95):
    """
    Creates a 2D and 3D embeddings of paper titles and abstracts using TSNE and UMAP. For TSNE, we first use PCA with
    the specified variance to reduce the dimensionality of the embeddings.
    """
    print("generating embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    df = pd.read_csv(data_path)
    titles = df["Title"].tolist()
    abstracts = df["Abstract"].tolist()
    title_embeddings = model.encode(titles)
    abstract_embeddings = model.encode(abstracts)

    # PCA
    pca = PCA(n_components=pca_variance)
    title_pca = pca.fit_transform(title_embeddings)
    abstract_pca = pca.fit_transform(abstract_embeddings)

    # TSNE
    tsne = TSNE(n_components=2)
    title_tsne = tsne.fit_transform(title_pca)
    abstract_tsne = tsne.fit_transform(abstract_pca)
    tsne_3d = TSNE(n_components=3)
    title_tsne_3d = tsne_3d.fit_transform(title_pca)
    abstract_tsne_3d = tsne_3d.fit_transform(abstract_pca)

    # UMAP
    umap_model = umap.UMAP(n_components=2)
    title_umap = umap_model.fit_transform(title_embeddings)
    abstract_umap = umap_model.fit_transform(abstract_embeddings)
    umap_model_3d = umap.UMAP(n_components=3)
    title_umap_3d = umap_model_3d.fit_transform(title_embeddings)
    abstract_umap_3d = umap_model_3d.fit_transform(abstract_embeddings)

    # Make TSNE and UMAP 2D and 3D columns in the CSV
    df["title_tsne_2d"] = [list(x) for x in title_tsne]
    df["abstract_tsne_2d"] = [list(x) for x in abstract_tsne]
    df["title_tsne_3d"] = [list(x) for x in title_tsne_3d]
    df["abstract_tsne_3d"] = [list(x) for x in abstract_tsne_3d]
    df["title_umap_2d"] = [list(x) for x in title_umap]
    df["abstract_umap_2d"] = [list(x) for x in abstract_umap]
    df["title_umap_3d"] = [list(x) for x in title_umap_3d]
    df["abstract_umap_3d"] = [list(x) for x in abstract_umap_3d]

    df.to_csv(data_path, index=False)

    print(f"Embeddings saved to {data_path}")


def create_concatenated_data(data_dir, save_dir="data/"):
    """
    Concatenates all of the ISMIR CSV files into one CSV
    """
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            data = pd.read_csv(os.path.join(data_dir, file))
            all_data.append(data)
    concatenated_data = pd.concat(all_data, ignore_index=True)
    pdb.set_trace()
    concatenated_data.to_csv(
        os.path.join(save_dir, "ismir_all_papers.csv"), index=False
    )


def create_first_author_columns(data_path="data/ismir_all_papers.csv"):
    """
    Given a concatenated ISMIR CSV dataset, creates the following new columns and overwrites the CSV.
    These columns are only used for graphing purposes.

    first_country: The country of the first author
    first_aff_cat: The category of the first author's affiliation
    first_aff_cat_UN: The UN category of the first author's country. This will be retrieved from a separate google sheets URL

    """

    df = pd.read_csv(data_path)
    df["first_country"] = df["Authors with Affiliations"].apply(
        lambda x: x.split(";")[0].split("+")[1].split("|")[0].split(">")[1]
    )
    df["first_aff_cat"] = df["Authors with Affiliations"].apply(
        lambda x: x.split(";")[0].split("+")[1].split("|")[0].split(">")[2]
    )

    # ISO code is in a column called "ISO Code" and un economy category is in a column called "Economic Category"
    un_cat = pd.read_csv(
        f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={GOOGLE_SHEET_TAB_NAME}"
    )
    df["first_aff_cat_UN"] = df["first_country"].apply(
        lambda x: (
            un_cat[un_cat["ISO Code"] == x]["Economic Category"].values[0]
            if x in un_cat["ISO Code"].values
            else "Unknown"
        )
    )

    df.to_csv(data_path, index=False)


if __name__ == "__main__":
    pass
