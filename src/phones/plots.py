from typing import List, Union, Optional

import numpy as np
import pandas as pd

from .features import Phone
from . import PhoneCollection

try:
    import plotly.express as px
    import plotly.io as pio
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'when using plots, make sure to install the optional "plot" dependency'
    )

# def set_renderer(renderer):
#     pio.renderers.default = renderer

# set_renderer("jupyterlab")


def plot_phone(phone):
    fig = px.bar(x=phone.feature_names, y=phone.vector)
    fig.show()


Phone.plot = plot_phone


def plot_collection(collection, n_components: int = 2, **kwargs):
    vectors = []
    phones = []
    langs = []

    if n_components not in [2, 3, 4]:
        raise ValueError("n_components must be either 2, 3 or 4")

    for phone in collection.values:
        vectors.append(phone.vector)
        langs.append(phone.language_code)
        phones.append(phone.index)
    pca = PCA(n_components=n_components)
    vectors = pca.fit_transform(np.array(vectors))
    if n_components == 2:
        df = pd.DataFrame(
            {
                "phone": phones,
                "language": langs,
                "PC1": vectors[:, 0],
                "PC2": vectors[:, 1],
            }
        )
    elif n_components == 3:
        df = pd.DataFrame(
            {
                "phone": phones,
                "language": langs,
                "PC1": vectors[:, 0],
                "PC2": vectors[:, 1],
                "PC3": vectors[:, 2],
            }
        )
    else:
        df = pd.DataFrame(
            {
                "phone": phones,
                "language": langs,
                "PC1": vectors[:, 0],
                "PC2": vectors[:, 1],
                "PC3": vectors[:, 2],
                "PC4": ((vectors[:, 3]-vectors[:, 3].min())/(vectors[:, 3].max()-vectors[:, 3].min()))*20+1,
            }
        )
    if collection.source.language_column is None:
        if n_components == 2:
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="language",
                text="phone",
                title="PCA of Phones",
                **kwargs,
            )
        elif n_components == 3:
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="language",
                text="phone",
                title="PCA of Phones",
                **kwargs,
            )
        else:
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="language",
                text="phone",
                title="PCA of Phones",
                **kwargs,
            )
    else:
        new_langs = []
        for i, row in df.iterrows():
            if n_components == 2:
                shared = df[(df["PC1"] == row["PC1"]) & (df["PC2"] == row["PC2"])]
            elif n_components == 3:
                shared = df[
                    (df["PC1"] == row["PC1"])
                    & (df["PC2"] == row["PC2"])
                    & (df["PC3"] == row["PC3"])
                ]
            else:
                shared = df[
                    (df["PC1"] == row["PC1"])
                    & (df["PC2"] == row["PC2"])
                    & (df["PC3"] == row["PC3"])
                    & (df["PC4"] == row["PC4"])
                ]
            
            if len(shared) > 1:
                unique_langs = pd.unique(shared["language"])
                new_langs.append(",".join(sorted(unique_langs)))
            else:
                new_langs.append(row["language"])
        df["language"] = new_langs
        df = df.drop_duplicates()
        if n_components == 2:
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="language",
                text="phone",
                title="PCA of Phones",
                **kwargs,
            )
        elif n_components == 3:
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="language",
                text="phone",
                title="PCA of Phones",
                **kwargs,
            )
        else:
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                size="PC4",
                color="language",
                text="phone",
                title="PCA of Phones",
                **kwargs,
            )
    fig.update_traces(textposition="bottom right")
    fig.show()


PhoneCollection.plot = plot_collection
