import numpy as np
import pandas as pd

from .features import Phone
from . import PhoneCollection

try:
    import plotly.express as px
    import plotly.io as pio
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    raise ModuleNotFoundError("when using plots, make sure to install the optional \"plot\" dependency")

# def set_renderer(renderer):
#     pio.renderers.default = renderer

# set_renderer("jupyterlab")

def plot_phone(phone):
    fig = px.bar(x=phone.feature_names, y=phone.vector)
    fig.show()

Phone.plot = plot_phone

def plot_collection(collection):
    vectors = []
    phones = []
    langs = []
    for phone in collection.values:
        vectors.append(phone.vector)
        langs.append(phone.language_code)
        phones.append(phone.index)
    pca = PCA(n_components=2)
    vectors = pca.fit_transform(np.array(vectors))
    df = pd.DataFrame({
        'phone': phones,
        'language': langs,
        'PC1': vectors[:,0],
        'PC2': vectors[:,1],
    })
    if collection.source.language_column is None:
        fig = px.scatter(df, x='PC1', y='PC2', text='phone')
    else:
        new_langs = []
        for i, row in df.iterrows():
            shared = df[(df['PC1']==row['PC1'])&(df['PC2']==row['PC2'])]
            if len(shared) > 1:
                unique_langs = pd.unique(shared['language'])
                new_langs.append(','.join(sorted(unique_langs)))
            else:
                new_langs.append(row['language'])
        df['language'] = new_langs
        df = df.drop_duplicates()
        fig = px.scatter(df, x='PC1', y='PC2', text='phone', color='language')
    fig.update_traces(textposition="bottom right")
    fig.show()

PhoneCollection.plot = plot_collection