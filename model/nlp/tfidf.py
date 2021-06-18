import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm

IS_GPU = torch.cuda.is_available()
if IS_GPU:
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml import PCA
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA


def TFIDF_feateure(df, min_PCA = 5000):
    if IS_GPU:
        df_cu = cudf.DataFrame(df)
    else:
        df_cu = df
    max_features = 15000
    n_components = min(min_PCA, len(df_cu))
    nlp_model = TfidfVectorizer(stop_words = 'english', binary = True, max_features = max_features)
    text_embeddings = nlp_model.fit_transform(df_cu['title']).toarray()
    pca = PCA(n_components = n_components)
    if IS_GPU:
        text_embeddings = pca.fit_transform(text_embeddings).get()
    else:
        text_embeddings = pca.fit_transform(text_embeddings)
    print(f'Our title text embedding shape is {text_embeddings.shape}')
    return text_embeddings