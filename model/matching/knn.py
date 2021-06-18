import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm

IS_GPU = torch.cuda.is_available()
if IS_GPU:
    from cuml.neighbors import NearestNeighbors
else:
    from sklearn.neighbors import NearestNeighbors

def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    correct = np.array([len(set(x[0]).intersection(set(x[1]))) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    
    precision = correct/ len_y_pred
    recall = correct/ len_y_true
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1, precision, recall


def KNN_predict(df, embeddings, KNN=50, thresh=None, thresh_range=None):
    '''
    thresh_range: np.arrange for threshold selection
    thresh: distance threshold for result matching 

    image: 2.7, tfidf: 0.6
    image: list(np.arange(2,10,0.5))
    text : list(np.arange(0.1, 1, 0.1))   
    '''
    assert ((thresh is None) or (thresh_range is None)), "Must provide either `thresh` or `thresh_range`"
    assert ((thresh_range is not None) or ('matches' in df.columns)), "Cannot perform threshold selection on testing data"

    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
    # Iterate through different thresholds to maximize cv, run this in interactive mode, then replace else clause with a solid threshold
    thresholds_scores = None
    if thresh is None:
        thresholds = thresh_range

        scores = []
        recalls = []
        precisions = []
        for threshold in thresholds:
            predictions = []
            for k in range(embeddings.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                ids = indices[k,idx]
                posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
                predictions.append(posting_ids)
            df['pred_matches'] = predictions
            f1, precision, recall = f1_score(df['matches'], df['pred_matches'])
            print(f'Threshold {threshold:.2f}: F1 {f1.mean():.4f} Precision {precision.mean():.4f} Recall {recall.mean():.4f}')
            scores.append(f1.mean())
            recalls.append(recall.mean())
            precisions.append(precision.mean())
        thresholds_scores = pd.DataFrame({
            'thresholds': thresholds, 
            'scores': scores, 
            'recalls': recalls, 
            'precisions': precisions
            })
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score is {best_score} and has a threshold {best_threshold}')
    
        thresh = best_threshold
    # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < thresh)[0]
        ids = indices[k,idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)
        
    del model, distances, indices
    gc.collect()
    return df, predictions, thresholds_scores