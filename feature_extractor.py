import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm

from model.recognition.ShopeeCurricularFaceModel import ShopeeCurricularFaceModel
from model.nlp.tfidf import TFIDF_feateure

def extract_image_feature(config, weight_path, dataloader):
    ## load model and weight
    model = ShopeeCurricularFaceModel(
        n_classes = config.CLASSES,
        model_name = config.MODEL_NAME,
        fc_dim = config.FC_DIM,
        margin = config.MARGIN,
        scale = config.SCALE,
        pretrained = False)
    model.eval()
    model = model.to(config.DEVICE)
    model.load_state_dict(
        torch.load(weight_path, map_location=torch.device(config.DEVICE)),
        strict=False
    )

    ## to image embeding
    dataset_embeds = []
    with torch.no_grad():
        for img,label in tqdm(dataloader): 
            img = img.to(config.DEVICE)
            label = label.to(config.DEVICE)
            feat = model.extract_feat(img)
            image_embeddings = feat.detach().cpu().numpy()
            dataset_embeds.append(image_embeddings)
    features_embeds = np.concatenate(dataset_embeds)
    print(f'Our image embeddings shape is {features_embeds.shape}')

    del model, dataset_embeds
    gc.collect()

    return features_embeds

def extract_tfidf_feature(df, min_PCA = 5000):
    return TFIDF_feateure(df, min_PCA)