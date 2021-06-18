import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm

import transformers

from model.recognition.ShopeeCurricularFaceModel import ShopeeCurricularFaceModel
from model.nlp.tfidf import TFIDF_feateure
from model.nlp.BERT import BERTNet
from dataset.ImageDataloader import BuildInferDataloader_TEXT

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



def extract_text_feature(config, df, pretrained_path, weight_path):


    TOKENIZER = transformers.AutoTokenizer.from_pretrained(pretrained_path)
    model_params = {
        'n_classes':config.CLASSES,
        'model_name':pretrained_path,
        'use_fc':False,
        'fc_dim':config.FC_DIM,
        'dropout':0.3,
    }
    model = BERTNet(**model_params)
    model.eval()
    model.load_state_dict(
        dict(list(torch.load(
            weight_path, map_location=torch.device(config.DEVICE)
            ).items())[:-1]),
        strict=False
        )

    model = model.to(config.DEVICE)
    text_dataloader = BuildInferDataloader_TEXT(df, TOKENIZER, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, device=config.DEVICE)

    embeds = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(text_dataloader): 
            input_ids = input_ids.to(config.DEVICE)
            attention_mask = attention_mask.to(config.DEVICE)
            feat = model(input_ids, attention_mask)
            text_embeddings = feat.detach().cpu().numpy()
            embeds.append(text_embeddings)

    del model
    text_embeddings = np.concatenate(embeds)
    print(f'Our text embeddings shape is {text_embeddings.shape}')
    del embeds
    gc.collect()
    return text_embeddings