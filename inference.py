import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from dataset.ImageDataloader import BuildInferDataloader
from torch_utils.Config import DEFAULT_CFG
from model.recognition.ShopeeCurricularFaceModel import ShopeeCurricularFaceModel

from model.matching.knn import KNN_predict
from feature_extractor import extract_image_feature, extract_tfidf_feature, extract_text_feature
import torch

if __name__ == "__main__":
    ## Setup 
    dataset_folder = '../../'
    # csv_train = f"{dataset_folder}/shopee-product-matching/train.csv"
    csv_train = f"{dataset_folder}/shopee-product-matching/tiny.csv"
    image_folder_train = f"{dataset_folder}/shopee-product-matching/train_images/"

    csv_test = f"{dataset_folder}/shopee-product-matching/test.csv"
    image_folder_test = f"{dataset_folder}/shopee-product-matching/test_images/"

    csv = csv_train
    image_folder = image_folder_train

    CFG = DEFAULT_CFG
    CFG.BATCH_SIZE = 4
    CFG.NUM_WORKERS = 0


    df, dataloader = BuildInferDataloader(csv, image_folder, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, device=CFG.DEVICE)
    df, dataloader_384 = BuildInferDataloader(csv, image_folder, img_size=384, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, device=CFG.DEVICE)

    nfnet_weight = "./weights/init_weight_curriuclarFace.pt"
    nfnet_config = CFG()
    nfnet_config.MODEL_NAME = "eca_nfnet_l0"
    image_embeds = extract_image_feature(nfnet_config, nfnet_weight, dataloader)

    # swim_weight = "./weights/swin_base_patch4_window12_384_cuFace_model_14.pt"
    # swim_config = CFG()
    # swim_config.MODEL_NAME = "swin_base_patch4_window12_384"
    # image_embeds = extract_image_feature(swim_config, swim_weight, dataloader_384)
    
    # tfidf_embeds = extract_tfidf_feature(df)

    # model_bert = "cahya/bert-base-indonesian-522M"
    # weight_bert = './weights/bert-indonesian-522m_best_loss_num_epochs_30_arcface.bin'
    # text_embeds = extract_text_feature(CFG, df, model_bert, weight_bert)

    # model_sbert = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    # weight_sbert = './weights/paraphrase-xlm-r-multilingual-v130.bin'
    # text_embeds = extract_text_feature(CFG, df, model_sbert, weight_sbert)


    data_emb = image_embeds
    # data_emb = tfidf_embeds
    # data_emb = text_embeds

    KNN = min(len(df), 50)
    thresh = 15 #21.0
    df_text, group_predictions, scores_df = KNN_predict(
        df, data_emb, KNN=KNN, 
        thresh_range=list(np.arange(15,25,1))
    )
    # df_text, image_predictions, scores_df = KNN_predict(
    #     df, data_emb, KNN=KNN, 
    #     thresh=thresh
    # )
    df['image_predictions'] = group_predictions

    