import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from dataset.ImageDataloader import BuildInferDataloader
from torch_utils.Config import DEFAULT_CFG
from model.recognition.ShopeeCurricularFaceModel import ShopeeCurricularFaceModel

from model.matching.knn import KNN_predict
import torch

if __name__ == "__main__":
    ## Setup 
    dataset_folder = '../../'
    # csv_train = f"{folder}/shopee-product-matching/train.csv"
    csv_train = f"{dataset_folder}/shopee-product-matching/tiny.csv"
    image_folder = f"{dataset_folder}/shopee-product-matching/train_images/"

    csv_test = f"{dataset_folder}/shopee-product-matching/test.csv"
    image_folder_test = f"{dataset_folder}/shopee-product-matching/test_images/"


    CFG = DEFAULT_CFG
    CFG.BATCH_SIZE = 4
    CFG.DEVICE = 'cpu'
    CFG.NUM_WORKERS = 0


    df, dataloader = BuildInferDataloader(csv_train, image_folder, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, device=CFG.DEVICE)


    ## load model and weight
    IMG_MODEL_PATH = "./weights/init_weight_curriuclarFace.pt"
    model = ShopeeCurricularFaceModel(
        n_classes = CFG.CLASSES,
        model_name = CFG.MODEL_NAME,
        fc_dim = CFG.FC_DIM,
        margin = CFG.MARGIN,
        scale = CFG.SCALE,
        pretrained = False)
    model.eval()
    model.load_state_dict(torch.load(IMG_MODEL_PATH),strict=False)
    model = model.to(CFG.DEVICE)

    ## to image embeding
    dataset_embeds = []
    with torch.no_grad():
        for img,label in tqdm(dataloader): 
            img = img.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)
            feat = model.extract_feat(img)
            image_embeddings = feat.detach().cpu().numpy()
            dataset_embeds.append(image_embeddings)
    dataset_embeds = np.concatenate(dataset_embeds)
    print(f'Our image embeddings shape is {dataset_embeds.shape}')


    KNN = min(len(df), 50)
    thresh = 15 #21.0
    df_text, group_predictions, scores_df = KNN_predict(
        df, dataset_embeds, KNN=KNN, 
        thresh_range=list(np.arange(15,25,1))
    )
    # df_text, image_predictions, scores_df = KNN_predict(
    #     df, dataset_embeds, KNN=KNN, 
    #     thresh=thresh
    # )
    df['image_predictions'] = group_predictions

    