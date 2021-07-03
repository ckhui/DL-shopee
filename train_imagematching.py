import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from dataset.ImageDataloader import ImageDataLoader
from torch_utils.Config import DEFAULT_CFG
from model.recognition.ShopeeCurricularFaceModel import ShopeeCurricularFaceModel
from torch_utils.Runner import train_fn
from torch_utils.Optimizer import Ranger
from torch_utils.Scheduler import ShopeeScheduler
import torch


if __name__ == "__main__":
    CFG = DEFAULT_CFG()
    ## Setup 
    dataset_folder = 'data'
    # csv_train = f"{folder}/shopee-product-matching/train.csv"
    csv_train = f"{dataset_folder}/train.csv"
    image_folder = f"{dataset_folder}/train_images/"
    
    ## Read Dataframe
    df = pd.read_csv(csv_train)
    labelencoder= LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])

    # ## Read Dataset
    # trainloader = BuildImageDataloader(df, image_folder,batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, device=CFG.DEVICE)


if __name__ == '__main__':
    ## Read Dataset
    dataloader = ImageDataLoader(IMG_SIZE=384)
    trainloader = dataloader.BuildImageDataloader(df, image_folder, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, device=CFG.DEVICE)

    CFG = DEFAULT_CFG
    CFG.BATCH_SIZE = 2
    CFG.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    CFG.NUM_WORKERS = 0
    CFG.CLASSES = df['label_group'].nunique()
    CFG.MODEL_NAME = 'swin_base_patch4_window12_384'


    ## Init Model and Training
    model = ShopeeCurricularFaceModel(
        n_classes = CFG.CLASSES,
        model_name = CFG.MODEL_NAME,
        fc_dim = CFG.FC_DIM,
        margin = CFG.MARGIN,
        scale = CFG.SCALE).to(CFG.DEVICE)

    optimizer = Ranger(model.parameters(), lr = CFG.SCHEDULER_PARAMS['lr_start'])
    # optimizer = torch.optim.Adam(model.parameters(), lr = config.SCHEDULER_PARAMS['lr_start'])
    scheduler = ShopeeScheduler(optimizer,**CFG.SCHEDULER_PARAMS)

    print("START Training ... ")
    torch.save(model.state_dict(),'./weights/{}_cuFace_model_0.pt'.format(CFG.MODEL_NAME))
    for i in range(CFG.EPOCHS):
        avg_loss_train = train_fn(model, trainloader, optimizer, scheduler, i)
        if i%10 == 0:
            torch.save(model.state_dict(),'./weights/{}_cuFace_model_{}.pt'.format(CFG.MODEL_NAME, i))
