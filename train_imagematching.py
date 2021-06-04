import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from dataset.ImageDataloader import BuildImageDataloader
from torch_utils.Config import DEFAULT_CFG
from torch_utils.Runner import train_fn
from model.recognition.ShopeeCurricularFaceModel import ShopeeCurricularFaceModel
from torch_utils.Optimizer import Ranger
from torch_utils.Scheduler import ShopeeScheduler
from torch_utils.Runner import train_fn
import torch

## Setup 
csv_train = "../../shopee-product-matching/train.csv"
image_folder = "../../shopee-product-matching/train_images/"

## Read Dataframe
df = pd.read_csv(csv_train)
labelencoder= LabelEncoder()
df['label_group'] = labelencoder.fit_transform(df['label_group'])

CFG = DEFAULT_CFG
CFG.BATCH_SIZE = 8
CFG.DEVICE = 'cpu'
CFG.NUM_WORKERS = 0
CFG.CLASSES = df['label_group'].nunique()

## Read Dataset
trainloader = BuildImageDataloader(df, image_folder,batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, device=CFG.DEVICE)

## Init Model and Training
model = ShopeeCurricularFaceModel(
    n_classes = CFG.CLASSES,
    model_name = CFG.MODEL_NAME,
    fc_dim = CFG.FC_DIM,
    margin = CFG.MARGIN,
    scale = CFG.SCALE)

optimizer = Ranger(model.parameters(), lr = CFG.SCHEDULER_PARAMS['lr_start'])
# optimizer = torch.optim.Adam(model.parameters(), lr = config.SCHEDULER_PARAMS['lr_start'])
scheduler = ShopeeScheduler(optimizer,**CFG.SCHEDULER_PARAMS)

print("START Training ... ")
torch.save(model.state_dict(),'./weights/nfnet_cuFace_model_0.pt')
for i in range(CFG.EPOCHS):
    avg_loss_train = train_fn(model, trainloader, optimizer, scheduler, i)
    if i%10 == 0:
        torch.save(model.state_dict(),f'./weights/nfnet_cuFace_model_{i}.pt')