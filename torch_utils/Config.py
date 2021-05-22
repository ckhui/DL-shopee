class DEFAULT_CFG:
    
    ## Training
    EPOCHS = 15

    ## Dataloader
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    DEVICE = 'cuda'

    ## For CurricularFace
    SCALE = 30
    MARGIN = 0.5
    
    ## Timm Pretrained
    MODEL_NAME = 'eca_nfnet_l0'

    ## Classifier Head
    CLASSES = 11014 # len(df.label_group.unique())
    FC_DIM = 512
    
    ## Learning Rate
    SCHEDULER_PARAMS = {
            "lr_start": 1e-5,
            "lr_max": 1e-5 * 32,
            "lr_min": 1e-6,
            "lr_ramp_ep": 5,
            "lr_sus_ep": 0,
            "lr_decay": 0.8,
        }