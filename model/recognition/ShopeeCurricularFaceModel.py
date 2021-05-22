import torch
from torch import tensor, nn
from .CurricularFace import CurricularFace
import timm

from torch_utils.Config import DEFAULT_CFG as CFG

def createTimmBackbone(model_name, pretrained=True):
    print('Building Model Backbone for {} model'.format(model_name))
    model = timm.create_model(model_name, pretrained=pretrained)
    final_in_features = 0

    if 'efficientnet' in model_name:
        final_in_features = model.classifier.in_features
        model.classifier = nn.Identity()
        model.global_pool = nn.Identity()
    
    elif 'nfnet' in model_name:
        final_in_features = model.head.fc.in_features
        model.head.fc = nn.Identity()
        model.head.global_pool = nn.Identity()

    return model, final_in_features


class ShopeeCurricularFaceModel(nn.Module):

    def __init__(
        self,
        n_classes = CFG.CLASSES,
        model_name = CFG.MODEL_NAME,
        fc_dim = CFG.FC_DIM,
        margin = CFG.MARGIN,
        scale = CFG.SCALE,
        use_fc = True,
        pretrained = True):


        super(ShopeeCurricularFaceModel,self).__init__()
        
        self.backbone, final_in_features = createTimmBackbone(model_name, pretrained)
        self.pooling =  nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_fc_params()
            final_in_features = fc_dim

        self.final = CurricularFace(final_in_features, 
                                           n_classes, 
                                           s=scale, 
                                           m=margin)

    def _init_fc_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feat(image)
        logits = self.final(feature,label)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x