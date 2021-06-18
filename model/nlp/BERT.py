import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F

import transformers

# model_params = {
#     'n_classes':11014,
#     'model_name':"cahya/bert-base-indonesian-522M",
#     'use_fc':False,
#     'fc_dim':512,
#     'dropout':0.3,
# }

class BERTNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='bert-base-uncased',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(BERTNet, self).__init__()

        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        final_in_features = self.transformer.config.hidden_size
        
        self.use_fc = use_fc
    
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim


    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, input_ids,attention_mask):
        feature = self.extract_feat(input_ids,attention_mask)
        return F.normalize(feature)

    def extract_feat(self, input_ids,attention_mask):
        x = self.transformer(input_ids=input_ids,attention_mask=attention_mask)
        
        features = x[0]
        features = features[:,0,:]

        if self.use_fc:
            features = self.dropout(features)
            features = self.fc(features)
            features = self.bn(features)

        return features