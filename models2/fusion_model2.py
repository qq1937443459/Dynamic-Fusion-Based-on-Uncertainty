import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from pos_embed import get_2d_sincos_pos_embed
from param import args
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from models2 import models_vit
from models2.visual_model import visual_model
# from models2.radar_model import radar_model
matplotlib.use('TKAgg')
from models.radar_model import r_model
from models.ETMC import ETMC
class fusion_model(nn.Module):

    def __init__(self):
        super().__init__()
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.radar_model = r_model()
        # self.radar_model.to(device)
        self.visual_model = visual_model()
        # self.visual_model.to(device)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024, 10)
        )
        self.etmc = ETMC()

    def forward(self, radar, visual, labels, epoch):
        criterion = torch.nn.CrossEntropyLoss()

        feat_r, feature_r = self.radar_model(radar)
        feat_v, feature_v = self.visual_model(visual)
        feature_f = torch.cat((feature_r, feature_v), dim=1)
        feat_f = self.fusion_mlp(feature_f)

        loss_r = criterion(feat_r, labels)
        loss_v = criterion(feat_v, labels)
        loss_f = criterion(feat_f, labels)
        loss = 1 / 3 * loss_r + 1 / 3 * loss_v + 1 / 3 * loss_f
        loss_, feat_r, feat_v, feat_f = self.etmc(feat_r, feat_v, feat_f, labels, epoch)
        loss = 0.5 * loss + 0.5 * loss_
        return loss, feat_r, feat_v, feat_f



if __name__=="__main__":
    net = fusion_model()