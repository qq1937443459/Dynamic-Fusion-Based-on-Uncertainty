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
matplotlib.use('TKAgg')
from models2.resnet import ResNet
from models.ETMC import ETMC
from models2.FFM import FFM

class fusion_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.radar_model = ResNet([2, 2, 2, 2])
        self.visual_model = visual_model()

        self.fusion_mlp = nn.Sequential(
            nn.Linear(512, 10)
        )
        self.etmc = ETMC()
        self.pretrain()
        self.FFMS = FFM(v_in_dim=75*18, v_out_dim=256, dim=512)

    def pretrain(self):
        # path_r = '/home/ywh/project/RGB_radar/checkpoints/model_epoch_95_rarar_ppt_2024-06-15-1307-37.pkl'
        path_r = '/home/ywh/project/RGB_radar/models2/resnet18-f37072fd.pth'
        self.radar_model.load_state_dict(torch.load(path_r, map_location='cpu'), strict=False)
        # path_v = '/home/ywh/project/RGB_radar/checkpoints/epoch_9_visual_2024-06-13-20-36-17.pkl'
        path_v = "/home/ywh/project/RGB_radar/models2/st_gcn.kinetics.pt"
        self.visual_model.load_state_dict(torch.load(path_v, map_location='cpu'), strict=False)

    def forward(self, radar, visual, labels, epoch):
        criterion = torch.nn.CrossEntropyLoss()

        feat_r, feature_r = self.radar_model(radar)
        feat_v, feature_v = self.visual_model(visual)
        # feature_f = torch.cat((feature_r, feature_v), dim=1)
        feature_f = self.FFMS(feature_r, feature_v)
        feat_f = self.fusion_mlp(feature_f)

        # feat_F = (feat_r + feat_v) / 2
        loss_r = criterion(feat_r, labels)
        loss_v = criterion(feat_v, labels)
        loss_f = criterion(feat_f, labels)

        # loss = 1/3 * loss_r + 1/3 * loss_v + 1/3 * loss_f
        loss_, feat_r, feat_v, feat_f = self.etmc(feat_r, feat_v, feat_f, labels, epoch)
        # loss = 0.5 * loss + 0.5 * loss_
        return loss_, feat_r, feat_v, feat_f



if __name__=="__main__":
    net = fusion_model()