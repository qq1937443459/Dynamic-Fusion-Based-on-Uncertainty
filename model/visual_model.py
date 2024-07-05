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

matplotlib.use('TKAgg')

from models2.st_gcn import Model
class visual_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_model = Model(in_channels=3, num_class=10, edge_importance_weighting=True,
                        graph_args={'layout': 'openpose', 'strategy': 'spatial'}, )
        # self.load_model()
    def load_model(self):
        path = "/home/ywh/project/RGB_radar/models2/st_gcn.kinetics.pt"
        self.v_model.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
    def forward(self,visual):
        feat_v, feature_v = self.v_model(visual)
        return feat_v, feature_v




if __name__=="__main__":
    net = visual_model()