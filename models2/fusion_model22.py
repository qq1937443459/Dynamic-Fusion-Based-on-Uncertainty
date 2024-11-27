
import torch
import torch.nn as nn

from models2.visual_model import visual_model

from models.radar_model import r_model
from models.ETMC2 import ETMC
class fusion_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.radar_model = r_model()
        self.visual_model = visual_model()
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
        loss = 0.99 * loss + 0.1 * loss_
        return loss, feat_r, feat_v, feat_f



if __name__=="__main__":
    net = fusion_model()