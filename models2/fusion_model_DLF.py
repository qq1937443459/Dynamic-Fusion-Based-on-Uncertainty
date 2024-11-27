import torch
import torch.nn as nn
from models2.visual_model import visual_model
from models2.resnet import ResNet
from models.ETMC import ETMC, TMC
from models2.FFM import FFM, FRM
import torch.nn.functional as F


class fusion_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.radar_model = ResNet([2, 2, 2, 2])
        self.visual_model = visual_model()

        self.fusion_mlp = nn.Sequential(
            nn.Linear(512, 10)
        )
        self.fusion_mlp_1024 = nn.Sequential(
            nn.Linear(1024, 10)
        )
        self.etmc = ETMC()
        self.tmc = TMC()
        self.pretrain()
        self.FFMs = FFM(v_in_dim=75*18, v_out_dim=256, dim=512)
        FRM_layers = [0, 1, 2, 3]
        self.FRMs = nn.ModuleList([
            FRM(dim) if i in FRM_layers else None
            for i, dim in enumerate([64, 128, 256, 512])
        ])
        
        
        self.el_fc = nn.Linear(1024, 10)
    def pretrain(self):
        path_r = '/home/ywh/project/RGB_radar/checkpoints/model_epoch_95_rarar_ppt_2024-06-15-1307-37.pkl'
        # path_r = '/home/ywh/project/RGB_radar/models2/resnet18-f37072fd.pth'
        self.radar_model.load_state_dict(torch.load(path_r, map_location='cpu'), strict=False)
        path_v = '/home/ywh/project/RGB_radar/checkpoints/epoch_9_visual_2024-06-13-20-36-17.pkl'
        # path_v = "/home/ywh/project/RGB_radar/models2/st_gcn.kinetics.pt"
        self.visual_model.load_state_dict(torch.load(path_v, map_location='cpu'), strict=False)

    def forward(self, radar, visual, labels, epoch):
        criterion = torch.nn.CrossEntropyLoss()

        # feat_r, feature_r = self.radar_model(radar)
        # feat_v, feature_v = self.visual_model(visual)

        # stage 1
        feat_r, feat_r_64 = self.radar_model.forward_feature_64(radar)  # 前面的是原始特征
        feat_v, feat_v_64 = self.visual_model.v_model.forward_feature_64(visual)
        feat_r_64, feat_v_64 = self.FRMs[0](feat_r_64, feat_v_64)
        # stage 2
        feat_r, feat_r_128 = self.radar_model.forward_feature_128(feat_r, feat_r_64)
        feat_v, feat_v_128 = self.visual_model.v_model.forward_feature_128(feat_v, feat_v_64)
        feat_r_128, feat_v_128 = self.FRMs[1](feat_r_128, feat_v_128)

        # stage 3
        feat_r, feat_r_256 = self.radar_model.forward_feature_256(feat_r, feat_r_128)
        feat_v, feat_v_256 = self.visual_model.v_model.forward_feature_256(feat_v, feat_v_128)
        feat_r_256, feat_v_256 = self.FRMs[2](feat_r_256, feat_v_256)

        # stage 4  feat_r和feat_v是原始特征映射到10分类的输出  feat_r_和feat_v_是校正特征映射到10分类的输出
        feat_r, feature_r, feat_r_,origin_r = self.radar_model.forward_feature_512(feat_r, feat_r_256)
        feat_v, feature_v, feat_v_,origin_x = self.visual_model.v_model.forward_feature_512(feat_v, feat_v_256)
        # 自己的  关掉特征优化   
        # feature_r, feature_v = self.FRMs[3](feature_r, feature_v)       
        # feature_r (16,512,16,16)  feature_v(16.512,75,18)   
        # loss_r_, feat_r = self.tmc(feat_r, feat_r_, labels, epoch)
        # loss_v_, feat_v = self.tmc(feat_v, feat_v_, labels, epoch)
        # loss_rv = loss_r_ + loss_v_
        batch_size = origin_r.size(0)
        pool_r = nn.AdaptiveAvgPool2d((1, 1))(origin_r)  # 输出形状 (16, 512, 1, 1)
        pool_r = pool_r.view(batch_size, 512)  # 变形为 (16, 512)

        # 对视觉特征进行全局平均池化
        pool_v = nn.AdaptiveAvgPool2d((1, 1))(origin_x)  # 输出形状 (16, 512, 1, 1)
        pool_v = pool_v.view(batch_size, 512)  # 变形为 (16, 512)
        combined_features = torch.cat((pool_r, pool_v), dim=1)
        
        FLV_output = self.el_fc(combined_features)
        # FSFM
        feature_f = self.FFMs(feature_r, feature_v)   
        feat_f = self.fusion_mlp(feature_f)   #(16,10)  =   mlp(16,512)

        # feat_F = (feat_r + feat_v) / 2
        loss_r = criterion(feat_r_, labels)
        loss_v = criterion(feat_v_, labels)
        loss_f = criterion(FLV_output, labels)

        # loss = 1/3 * loss_r + 1/3 * loss_v + 1/3 * loss_f
        # loss_, feat_r, feat_v, feat_f = self.etmc(feat_r, feat_v, feat_f, labels, epoch)
        # loss = 0.3 * loss + 0.3 * loss_ + 0.3 * loss_rv #11.12
        
        # loss = 0.5 * loss + 0.5 * loss_rv
        loss = (loss_r + loss_v + loss_f) / 3
        return loss, feat_r, feat_v, FLV_output



if __name__=="__main__":
    net = fusion_model()