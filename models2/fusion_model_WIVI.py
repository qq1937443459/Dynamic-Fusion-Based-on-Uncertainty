import torch
import torch.nn as nn
from models2.visual_model import visual_model
from models2.resnet import ResNet
from models.ETMC import ETMC, TMC
from models2.FFM import FFM, FRM
import torch.nn.functional as F

class FusionDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FusionDNN, self).__init__()
        # 定义 DNN 的全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # 输出的形状为 (B, C)
        return x

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
        self.WIVI_DNN = FusionDNN(20, [128, 256, 128], 10)
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
  
        feature_r, feature_v = self.FRMs[3](feature_r, feature_v)  
        # origin_r_softmax= F.softmax(feat_r, dim=1)
        # origin_x_softmax = F.softmax(feat_v, dim=1)
        out_DNN = self.WIVI_DNN(torch.cat((feat_r, feat_v), dim=1))     
       
        # # feat_F = (feat_r + feat_v) / 2
        # loss_r = criterion(feat_r, labels)
        # loss_v = criterion(feat_v, labels)
        loss_f = criterion(out_DNN, labels)


        return loss_f, 0, 0, out_DNN



if __name__=="__main__":
    net = fusion_model()