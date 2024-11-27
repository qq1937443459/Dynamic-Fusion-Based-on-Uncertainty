import torch
import torch.nn as nn
from models2.visual_model import visual_model
from models2.resnet import ResNet
from models.ETMC import ETMC, TMC
from models2.FFM import FFM, FRM
import torch.nn.functional as F

class MMTM(nn.Module):
  def __init__(self, dim_visual, dim_skeleton, ratio):
    super(MMTM, self).__init__()
    dim = dim_visual + dim_skeleton
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Linear(dim, dim_out)

    self.fc_visual = nn.Linear(dim_out, dim_visual)
    self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    # initialize
    # with torch.no_grad():
    #   self.fc_squeeze.apply(init_weights)
    #   self.fc_visual.apply(init_weights)
    #   self.fc_skeleton.apply(init_weights)

  def forward(self, visual, skeleton):
    squeeze_array = []
    for tensor in [visual, skeleton]:
      tview = tensor.view(tensor.shape[:2] + (-1,))
      squeeze_array.append(torch.mean(tview, dim=-1))
    squeeze = torch.cat(squeeze_array, 1)

    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)

    vis_out = self.fc_visual(excitation)
    sk_out = self.fc_skeleton(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(visual.shape) - len(vis_out.shape)
    vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

    dim_diff = len(skeleton.shape) - len(sk_out.shape)
    sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

    return visual * vis_out, skeleton * sk_out


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
        
        self.mmtm_64 = MMTM(64,64,4)
        self.mmtm_128 = MMTM(128,128,4)
        self.mmtm_256 = MMTM(256,256,4)
        
    def pretrain(self):
        path_r = '/home/ywh/project/RGB_radar/checkpoints/model_epoch_95_rarar_ppt_2024-06-15-1307-37.pkl'
        # path_r = '/home/ywh/project/RGB_radar/checkpoints/resnet18-f37072fd.pth'
        self.radar_model.load_state_dict(torch.load(path_r, map_location='cpu'), strict=False)
        path_v = '/home/ywh/project/RGB_radar/checkpoints/epoch_9_visual_2024-06-13-20-36-17.pkl'
        # path_v = "/home/ywh/project/RGB_radar/models2/st_gcn.kinetics.pt"
        self.visual_model.load_state_dict(torch.load(path_v, map_location='cpu'), strict=False)

    
    def forward(self, radar, visual, labels, epoch):
        criterion = torch.nn.CrossEntropyLoss()

        # feat_r, feature_r = self.radar_model(radar)
        # feat_v, feature_v = self.visual_model(visual)
        # mmtm 只用原始特征
        # stage 1
        B,_ ,_,_= radar.shape
        feat_r, feat_r_64 = self.radar_model.forward_feature_64(radar)  # 前面的是原始特征 torch.Size([16, 64, 128, 128])
        feat_v, feat_v_64 = self.visual_model.v_model.forward_feature_64(visual)   # torch.Size([32, 64, 300, 18])
        
        feat_r = feat_r.unsqueeze(2)
        feat_v = feat_v.view(B,64,2,300,18)
        feat_r, feat_v = self.mmtm_64(feat_r,feat_v)
        feat_r = feat_r.view(B,64,128,128)      
        feat_v = feat_v.view(B*2,64,300,18)
        
        
        feat_r_64, feat_v_64 = self.FRMs[0](feat_r_64, feat_v_64)
        # stage 2
        feat_r, feat_r_128 = self.radar_model.forward_feature_128(feat_r, feat_r_64)   # torch.Size([16, 128, 64, 64])
        feat_v, feat_v_128 = self.visual_model.v_model.forward_feature_128(feat_v, feat_v_64)  # torch.Size([32, 128, 150, 18])
        
        feat_r = feat_r.unsqueeze(2)
        feat_v = feat_v.view(B,128,2,150,18)
        feat_r, feat_v = self.mmtm_128(feat_r,feat_v)
        feat_r = feat_r.view(B,128,64,64)      
        feat_v = feat_v.view(B*2,128,150,18)
        
        feat_r_128, feat_v_128 = self.FRMs[1](feat_r_128, feat_v_128)

        # stage 3
        feat_r, feat_r_256 = self.radar_model.forward_feature_256(feat_r, feat_r_128)  # torch.Size([16, 256, 32, 32])
        feat_v, feat_v_256 = self.visual_model.v_model.forward_feature_256(feat_v, feat_v_128)  # torch.Size([32, 256, 75, 18])
        
        feat_r = feat_r.unsqueeze(2)
        feat_v = feat_v.view(B,256,2,75,18)
        feat_r, feat_v = self.mmtm_256(feat_r,feat_v)
        feat_r = feat_r.view(B,256,32,32)      
        feat_v = feat_v.view(B*2,256,75,18)
        
        feat_r_256, feat_v_256 = self.FRMs[2](feat_r_256, feat_v_256)

        # stage 4  feat_r和feat_v是原始特征映射到10分类的输出  feat_r_和feat_v_是校正特征映射到10分类的输出
        feat_r, feature_r, feat_r_,origin_r = self.radar_model.forward_feature_512(feat_r, feat_r_256)  # torch.Size([16, 10])
        feat_v, feature_v, feat_v_,origin_x = self.visual_model.v_model.forward_feature_512(feat_v, feat_v_256) # torch.Size([16, 10])
        
        feature_r, feature_v = self.FRMs[3](feature_r, feature_v)  
        loss_r_, feat_r = self.tmc(feat_r, feat_r_, labels, epoch)
        loss_v_, feat_v = self.tmc(feat_v, feat_v_, labels, epoch)
        feat_f = (feat_r + feat_v) / 2
        # feat_f = self.fusion_mlp(feature_f)   #(16,10)  =   mlp(16,512)

        loss_r = criterion(feat_r, labels)
        loss_v = criterion(feat_v, labels)
        loss_f = criterion(feat_f, labels)

        loss =  loss_r +  loss_v +  loss_f + loss_r_ + loss_v_
        loss /= 5.0
        # loss_, feat_r, feat_v, feat_f = self.etmc(feat_r, feat_v, feat_f, labels, epoch)
        # loss = 0.3 * loss + 0.3 * loss_ + 0.3 * loss_rv #11.12
        
        # loss = 0.5 * loss + 0.5 * loss_rv

        return loss, feat_r, feat_v, feat_f



if __name__=="__main__":
    net = fusion_model()