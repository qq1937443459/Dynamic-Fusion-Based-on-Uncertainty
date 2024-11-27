import torch
import torch.nn as nn

class radar_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_model = models_vit.__dict__['vit_base_patch16'](
            num_classes=10,
            drop_path_rate=0.1,
            global_pool=True,
            )
        # self.load_model()
    def load_model(self):
        path = "/home/ywh/project/RGB_radar/models2/mae_pretrain_vit_base.pth"
        self.r_model.load_state_dict(torch.load(path),strict=False)
    def forward(self,radar):
        feat_r = self.r_model(radar)
        return feat_r



if __name__=="__main__":
    # path = "../models/mae_pretrain_vit_base.pth"
    # checkpoints = torch.load("mae_pretrain_vit_base.pth", map_location='cpu')
    # print(checkpoints['model'])
    # input = torch.randn(1,3,224,224)
    # model = models_vit.__dict__['vit_base_patch16'](
    #     num_classes=10,
    #     drop_path_rate=0.5,
    #     global_pool=False,
    # )
    net = radar_model()
    # output = model(input)