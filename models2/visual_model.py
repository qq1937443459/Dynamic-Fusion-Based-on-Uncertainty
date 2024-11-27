import torch
import torch.nn as nn

from models2.st_gcn import Model
class visual_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_model = Model(in_channels=3, num_class=10, edge_importance_weighting=True,
                        graph_args={'layout': 'openpose', 'strategy': 'spatial'}, )
        self.load_model()

    def load_model(self):
        path = "/home/ywh/project/RGB_radar/models2/st_gcn.kinetics.pt"
        # path = "./checkpoints/epoch_9_visual_2024-06-13-20-36-17.pkl"
        self.v_model.load_state_dict(torch.load(path), strict=False)

    def forward(self,visual):
        feat_v, feature_v = self.v_model(visual)
        return feat_v, feature_v




if __name__=="__main__":
    net = visual_model()