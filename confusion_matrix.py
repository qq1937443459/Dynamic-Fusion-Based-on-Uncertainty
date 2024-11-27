import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
# from dataset2 import CRPTorchDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from dataset_pose import Feeder
from models2.fusion_model import fusion_model
def draw_matrix(label_ture,average_unc, label_pre, label_name, title="confusion Maatrix", pdf_save_path=None, dpi=100):
    cm = confusion_matrix(y_true=label_ture, y_pred=label_pre,normalize='true')
    # new_column = np.random.rand(10)

    cm = np.column_stack((cm,average_unc))
    # cm = np.insert(cm,10,0,axis=1)
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.xticks(range(label_name.__len__()),label_name,rotation=45)
    plt.yticks(range(label_name.__len__()),label_name)
    plt.colorbar()

    # extra_column = np.array([0,1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # ax2.imshow(extra_column.reshape((-1,1)),cmap=plt.cm.Blues,alpha=0)
    # ax2.set_ylabel("extra metric",rotation=45,labelpad=0)
    # ax2.grid(None)

    plt.tight_layout()

    for i in range(11):
        for j in range(10):
            color = (1, 1, 1) if i == j else (0, 0, 0)
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)
    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)



def load():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cpu')
    batch_size = 8
    y = []
    y_pre = []
    uncertainty_ = []
    train_data = Feeder('train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data = Feeder('test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # model = Module()
    model = fusion_model()
    model.to(device)

    model.load_state_dict(torch.load('./checkpoints_NEW/epoch_0_cross_s_fusion_2024-11-19-18-16-14.pkl', map_location=device), strict=False)   # FFM FRM tmc
    lu = []
    errors = []
    incorrect_indices_label_1 = []
    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    for _, samples in loop:
        model.eval()
        # if not samples:
        #     continue
        visual_feature = samples[0].to(device, non_blocking=True)
        # visual_feature = samples[0]
        samples[1] = samples[1].to(device, non_blocking=True)
        radar_feature = samples[1]
        targets = samples[2].to(device, non_blocking=True)
        # index = samples[3].to(device)
        size = visual_feature.shape[0]
        confidence_data = torch.ones((size, 1, 300, 18, 2)).to(device, non_blocking=True)
        shape = (size, 3, 408, 524)
        new_visual = torch.cat((visual_feature, confidence_data), dim=1)

        aaa = torch.zeros_like(new_visual)
        # new_visual = torch.randn(size, 3, 300, 18, 2).to(device)
        # radar_feature = torch.full(shape, 255, dtype=torch.float32).to(device)

        loss, feat_r, feat_v, feat_f = model(radar_feature, visual_feature, targets, 1)

        # output = model(img_variable)
        # alpha = feat_f

        evidence_r = F.relu(feat_v)
        alpha_r = evidence_r + 1

        # uncertainty = 10 * 1.8 / torch.sum(1.8 ** feat_f, dim=1, keepdim=True)
        uncertainty = 10 / torch.sum(feat_f, dim=1, keepdim=True)
        # uncertainty = 10 / torch.sum(feat_f, dim=1, keepdim=True)
        uncertainty = uncertainty.cpu().detach().numpy()
        uncertainty_ = np.append(uncertainty_, uncertainty)

        y_F = F.softmax(feat_f, dim=1)
        # y_v = F.softmax(feat_v, dim=1)

        # y_F = (y_r + y_v) / 2


        # y_r = np.argmax(y_r.cpu().detach().numpy(), -1)
        # y_v = np.argmax(y_v.cpu().detach().numpy(), -1)
        y_F = np.argmax(y_F.cpu().detach().numpy(), -1)

        targets = targets.cpu().detach().numpy()
        # targets = targets.view(1, -1)
        y.append(targets)

        y_pre.append(y_F)


    # for i, error in enumerate(errors):
    #     if error['label'] == 1:
    #         incorrect_indices_label_1.append([error['index']])
    #
    # with open('index_yep.txt', 'w') as f:
    #     f.write(str(incorrect_indices_label_1))

    y = np.concatenate(y)
    y_pre = np.concatenate(y_pre)
    # uncertainty_= np.concatenate(uncertainty_)
    average_unc = np.zeros((10, 1))
    for i in range(10):
        class_uncertainties = uncertainty_[y == i]
        average_unc[i] = class_uncertainties.mean()
    print(average_unc)

    draw_matrix(label_ture=y, average_unc=average_unc,
                label_pre=y_pre,
                label_name=["ch", "cl", "k", "l", "p", "pa", "r", "up", "v", "x"],
                title="confusion Matrix", pdf_save_path="matrix/ablation/feat_f.jpg", dpi=300)

if __name__=="__main__":
    load()

    # y = []
    # y_pre = []
    # # label = torch.randint(0, 9, (8,))
    # # label = torch.randint(0, 9, (8,))
    # label = torch.arange(10)
    # label1 = torch.arange(10)
    # y.append(label)
    # y_pre.append(label1)
    # y = np.concatenate(y)
    # y_pre = np.concatenate(y_pre)
    # draw_matrix(label_ture=y,
    #             label_pre=y_pre,
    #             label_name=["ch", "cl", "k", "l","p", "pa", "r", "up", "v","x"],
    #             title="confusion Maatrix", pdf_save_path="matrix_unc.jpg", dpi=300)

