import torch
from log.log import save_result
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import time
from dataset_pose import Feeder
from models2.resnet import ResNet

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loss1 = []
    train_acc1 = []
    test_loss1 = []
    test_acc1 = []

    lr = 1e-6  # -6
    batch_size = 16
    epochs = 100
    train_data = Feeder('train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data = Feeder('test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # model = r_model()
    # model = models.resnet18(pretrained=True)
    model = ResNet([2, 2, 2, 2])
    # path = '/home/ywh/project/RGB_radar/checkpoints/model_epoch_30_rarar_ppt_2024-06-15-1020-12.pkl'
    # path = '/home/ywh/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    path = '/home/ywh/project/RGB_radar/checkpoints/model_epoch_95_rarar_ppt_2024-06-15-1307-37.pkl'
    model.load_state_dict(torch.load(path, map_location='cpu'),strict=False)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs,10)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    tata_loader = train_loader

    for epoch in range(epochs):
        total = 0.
        sum_loss = 0.
        correct = 0
        optimizer.zero_grad()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, samples in loop:
            model.train(True)
            samples[1] = samples[1].to(device, non_blocking=True)
            radar_feature = samples[1]
            targets = samples[2].to(device, non_blocking=True)

            out, _ = model(radar_feature)
            loss = criterion(out, targets)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_ = F.softmax(out, dim=1)
                correct += (torch.argmax(y_, 1) == targets).sum().item()
                total += len(targets)
                sum_loss += loss.sum().item()
            acc = correct / total
            train_loss = sum_loss / total
            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            loop.set_postfix(train_loss=train_loss, acc=acc)
        train_epoch_loss = sum_loss / total
        train_epoch_acc = correct / total
        scheduler.step()

        if epoch % 19 == 0 and epoch != 0:
            current_time = time.strftime("%Y-%m-%d-%-H%M-%S")
            checkpoint_path = os.path.join("./checkpoints", f'model_epoch_{epoch}_rarar_ppt_{current_time}.pkl')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

        #  测试
        t_sum_loss = 0.
        t_total = 0
        test_acc = []
        t_correct_r = 0
        t_correct_v = 0
        t_correct_f = 0
        model.eval()
        with torch.no_grad():
            loop2 = tqdm(enumerate(test_loader), total=len(test_loader))
            for data_iter_step, samples in loop2:
                samples[1] = samples[1].to(device, non_blocking=True)
                radar_feature = samples[1]
                targets = samples[2].to(device, non_blocking=True)

                t_feat_r, _ = model(radar_feature)

                t_loss = criterion(t_feat_r, targets)
                with torch.no_grad():
                    y_r = F.softmax(t_feat_r, dim=1)
                    t_correct_r += (torch.argmax(y_r, 1) == targets).sum().item()
                    t_total += len(targets)
                    t_sum_loss += t_loss.sum().item()
                acc_r = t_correct_r / t_total
                test_loss = t_sum_loss / t_total
                loop2.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                loop2.set_postfix(test_loss=test_loss, test_acc_r=acc_r)
        test_epoch_loss = t_sum_loss / t_total
        test_epoch_acc = t_correct_r / t_total

        train_loss1.append(train_epoch_loss)
        train_acc1.append(train_epoch_acc)
        test_loss1.append(test_epoch_loss)
        test_acc1.append(test_epoch_acc)

# 保存训练结果
    from log.log import create_log_dir
    log_dir = create_log_dir()
    save_result('train_r_loss.txt', train_loss1, log_dir)
    save_result('train_r_acc.txt', train_acc1, log_dir)
    save_result('test_r_loss.txt', test_loss1, log_dir)
    save_result('test_r_acc.txt', test_acc1, log_dir)