import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset_pose import Feeder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from models2.fusion_model import fusion_model
from datetime import datetime
import logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# log_filename = time.strftime('log_%Y%m%d_%H%M%S.txt')
log_dir = 'logs_cross_s'
log_filename = os.path.join(log_dir, f'training_log_{timestamp}.txt')  # 设置日志文件的完整路径
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # 创建目录
def train():
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 100
    radar_lr = 1e-6
    visual_lr = 1e-5
    train_dataset = Feeder("train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataset = Feeder("test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
    model = fusion_model()
    model.to(device)
    # pre = torch.load('./checkpoints_NEW/epoch_0_cross_s_fusion_2024-11-19-18-16-14.pkl')
    # model.load_state_dict(pre)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam([
        {'params': model.radar_model.parameters(), 'lr': radar_lr},
        {'params': model.visual_model.parameters(), 'lr': visual_lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    logging.basicConfig(filename=log_filename, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
    # pre
    # pretrain_dict = torch.load('./checkpoints/epoch_49_fusion_2024-06-13-12-56-41.pkl')
    # model_dict = model.state_dict()
    # filtered = ["fusion_mlp", "radar_model.mlp", "visual_model.v_model.fcn1"]
    # filtered_pretrain_dict = {k: v for k, v in pretrain_dict.items() if not any(f in k for f in filtered)}
    # model_dict.update(filtered_pretrain_dict)
    # model.load_state_dict(pretrain_dict, strict=False)
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.radar_model.mlp.parameters():
    #     param.requires_grad = True
    # for param in model.visual_model.v_model.fcn1.parameters():
    #     param.requires_grad = True
    # for param in model.fusion_mlp.parameters():
    #     param.requires_grad = True

    correct_r = 0
    correct_v = 0
    correct_f = 0
    total = 0
    sum_loss = 0
    train_loss1 = []
    train_acc1 = []
    test_loss1 = []
    test_acc1 = []
    for epoch in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for data_iter_step, samples in loop:
            model.train()
            visual_feature = samples[0].to(device, non_blocking=True)
            radar_feature = samples[1].to(device, non_blocking=True)
            targets = samples[2].to(device, non_blocking=True)
            size = visual_feature.shape[0]
            confidence_data = torch.ones((size, 1, 300, 18, 2)).to(device, non_blocking=True)
            new_visual = torch.cat((visual_feature, confidence_data), dim=1)

            loss, feat_r, feat_v, feat_f = model(radar_feature, visual_feature, targets, epoch)
            # loss = criterion(feat_v, targets)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_r = F.softmax(feat_r, dim=1)
                y_v = F.softmax(feat_v, dim=1)
                y_f = F.softmax(feat_f, dim=1)
                correct_r += (torch.argmax(y_r, 1) == targets).sum().item()
                correct_v += (torch.argmax(y_v, 1) == targets).sum().item()
                correct_f += (torch.argmax(y_f, 1) == targets).sum().item()
                total += len(targets)
                sum_loss += loss.sum().item()
            acc_r = correct_r / total
            acc_v = correct_v / total
            acc_f = correct_f / total
            train_loss = sum_loss / total
            loop.set_description(f'Epoch[{epoch + 1}/{epochs}]')
            loop.set_postfix(train_loss=train_loss, train_acc_r=acc_r, train_acc_v=acc_v, train_acc_f=acc_f)
        logging.info(f"cross_s Train Epoch {epoch}: train_loss={train_loss}, acc_r={acc_r}, acc_v={acc_v}, acc_f={acc_f}")
        
        train_epoch_loss = sum_loss / total
        train_epoch_acc = correct_f / total
        scheduler.step()
        # if (epoch + 1) % 1 == 0:
        if epoch % 10 == 0 and epoch != 0:
            cur_time = time.strftime("%Y-%m-%d-%H-%M-%S")
            checkpoint_path = os.path.join("./checkpoints_NEW", f'epoch_{epoch}_cross_s_fusion_{cur_time}.pkl')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

        # ---------------------------------------------
        t_sum_loss = 0
        t_total = 0
        t_correct_r = 0
        t_correct_v = 0
        t_correct_f = 0
        model.eval()
        loop2 = tqdm(enumerate(test_loader), total=len(test_loader))
        with torch.no_grad():
            for data_iter_step, samples in loop2:
                visual_feature = samples[0].to(device, non_blocking=True)
                radar_feature = samples[1].to(device, non_blocking=True)
                targets = samples[2].to(device, non_blocking=True)
                size = visual_feature.shape[0]
                confidence_data = torch.ones((size, 1, 300, 18, 2)).to(device)
                new_visual = torch.cat((visual_feature, confidence_data), dim=1)
                random_tensor = torch.zeros_like(visual_feature)
                t_loss, t_feat_r, t_feat_v, t_feat_f = model(radar_feature, visual_feature, targets, epoch)
                loss = t_loss.sum()
                with torch.no_grad():
                    
                    t_y_r = F.softmax(t_feat_r, dim=1)
                    t_y_v = F.softmax(t_feat_v, dim=1)
                    t_y_f = F.softmax(t_feat_f, dim=1)
                    t_correct_r += (torch.argmax(t_y_r, 1) == targets).sum().item()
                    t_correct_v += (torch.argmax(t_y_v, 1) == targets).sum().item()
                    t_correct_f += (torch.argmax(t_y_f, 1) == targets).sum().item()
                    t_total += len(targets)
                    t_sum_loss += loss.sum().item()
                t_acc_r = t_correct_r / t_total
                t_acc_v = t_correct_v / t_total
                t_acc_f = t_correct_f / t_total
                test_loss = t_sum_loss / t_total
                loop2.set_description(f'Epoch[{epoch + 1}/{epochs}]')
                loop2.set_postfix(test_loss=test_loss, test_acc_r=t_acc_r, test_acc_v=t_acc_v, test_acc_f=t_acc_f)
            test_epoch_loss = t_sum_loss / t_total
            test_epoch_acc = t_correct_f / t_total
        logging.info(f"cross_s test Epoch {epoch}: test_loss={train_loss}, t_acc_r={t_acc_r}, t_acc_v={t_acc_v}, t_acc_f={t_acc_f}")

        train_loss1.append(train_epoch_loss)
        train_acc1.append(train_epoch_acc)
        test_loss1.append(test_epoch_loss)
        test_acc1.append(test_epoch_acc)

    return train_loss1, train_acc1, test_loss1, test_acc1


if __name__ == "__main__":
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    train_loss, train_acc, test_loss, test_acc = train()

    # 保存训练结果
    from log.log import create_log_dir
    from log.log import save_result


    log_dir = create_log_dir()
    save_result('train_loss.txt', train_loss, log_dir)
    save_result('train_acc.txt', train_acc, log_dir)
    save_result('test_loss.txt', test_loss, log_dir)
    save_result('test_acc.txt', test_acc, log_dir)


