import numpy as np
import os
from PIL import Image
import torch

class Feeder(torch.utils.data.Dataset):
    def __init__(self, splits):
        self.name = splits
        self.splits = splits.split(',')
        print('dataset.split', self.splits)

        if 'train' in self.splits:
            self.loaddata('train')
        if 'valid' in self.splits:
            self.loaddata('valid')
        if 'test' in self.name:
            self.loaddata('test')

    def loaddata(self, splits):
        path = "/home/ywh/project/RGB_radar/data_dark8/" + splits
        self.video_data_path ="/home/ywh/project/RGB_radar/data_dark8/" + splits + "/" + "video" + "/" + splits + ".npy"
        # self.video_data_path = np.load(path1)
        self.video_data = np.load(self.video_data_path, mmap_mode='r')
        v_r_chose = os.listdir(path)
        v_r_chose.sort()
        self.radar_image = {}
        self.label = {}
        count = 0
        num = 0
        for data_type in v_r_chose:
            if data_type == 'radar':
                feature_1_path = os.path.join(path, data_type)
                name_list = os.listdir(feature_1_path)
                name_list.sort()
                for name in name_list:
                    name_path = os.path.join(feature_1_path, name)
                    calss_list = os.listdir(name_path)
                    calss_list.sort()
                    for label_name, calss in enumerate(calss_list):
                        calss_path = os.path.join(name_path, calss)
                        image = os.listdir(calss_path)
                        image.sort()
                        for i, image_name in enumerate(image):
                            self.label[count] = label_name
                            image_path = os.path.join(calss_path, image_name)
                            self.radar_image[count] = image_path
                            count += 1
                            if count % 500 == 0:
                                print('processing radar %d/%d data ' % (
                                    count, (len(image) * len(calss_list) * len(name_list))))

        return self.video_data, self.radar_image, self.label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_list = []
        # clip = self.video_data[index]

        data_numpy = np.array(self.video_data[index])

        # frame_data = data_numpy[:,12,:,0]
        # x = frame_data[0] * 720
        # y = frame_data[1] * 480
        #
        # fig, ax = plt.subplots(figsize=(720/100,480/100),dpi=100)
        # ax.scatter(x, y, color='blue')
        # ax.set_xlim(0, 720)
        # ax.set_ylim(420, 0)
        # plt.show()
        data_numpy = torch.tensor(data_numpy.tolist(), dtype=torch.float)


        data_list.append(data_numpy)
        radar_image_path = self.radar_image[index]

        # img = cv2.imread(radar_image_path)
        img = Image.open(radar_image_path)
        # img = cv2.imread(radar_image_path)
        # img = np.array(img)
        # plt.imshow(img)
        # plt.show()

        # img = img.resize((524, 408))
        # img = img.resize((224, 224))  # h * w
        img = np.array(img)


        list_every = np.array(img.transpose(2, 0, 1), dtype='float32')
        data_list.append(list_every)
        label = self.label[index]
        label = np.array(label, dtype='float32')
        label = torch.LongTensor(label)

        data_list.append(label)
        return data_list





if __name__=="__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    train_dataset = Feeder("train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_dataset = Feeder("test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for data_iter_step, samples in loop:
        # model.train()
        visual_feature = samples[0]
        radar_feature = samples[1]
        targets = samples[2]