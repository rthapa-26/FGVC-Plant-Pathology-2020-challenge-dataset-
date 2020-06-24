import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from skimage import io
import random
import os
import collections


name2label_dict = collections.OrderedDict([
    ('Healthy', 0),
    ('CedarAppleRust', 1),
    ('Scab', 2),
    ('Combinations', 3)
])


label2name_dict = collections.OrderedDict([(name2label_dict[x], x) for x in name2label_dict])


class AppleDiseaseData(Dataset):
    def __init__(self, data_list_file, center_crop=False):
        super(AppleDiseaseData,  self).__init__()

        self.data_list_file = data_list_file
        self.center_crop = center_crop

        self.img_files = []
        self.labels = []
        with open(self.data_list_file) as fp:
            for line in fp.readlines():
                line = line.strip()
                if line:
                    disease, fpath = line.split('\t')
                    self.labels.append(name2label_dict[disease])
                    self.img_files.append(fpath)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225)),
                                    ])

        self.resized_width = 640
        self.resized_height = 480

        self.input_width = 512
        self.input_height = 384

    def __getitem__(self, index):
        img = io.imread(self.img_files[index])[:, :, :3]
        label = self.labels[index]

        # first resize, then apply random crop
        if self.resized_width != img.shape[1]:
            img = cv2.resize(img, (self.resized_width, self.resized_height), interpolation=cv2.INTER_AREA)

        assert (img.shape[0] == self.resized_height and img.shape[1] == self.resized_width)
        if self.center_crop:
            start_y = int((self.resized_height - self.input_height) / 2.0)
            start_x = int((self.resized_width - self.input_width) / 2.0)
        else:   # random crop
            start_y = random.randint(0, self.resized_height - self.input_height)
            start_x = random.randint(0, self.resized_width - self.input_width)

        img = img[start_y:start_y + self.input_height, start_x:start_x + self.input_width, :]

        img_ori = torch.from_numpy(img)     # pixel intensity between [0, 255]
        img = self.transform(img)

        label = torch.Tensor([label,]).long()

        return {'im': img, 'im_ori': img_ori, 'label': label, 'fname': os.path.basename(self.img_files[index])}

    def __len__(self):
        return len(self.img_files)


def build_dataloader(batch_size, mode, num_workers=3):
    base_dir = '/phoenix/S7/kz298/AppleDiseaseClassification/data_split'
    if mode == 'train':
        data_list_file = os.path.join(base_dir, 'train.txt')
        shuffle = True
        center_crop = False
    elif mode == 'val':
        data_list_file = os.path.join(base_dir, 'val.txt')
        shuffle = False
        center_crop = True
    elif mode == 'train_val':
        print('building train_val dataloader...')
        data_list_file = os.path.join(base_dir, 'train_val.txt')
        shuffle = True
        center_crop = False
    elif mode == 'test':
        print('building test dataloader...')
        data_list_file = os.path.join(base_dir, 'test.txt')
        shuffle = False
        center_crop = True

    data = AppleDiseaseData(data_list_file, center_crop=center_crop)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


if __name__ == '__main__':
    data_list_file = '/phoenix/S7/kz298/AppleDiseaseClassification/data_split/train.txt'
    train_dataset = AppleDiseaseData(data_list_file)

    # print('\ntesting dataset...')
    # item = train_dataset[0]
    # for x in item:
    #     print('{}: {} {} range: ({}, {})'.format(x, type(item[x]), item[x].shape, item[x].min(), item[x].max()))

    print('\ntesting dataloader...')
    loader = build_dataloader(batch_size=5, mode='train')
    for batch_idx, item in enumerate(loader):
        for x in item:
            print('{}: {} {} range: ({}, {})'.format(x, type(item[x]), item[x].shape, item[x].min(), item[x].max()))
        break
