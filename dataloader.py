import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy
import random

class catadata(torch.utils.data.Dataset):
    def __init__(self, filepath, train, transform=None, target_transform=None, download=False):
        datalist = []
        self.taglist = []
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        abnormal_path = filepath + '/abnormal/'
        normal_path = filepath + '/normal/'
        width, height = [], [] 
        for filename in os.listdir(abnormal_path):
            pic = Image.open(abnormal_path + filename)
            pic = self.transform(pic)
            datalist.append((pic, 1, filename))
        # print('min width:{} min height:{}'.format(min(width), min(height)))

        for filename in os.listdir(normal_path):
            pic = Image.open(normal_path + filename)
            pic = self.transform(pic)
            datalist.append((pic, 0, filename))
        datasize = len(datalist)
        random.shuffle(datalist)
        if self.train:
            train_list = datalist[:int(datasize*0.8)]
            imagelist = [item[0] for item in train_list]
            taglist= [item[1] for item in train_list]
            namelist = [item[2] for item in train_list]
            self.train_data = imagelist
            self.train_labels = taglist
            self.namelist = namelist
            print('train data size:{}'.format(len(self.train_data)))
        else:
            # resume
            test_list = datalist[int(datasize*0.8):]
            imagelist = [item[0] for item in test_list]
            taglist= [item[1] for item in test_list]
            namelist = [item[2] for item in test_list]
            self.test_data = imagelist
            self.test_labels = taglist
            self.namelist = namelist
            print('test data size:{}'.format(len(self.test_data)))

    def __getitem__(self, index):
        if self.train:
            img, target, name = self.train_data[index], self.train_labels[index], self.namelist[index]
        else:
            img, target, name = self.test_data[index], self.test_labels[index], self.namelist[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, name

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def getDataloaders(data, config_of_data, splits=['train', 'val', 'test'],
                   aug=True, use_validset=True, data_root='data/cata', batch_size=1, normalized=True,
                   num_workers=1, **kwargs):
    train_loader, val_loader, test_loader = None, None, None
    if data.find('cata')>=0 :
        print('loading ' + data)
        print(config_of_data)
        d_func = catadata
        #normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        if config_of_data['augmentation']:
            print('with data augmentation')
            aug_trans = [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            aug_trans = []
        common_trans = [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        if normalized:
            print('dataset is normalized')
            common_trans.append(normalize)
        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)
        if True:
            if 'train' in splits:
                train_set = d_func(data_root, train=True, transform=train_compose, download=True)
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size, shuffle=True)
            if 'val' in splits or 'test' in splits:
                test_set = d_func(data_root, train=False, transform=test_compose)
                test_loader = torch.utils.data.DataLoader(
                    test_set, batch_size=1, shuffle=True)
                val_loader = test_loader
    # print('train_set:{}'.format(train_set[0]))
    return train_loader, val_loader, test_loader
