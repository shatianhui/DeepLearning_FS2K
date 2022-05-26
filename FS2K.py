from torch.utils import data
import os
from PIL import Image

import config as cfg
from utils import Dejson


# 加载FS2K数据集的类
class FS2K(data.Dataset):
    def __init__(self, json_path, selected_attrs, transform, mode="train"):
        self.img_path_list, self.labels_list = Dejson(selected_attrs, json_path)  # 获取所有的图片路径和标签信息
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):  # 接收一个索引，返回图片及标签
        img_path = self.img_path_list[index]
        labels = self.labels_list[index]
        image = Image.open(os.path.join(cfg.root, img_path)).convert('RGB')
        if self.transform != None:
            image = self.transform(image)
        return image, labels


def get_loader(json_path, selected_attrs, batch_size, mode='train', transform=None):
    dataset = FS2K(json_path, selected_attrs, transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  drop_last=True)  # 如果最后一个迷你批次的数据小于batch_size,将其扔掉
    return data_loader
