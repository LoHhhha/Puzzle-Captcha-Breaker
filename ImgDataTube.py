from torch.utils.data import Dataset
from PIL import Image
import os

labels = {"C0": 0, "C1": 1}


class ImgDataTube(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_info = self.get_img_info(data_path)
        self.transform = transform

    def __getitem__(self, item):
        path_img, label, img_name = self.data_info[item]
        # print(path_img)
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_name

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_path):
        data_info = []
        img_path = os.listdir(data_path)
        for path in img_path:
            temp_val = path[0:2]
            if temp_val != "C0" or "C1":
                temp_val = "C0"
            data_info.append((data_path + '/' + path, labels[temp_val], path))
        return data_info
