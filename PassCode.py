import os
import cv2
import numpy as np
import torchvision.transforms
import torch
from torch.utils.data import DataLoader
from ImgDataTube import ImgDataTube


class PassCode:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_password(self):
        # init
        net = torch.load('./Net_save/Net_save.pth', map_location=torch.device('cpu'))
        net.eval()

        if not os.path.exists("./TEMP"):
            os.mkdir("./TEMP")

        image = cv2.imread(self.image_path)

        image_piece = []
        for i in range(0, 2):
            for j in range(0, 4):
                image_piece.append(image[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80, :])
        for i in range(0, 8):
            for j in range(i + 1, 8):
                # swap
                temp = image_piece[i]
                image_piece[i] = image_piece[j]
                image_piece[j] = temp

                # rebuild
                new_image = np.zeros((2 * 80, 4 * 80, 3))

                index = 0
                for r in range(0, 2):
                    for c in range(0, 4):
                        new_image[80 * r:80 * (r + 1), 80 * c:80 * (c + 1), :] = image_piece[index]
                        index += 1
                cv2.imwrite("./TEMP/" + str(i) + '_' + str(j) + '_' + 'temp.jpg', new_image)

                # recover
                temp = image_piece[i]
                image_piece[i] = image_piece[j]
                image_piece[j] = temp

        dataset = ImgDataTube(data_path="./TEMP", transform=torchvision.transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        max_output = -1
        pos_name = "X"
        with torch.no_grad():
            for data in dataloader:
                images, targets, img_name = data
                # puts into Net
                outputs = net(images)
                if outputs[0][1] > max_output:
                    max_output = outputs[0][1]
                    pos_name = img_name[0][0:3]

        remove_images_name = os.listdir("./TEMP/")
        for image_name in remove_images_name:
            os.remove("./TEMP/" + image_name)
        os.rmdir(path='./TEMP')

        return pos_name[0], pos_name[2]


if __name__ == '__main__':
    print("Try to use 'PassCode(image_path=XXX)' to get a Object.")
    print("Then, try to use 'get_password()'.")
    print("return the most possible answer A and B.")
    if os.path.exists("./temp_jigsaw.jpg"):
        get = PassCode("./temp_jigsaw.jpg")
        print(get.get_password())
