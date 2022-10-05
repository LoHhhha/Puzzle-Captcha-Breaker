import os
import cv2
import numpy as np
import torchvision.transforms
import torch
from torch.utils.data import DataLoader
from bisection_ImgDataSet import ImgDataSet

input_path = "./Demo/"

net = torch.load('./Net_save/Net_save.pth', map_location=torch.device('cpu'))
net.eval()
images_name = os.listdir(input_path)

for image_name in images_name:
    image = cv2.imread(input_path + image_name)
    image_piece = []
    for i in range(0, 2):
        for j in range(0, 4):
            image_piece.append(image[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80, :])
    for i in range(0, 8):
        for j in range(i + 1, 8):
            temp = image_piece[i]
            image_piece[i] = image_piece[j]
            image_piece[j] = temp
            new_image = np.zeros((2 * 80, 4 * 80, 3))
            index = 0
            for r in range(0, 2):
                for c in range(0, 4):
                    new_image[80 * r:80 * (r + 1), 80 * c:80 * (c + 1), :] = image_piece[index]
                    index += 1
            cv2.imwrite("./Demo_temp/" + str(i) + '_' + str(j) + '_' + image_name, new_image)
            temp = image_piece[i]
            image_piece[i] = image_piece[j]
            image_piece[j] = temp

dataset = ImgDataSet(data_path="./Demo_temp", transform=torchvision.transforms.ToTensor())
DataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

ans = []
max_output = -1
pos_name = "X"
with torch.no_grad():
    for data in DataLoader:
        images, targets, img_name = data
        # puts into Net
        outputs = net(images)
        if outputs[0][1] > max_output:
            max_output = outputs[0][1]
            pos_name = img_name[0][0:3]
        # get info
        results = outputs.argmax()
        if results == 1:
            ans.append(img_name[0][0:3])

Remove_images_name = os.listdir("./Demo_temp/")
for image in Remove_images_name:
    os.remove("./Demo_temp/" + image)

if not ans:
    ans.append(pos_name)

print("预测结果:", ans[0])