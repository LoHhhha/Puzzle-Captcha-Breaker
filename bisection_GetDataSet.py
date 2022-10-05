import random

import cv2
import os
import numpy as np

file_path = ["./material_320_160/" + path for path in os.listdir("./material_320_160")]
image_id = 0
for path in file_path:
    image = cv2.imread(path)
    if image_id < 400:
        cv2.imwrite("./Train_dataset/" + "C1" + '_' + str(image_id) + '.jpg', image)
    else:
        cv2.imwrite("./Test_dataset/" + "C1" + '_' + str(image_id) + '.jpg', image)
    image_piece = []
    # 切片
    for i in range(0, 2):
        for j in range(0, 4):
            image_piece.append(image[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80, :])
    # 粘合
    i = random.randint(0, 6)
    j = random.randint(i+1, 7)
    # 交换
    temp = image_piece[i]
    image_piece[i] = image_piece[j]
    image_piece[j] = temp
    # 生成
    new_image = np.zeros((2 * 80, 4 * 80, 3))
    index = 0
    for r in range(0, 2):
        for c in range(0, 4):
            new_image[80 * r:80 * (r + 1), 80 * c:80 * (c + 1), :] = image_piece[index]
            index += 1
    # 输出
    if image_id < 400:
        cv2.imwrite("./Train_dataset/" + 'C0' + "_" + str(i) + '_' + str(j) + '_' + str(image_id) + ".jpg",
                new_image)
    else:
        cv2.imwrite("./Test_dataset/" + 'C0' + "_" + str(i) + '_' + str(j) + '_' + str(image_id) + ".jpg",
                    new_image)
    image_id += 1
