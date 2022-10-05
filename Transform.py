from PIL import Image
import os

filenames = os.listdir("./material")
for i in range(0, len(filenames)):
    img = Image.open("./material/" + filenames[i])
    new_img = img.resize((320, 160))
    new_img.save("./material_320_160/" + filenames[i])
