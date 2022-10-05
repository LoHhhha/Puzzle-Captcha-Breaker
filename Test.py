import torchvision.transforms
import torch
from torch.utils.data import DataLoader
from bisection_ImgDataSet import ImgDataSet


net = torch.load('./Net_save/Net_save.pth', map_location=torch.device('cuda:0'))

labels = ["NO", "YES"]

test_dataset = ImgDataSet("./Test_img", transform=torchvision.transforms.ToTensor())
test_DataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# loading GPU/CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")
input("Press Enter to Continue")

net.eval()
with torch.no_grad():
    for data in test_DataLoader:
        images, targets, img_name = data
        # confirm device
        images = images.to(device)
        targets = targets.to(device)
        # puts into Net
        outputs = net(images)
        # get info
        results = outputs.argmax()

        # print(outputs, targets.item(), outputs.argmax().item(), '->', (results.item() == targets.item()))
        # print(img_name, "->", labels[outputs.argmax().item()])

        # if targets.item() == 1:
        #     print(outputs, targets.item(), outputs.argmax().item(), '->', (results.item() == targets.item()))
        #     print(img_name, "->", labels[outputs.argmax().item()])

        if results.item() != targets.item():
            print(outputs, targets.item(), outputs.argmax().item(), '->', (results.item() == targets.item()))
            print(img_name, "->", labels[outputs.argmax().item()])
