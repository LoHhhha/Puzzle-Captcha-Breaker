import os
import shutil
from bisection_ImgDataSet import ImgDataSet
from bisection_Net import Net
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# macro definition
learning_rate = 0.000001
epoch = 3

# loading dataset
train_dataset = ImgDataSet("./Train_dataset", transform=torchvision.transforms.ToTensor())
train_DataLoader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = ImgDataSet("./Test_dataset", transform=torchvision.transforms.ToTensor())
test_DataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True)

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

# confirm and load GPU/CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")
input("Press Enter to Continue")

# load pre_Net
if os.path.isfile("./Net_save/Net_save.pth"):
    net = torch.load("./Net_save/Net_save.pth")
    print("Load and ues a pre_Net")
else:
    net = Net()
    print("Use a new_Net")
if os.path.isfile("./Net_save/train_num"):
    train_num = ''
    with open("./Net_save/train_num", encoding='UTF-8') as fp:
        train_num += fp.read()
    train_num = int(train_num)
else:
    train_num = 0

# setting loss_func
loss = nn.CrossEntropyLoss()

# confirm device
net = net.to(device)
loss = loss.to(device)

# setting optim
optim = torch.optim.SGD(params=net.parameters(), lr=learning_rate)

# setting SummaryWriter
Writer = SummaryWriter("PC_logs")

cur_total_train = 0
cur_time = train_num / epoch

# train and test part
for cur_time in range(train_num, train_num + epoch):
    print("<*>cur_train_times:{}".format(cur_time + 1))
    # train
    true_num = 0
    net.train()
    for data in train_DataLoader:
        images, targets, img_name = data

        # confirm device
        images = images.to(device)
        targets = targets.to(device)

        # puts into Net
        outputs = net(images)
        # get loss
        loss_results = loss(outputs, targets)

        # recover optim
        optim.zero_grad()

        # backward
        loss_results.backward()
        optim.step()

        true_num += (outputs.argmax().item() == targets.item())
        cur_total_train += 1

        if cur_total_train % 20 == 0:
            print("\tcur_total_train:{},loss:{}".format(cur_total_train, loss_results.item()))
            Writer.add_scalar('Train_loss_{}'.format(cur_time), loss_results.item(), cur_total_train)
    # display Info
    accuracy = true_num / train_dataset_size
    print("<*>After this round, {}% accuracy in Train_dataset.".format(accuracy * 100))

    # test
    net.eval()
    running_loss = 0
    true_num = 0
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
            print("From", img_name, ":")
            print('\t', outputs, "target:", targets.item(), "get:", results.item(), '->',
                  (results.item() == targets.item()))

            true_num += (results.item() == targets.item())
            loss_results = loss(outputs, targets)
            running_loss += loss_results.item()

    accuracy = true_num / test_dataset_size
    # display Info
    print("<*>After this round, {}% accuracy in Test_dataset.".format(accuracy * 100))
    print("<*>After this round, total_loss:{} in Test_dataset.".format(running_loss))
    Writer.add_scalar("Test_loss", running_loss, cur_time)
    Writer.add_scalar("Test_accuracy", accuracy, cur_time)
    torch.save(net, "./Net_save/Net_save_{}.pth".format(cur_time))
    print("Net save successfully.")

# save and close
Writer.close()

train_num += epoch
shutil.copyfile("./Net_save/Net_save_{}.pth".format(train_num - 1), "./Net_save/Net_save.pth")

fp = open("./Net_save/train_num", mode='w')
fp.write(str(train_num))
fp.close()
