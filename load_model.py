import torch
import DataLoader
import Model
import configuration as cfg
import matplotlib.pyplot as plt
import QuantizationScheme as Q
import train_test


# define the way to draw input and output
# def plot_vis(figure, v_list):
#     ax = figure.subplots(2, 4)
#     ax[0, 0].hist(v_list["initial"], bins=200)
#     ax[0, 0].set_title("initial")
#     ax[0, 1].hist(v_list["after_conv1"], bins=200)
#     ax[0, 1].set_title("after_conv1")
#     ax[0, 2].hist(v_list["after_bn1"], bins=200)
#     ax[0, 2].set_title("after_bn1")
#     ax[0, 3].hist(v_list["after_relu"], bins=200)
#     ax[0, 3].set_title("after_relu")
#     ax[1, 0].hist(v_list["after_conv2"], bins=200)
#     ax[1, 0].set_title("after_conv2")
#     ax[1, 1].hist(v_list["after_bn2"], bins=200)
#     ax[1, 1].set_title("after_bn2")
#     ax[1, 2].hist(v_list["after_shortcut"], bins=200)
#     ax[1, 2].set_title("after_shortcut")
#     ax[1, 3].hist(v_list['out'], bins=200)
#     ax[1, 3].set_title("out")

def plot_list(figure, v_list):
    for fig in list(v_list.values()):
        plt.hist(fig, bins=200)
        plt.show()
        plt.clc()
    return None


_, test_loader = DataLoader.load_data()
model = Model.ResNet20().to(cfg.config['device'])
model.load_state_dict(torch.load('./result/saved_model/2bit/resnet.pth'))
model.eval()
train_test.test_net(model, test_loader)

for data in test_loader:
    img, label = data
    for figure in img:
        fig = figure
        break
img = img.cuda()
model.layer3[2].require_vis = True
out_1 = model.conv1(img)
out_1 = model.layer1(out_1)
out_1 = model.layer2(out_1)
out_1 = model.layer3[0](out_1)
out_2 = model.layer3[1](out_1)
out_2, out_list = model.layer3[2](out_1)
fig = plt.figure(figsize=(20, 9))
plot_vis(fig, out_list)
# plt.savefig('./result/process/2bit.jpg')
plt.show()