#训练文件 通过ImageFolder载入数据 通过DataLoader生成迭代器 使用预训练模型 进行训练


import torchvision
from torch import nn
import os
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import PIL.Image
import pandas as pd
import sys

root_dir = '/home/ding/DATA/dataset/'
batch_size = 16

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#训练数据transform 随机翻转+重设大小为256+中心裁剪224+转tensor+正态分布
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),#300
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(norm_mean, norm_std)
])

#训练数据transform 重设大小为256+中心裁剪224+转tensor+正态分布
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),#300
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(norm_mean, norm_std)
])

#数据集载入
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(root_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(root_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]


#生成迭代器
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)


#使用resnet预训练模型
def get_model():
    resnet18 = models.wide_resnet50_2(pretrained=True)
    #resnet18 = models.resnet18(pretrained=True)
    # 修改全连接层的输出
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)
    return resnet18



#评价准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train():
    model = get_model()
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.09)
    print(model)
    num_epoch = 10000
    for num in range(num_epoch):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for feature,label in train_iter:
            feature = feature.cuda()
            label = label.cuda()
            label_hat = model.forward(feature)
            loss_ = loss(label_hat,label)
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            train_l_sum += loss_.cpu().item()
            train_acc_sum += (label_hat.argmax(dim=1) == label).sum().cpu().item()
            n += label.shape[0]
            batch_count += 1
        valid_acc = evaluate_accuracy(valid_iter, model)
        torch.save({
            'epoch': num+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, './rennet18_epoch_'+str(num+1)+'_.tar')# save more
        print('epoch %d, loss %.4f, train acc %.3f, valid acc %.3f, time %.1f sec'
              % (num + 1, train_l_sum / batch_count, train_acc_sum / n, valid_acc, time.time() - start))





def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def pred_csv():
    model = get_model()
    model = model.to(device)
    checkpoint = torch.load('/home/ding/PycharmProjects/catsvsdogs/rennet18_epoch_12_.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    df = pd.read_csv("/home/ding/DATA/dataset/sample_submission.csv")
    for i in range(12500):  # 1000表示有1000张图片
        m = i + 1
        tpath = os.path.join(r'/home/ding/DATA/dataset/test/' + str(m) + '.jpg')  # 路径(/home/ouc/river/test)+图片名（img_m）
        fopen = PIL.Image.open(tpath)
        data = transform_test(fopen)
        data = data.view(1, 3, 224, 224)
        data = data.cuda()
        output = model.forward(data)  # 放入模型进行测试
        label_hat = softmax(output)
        label_hat = label_hat.cpu()
        label_hat = label_hat.detach().numpy()
        df.at[i, 'label'] = label_hat[0, 1].clip(min=0.005, max=0.995)
    print(df)
    df.to_csv('pred1.csv', index=None)


def pred_img(ipath):
    model = get_model()
    model = model.to(device)
    checkpoint = torch.load('/home/ding/PycharmProjects/catsvsdogs/rennet18_epoch_12_.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    img = PIL.Image.open(ipath)
    plt.figure("Image") # 图像窗口名称
    plt.imshow(img)
    plt.axis('on') # 关掉坐标轴为 off
    data = transform_test(img)
    data = data.view(1, 3, 224, 224)
    data = data.cuda()
    output = model.forward(data)  # 放入模型进行测试
    label_hat = nn.functional.softmax(output,dim = 1)
    label_hat = label_hat.cpu()
    label_hat = label_hat.detach().numpy()
    print(label_hat)
    if label_hat[0,1] > 0.5:
        print("It's a dog picture!",label_hat[0,1])
        plt.title('dog')  # 图像题目
    else:
        print("It's a cat picture!",label_hat[0,0])
        plt.title('cat')  # 图像题目
    plt.show()


def show_transform_result(ipath,transform):
    img = PIL.Image.open(ipath)
    result = transform(img)
    img_1 = torchvision.transforms.ToPILImage()(result).convert('RGB')
    plt.figure("Image1")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('orgin_image')  # 图像题目
    plt.figure("Image2")  # 图像窗口名称
    plt.imshow(img_1)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('result_image')  # 图像题目
    plt.show()

if __name__=="__main__":
    train()
    #pred_img(sys.argv[1])
    #show_transform_result(sys.argv[1],transform = transform_test)

