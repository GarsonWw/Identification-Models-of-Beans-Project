import os
import sys
import json
import pickle
import random
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import argparse
from .loss_plm import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square

def adjust_learning(optimizer,epoch,net):
    if (epoch == 20):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.00005, momentum=0.9)
    elif (epoch == 30):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0005, weight_decay=0.00005, momentum=0.9)
    elif (epoch == 50):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, weight_decay=0.00005, momentum=0.9)
    return optimizer

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch,alpha_plan,beta1_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # only change beta1


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[num_model {} epoch {}] mean loss {}".format(round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def dualevaluate(model1,model2,data_loader, device):
    model1.eval()
    model2.eval()

    # 用于存储预测正确的样本个数
    sum_num1 = torch.zeros(1).to(device)
    sum_num2 = torch.zeros(1).to(device)
    sum_num_com = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader,colour='green')
    MAE = 0
    target_num = torch.zeros((1, 200))
    predict_num = torch.zeros((1, 200))
    acc_num = torch.zeros((1, 200))
    for step, data in enumerate(data_loader):
        images, labels = data
        output1 = model1(images.to(device))
        output2 = model2(images.to(device))
        output_com = output1+output2
        pred1 = torch.max(output1, dim=1)[1]
        pred2 = torch.max(output2, dim=1)[1]
        pred_com = torch.max(output_com,dim=1)[1]
        sum_num1 += torch.eq(pred1, labels.to(device)).sum()
        sum_num2 += torch.eq(pred2, labels.to(device)).sum()
        sum_num_com += torch.eq(pred_com, labels.to(device)).sum()
        MAE+=mean_squared_error(labels.cpu(), pred1.cpu())

    return sum_num1.item(),sum_num2.item(),sum_num_com.item(),MAE/(len(data_loader)*32)

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=216)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="/home/junnkidesu/dataset/fewbutterfly")

    # download model weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
    parser.add_argument('--weights', type=str, default='./efficientnetb0.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    return opt


def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk) #maxk = 5
    N = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target is in shape (N,)
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(dim=0, keepdim=True)  # size is 1
        res.append(correct_k.mul_(100.0 / N))
    return res

def train(train_loader, epoch, model1, optimizer1, model2, optimizer2,device,rate_schedule):
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0
    epoch_loss1 = []
    epoch_loss2 = []
    train_loader = tqdm(train_loader)
    for it, (images, labels) in enumerate(train_loader):
        # if it > 60:
        #     break

        labels = labels.to(device)
        # Forward + Backward + Optimize
        logits1 = model1(images.to(device))
        prec1, _ = accuracy(logits1, labels, topk=(1, 1))
        train_total += 1
        train_correct += prec1.item()
        logits2 = model2(images.to(device))
        prec2, _ = accuracy(logits2, labels, topk=(1, 1))
        train_total2 += 1
        train_correct2 += prec2.item()
        loss_1, loss_2 = peer_learning_loss(logits1, logits2, labels, rate_schedule[epoch])
        epoch_loss1.append(loss_1.item())
        epoch_loss2.append(loss_2.item())

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()


    # train_acc1 = float(train_correct) / float(train_total)
    # train_acc2 = float(train_correct2) / float(train_total2)
    return train_correct, train_correct2

class PenaltyLoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True,
                 reweight_epoch=-1,
                 base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor  # 1.0
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch  # 80

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0

            if reweight_epoch != -1:
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                                  requires_grad=False).cuda()

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))

        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)

            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature

            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)

            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist,
                                                                                                      mean_output_dist,
                                                                                                      reduction='batchmean')

        return loss