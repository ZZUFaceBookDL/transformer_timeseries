import datetime
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score

from tst import Transformer
from tst.loss import OZELoss

from src.dataset import OzeDataset
from src.utils import compute_loss, fit, Logger, kfold
from src.benchmark import LSTM, BiGRU, ConvGru, FFN
from src.metrics import MSE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties
import math


# 数据集路径选择
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\ArabicDigits\\ArabicDigits.mat'
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\AUSLAN\\AUSLAN.mat'
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\CharacterTrajectories\\CharacterTrajectories.mat'
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\CMUsubject16\\CMUsubject16.mat'  # lenth=29,29  input=580 channel=62 output=2
path = 'E:\\PyCharmWorkSpace\\mtsdata\\ECG\\ECG.mat'  # lenth=100  input=152 channel=2 output=2
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\JapaneseVowels\\JapaneseVowels.mat'  # lenth=270  input=29 channel=12 output=9
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\KickvsPunch\\KickvsPunch.mat'  # lenth=270  input=29 channel=12 output=9
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\Libras\\Libras.mat'  # lenth=180  input=45 channel=2 output=15
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\UWave\\UWave.mat'  # lenth=4278  input=315 channel=3 output=8

# 获取相关参数可在 test1.py 中进行测试
data_length_p = 100  # 测试集数据量
d_input = 152  # 时间部数量
d_channel = 2  # 时间序列维度
d_output = 2  # 分类类别
EPOCHS = 1
BATCH_SIZE = 40
LR = 1e-4
optimizer_p = 'Adagrad'  # 优化器

draw_key = 1  # 大于等于draw_key才保存结果图
test_interval = 1  # 调用test()函数的epoch间隔
reslut_figure_path = 'result_figure'  # 保存结果图像的路径

# Model parameters
d_model = 512  # Lattent dim
q = 6  # Query size
v = 6  # Value size
h = 8  # Number of heads
N = 8  # Number of encoder and decoder to stack
dropout = 0.2  # Dropout rate
pe = True  # Positional encoding

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Load dataset
dataset_train = OzeDataset(path, 'train')
dataset_test = OzeDataset(path, 'test')
dataloader_train = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = Data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)

net = Transformer(d_input, d_channel, d_model, d_output, q, v, h, N,
                  dropout=dropout, pe=pe).to(device)
if optimizer_p == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)
elif optimizer_p == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=LR)
elif optimizer_p == 'Adamax':
    optimizer = optim.Adamax(net.parameters(), lr=LR)
elif optimizer_p == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)

# 创建损失函数
loss_function = OZELoss()

# 记录准确率，作图用
correct_list = []
correct_list_ontrain = []
def test(dataloader_test, flag='test_set'):
    correct = 0.0
    total = 0
    with torch.no_grad():
        for x_test, y_test in dataloader_test:
            enc_inputs, dec_inputs = x_test.to(device), y_test.to(device)
            test_outputs = net(enc_inputs)
            _, predicted = torch.max(test_outputs.data, dim=1)
            total += dec_inputs.size(0)
            correct += (predicted.float() == dec_inputs.float()).sum().item()
        if flag == 'test_set':
            correct_list.append((100 * correct / total))
        elif flag == 'train_set':
            correct_list_ontrain.append((100 * correct / total))
        print(f'Accuracy on {flag}: %d %%' % (100 * correct / total))


#  记录损失值 作图用
loss_list = []
# 记录起始时间
begin_time = time()
pbar = tqdm(total=EPOCHS)

# Prepare loss history
for idx_epoch in range(EPOCHS):
    for idx_batch, (x, y) in enumerate(dataloader_train):
        optimizer.zero_grad()

        # Propagate input
        netout = net(x.to(device))

        # Comupte loss
        loss = loss_function(y.to(device), netout)
        print('Epoch:', '%04d' % (idx_epoch + 1), 'loss =', '{:.6f}'.format(loss))
        loss_list.append(loss.item())

        # Backpropage loss
        loss.backward()

        # Update weights
        optimizer.step()

    if ((idx_epoch + 1) % test_interval) == 0:
        test(dataloader_test)
        test(dataloader_train, 'train_set')

    if pbar is not None:
        pbar.update()

end_time = time()
time_cost = round((end_time - begin_time) / 60, 2)


# 结果可视化 包括绘图和结果打印
def result_visualization():
    my_font = fp(fname=r"C:\windows\Fonts\msyh.ttc")  # 2、设置字体路径

    # 设置风格
    # plt.style.use('ggplot')
    plt.style.use('seaborn')

    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(311)  # 创建两个子图
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # 添加折线
    ax2.plot(correct_list, color='red', label='on Test Dataset')
    ax2.plot(correct_list_ontrain, color='blue', label='on Train Dataset')

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')

    # 设置文本
    fig.text(x=0.13, y=0.4, s=f'最小loss：{min(loss_list)}' '    '
                              f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}' '    '
                              f'最后一轮loss:{loss_list[-1]}' '\n'
                              f'最大correct：{max(correct_list)}%' '    '
                              f'最大correct对应的已训练epoch数:{(correct_list.index(max(correct_list)) + 1) * test_interval}' '    '
                              f'最后一轮correct：{correct_list[-1]}%' '\n'
                              f'd_model={d_model}   q={q}   v={v}   h={h}   N={N} drop_out={dropout}' '\n'
                              f'共耗时{round(time_cost, 2)}分钟', FontProperties=my_font)

    # 保存结果图   测试不保存图（epoch少于draw_key）
    if EPOCHS >= draw_key:
        plt.savefig(
            f'{reslut_figure_path}/{max(correct_list)}% {optimizer_p} epoch={EPOCHS} batch={BATCH_SIZE} lr={LR} [{d_model},{q},{v},{h},{N},{dropout}].png')

    # 展示图
    plt.show()

    print('正确率列表', correct_list)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：{max(correct_list)}\r\n'
          f'最correct对应的已训练epoch数:{(correct_list.index(max(correct_list)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_list[-1]}')

    print(f'共耗时{round(time_cost, 2)}分钟')


# 调用结果可视化
result_visualization()
