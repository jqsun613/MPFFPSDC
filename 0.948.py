import random
import torch.nn.functional as F
import torch.nn as nn
from MPFFPSDC_model import MPFFPSDC
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics



# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))

    model.train()
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)): #enumerate返回两个值一个是序号，一个是数据
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)#保证y为1列的向量
        y = y.squeeze(1)#去掉维度为1的维度列向量变行向量
        optimizer.zero_grad() #把loss关于weight的导数变成0.
        output = model(data1, data2) #进入前向传播2
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        #print('train:',loss.item())
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for batch_idx,data in enumerate(zip(drug1_loader_test, drug2_loader_test)):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()    #dim=1表示对第一维不同的数组计算
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))  #取出ys中label为1的概率
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)#36行
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()  #numpy.flattren把tensor转成一维数组


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

modeling = MPFFPSDC

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 500

print('Learning rate: ', LR )
print('Epochs: ', NUM_EPOCHS)
datafile = 'new_labels_0_10'
# CPU or GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')#x药物特征 y是标签 c_size原子个数
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

lenth = len(drug1_data)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)#测试集数目

seed = 16
random.seed(seed)
random_num = random.sample(range(0, lenth), lenth)   #ramdom.sample随机截取列表指定长度的随机数
i=0
test_num = random_num[pot*i:pot*(i+1)]
train_num = random_num[:pot*i] + random_num[pot*(i+1):]
print(len(train_num))
print(len(test_num))

drug1_data_train = drug1_data[train_num]
drug1_data_test = drug1_data[test_num]
print('type(drug1_data_train)', type(drug1_data_train))
print('drug1_data_train[0]', drug1_data_train[0])
print('len(drug1_data_train)', len(drug1_data_train))
drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None) #封装成batch_size大小的Tensor 每次取batch_size个数据
drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)


drug2_data_test = drug2_data[test_num]
drug2_data_train = drug2_data[train_num]
drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

model = modeling().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR ) #parameters会生成一个生成器（迭代器），生成器每次生成的tensor类型的数据/home/b/桌面/SimGNN-master

model_pathe = 'data/github/weight/' + '0.948.pt'
# with open(file_AUCs, 'w') as f:
#     f.write(AUCs + '\n')
model_param = torch.load(model_pathe)
model.load_state_dict(model_param)

best_auc = 0

T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
AUC = roc_auc_score(T, S)
precision, recall, threshold = metrics.precision_recall_curve(T, S)
PR_AUC = metrics.auc(recall, precision)
BACC = balanced_accuracy_score(T, Y)
tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
TPR = tp / (tp + fn)
PREC = precision_score(T, Y)
ACC = accuracy_score(T, Y)
KAPPA = cohen_kappa_score(T, Y)
recall = recall_score(T, Y)
best_auc = AUC

print('best_auc', best_auc)


