import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import learn2learn as l2l
from utils.random import *
from datasets.eta_dataset import *
import torch
import numpy as np
import torch.nn as nn
import math
from models.loss import *
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

class PETA(nn.Module):
    def __init__(self, opt):
        super(PETA, self).__init__()
        self.opt = opt

        # embedding
        self.id_embedding = nn.Embedding(opt.car_num, opt.embedding_size)
        self.static_trans = nn.Linear(opt.static_feature_num, opt.embedding_size, bias=False)
        self.driver_static_ln = nn.LayerNorm(normalized_shape=2 * opt.embedding_size)

        # NN
        self.fea_trans = nn.Linear(opt.target_feature_num, opt.hidden_size, bias=False)
        self.top_5_trans = nn.Linear(opt.top_5_feature_num, opt.hidden_size, bias=False)

        self.fc0 = nn.Linear(2 * opt.embedding_size + opt.hidden_size, opt.hidden_size)
        self.fc1 = nn.Linear(opt.hidden_size, opt.hidden_size2)
        self.fc2 = nn.Linear(opt.hidden_size2, 1)

        # MSA
        self.multihead_self_attn = nn.MultiheadAttention(embed_dim=opt.target_feature_num, num_heads=opt.heads_num, batch_first=True)
        self.fc_Preference_Learning_MLP = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.Preference_Learning_ln = nn.LayerNorm(normalized_shape=opt.hidden_size)

        # MCA
        self.multihead_cross_attn = nn.MultiheadAttention(embed_dim=opt.hidden_size, num_heads=opt.heads_num, batch_first=True)
        self.fc_Behavior_Prediction_MLP = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.Behavior_Prediction_ln = nn.LayerNorm(normalized_shape=opt.hidden_size)


        self.relu = nn.ReLU()
        self.gate_1 = nn.Linear(2 * opt.hidden_size, opt.hidden_size, bias=False)

        # loss function

        self.delta = opt.Asymmetric_delta
        self.alpha = opt.Asymmetric_alpha
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.reset_parameters()


    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) == 2:
                stdv = 1.0 / math.sqrt(weight.size(1))
                weight.data.uniform_(-stdv, stdv)
    def MAELoss(self, y_hat, y):
        AE = torch.abs(y_hat - y)
        MAE = torch.mean(AE)
        return MAE
    def AsymmetricHuberLoss(self, y_hat, y):
        error = y_hat - y
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.full_like(abs_error, self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        weight_positive = self.alpha
        weight_negative = 1 - self.alpha
        loss = torch.where(error > 0, loss * weight_positive, loss * weight_negative)
        return torch.mean(loss)
    def min_max_normalize(self,tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)
    def forward(self, id, feature, static, top_5, y):
        h_id = self.id_embedding(id)
        h_static = self.static_trans(static)
        driver_static = torch.cat([h_id, h_static], dim=1)
        driver_static =self.driver_static_ln(driver_static)

        h_fea = self.fea_trans(feature)
        h_top_5 = self.top_5_trans(top_5)

        query = h_top_5
        key = h_top_5
        value = h_top_5
        Preference_Learning_1, attn_output_weights1 = self.multihead_self_attn(query, key, value)
        Preference_Learning_1 = Preference_Learning_1 + h_top_5
        Preference_Learning_MLP = self.fc_Preference_Learning_MLP(Preference_Learning_1)
        Preference_Learning_MLP = self.relu(Preference_Learning_MLP)
        Preference_Learning = Preference_Learning_MLP + Preference_Learning_1
        Preference_Learning = self.Preference_Learning_ln(Preference_Learning)

        Q = h_fea  # (batch_size, seq_len_q, embed_dim)
        Q = Q.unsqueeze(1)
        K = Preference_Learning  # (batch_size, seq_len_k, embed_dim)
        V = K

        Behavior_Prediction1, attn_output_weights2 = self.multihead_cross_attn(Q, K, V)
        Behavior_Prediction1 = Behavior_Prediction1 + Q
        Behavior_Prediction_MLP = self.fc_Behavior_Prediction_MLP(Behavior_Prediction1)
        Behavior_Prediction_MLP = self.relu(Behavior_Prediction_MLP)
        Behavior_Prediction = Behavior_Prediction_MLP + Behavior_Prediction1
        Behavior_Prediction = self.Behavior_Prediction_ln(Behavior_Prediction)
        Behavior_Prediction = Behavior_Prediction.squeeze(1)
        Behavior_Prediction = self.relu(Behavior_Prediction)

        h2 = torch.cat([driver_static, Behavior_Prediction], dim=1)
        h2 = self.fc0(h2)
        h2 = self.relu(h2)
        gate_1 = torch.relu(self.gate_1(torch.cat([h_fea, h2], dim=-1)))
        h3 = h_fea * gate_1 + h2 * (1 - gate_1)

        h = self.fc1(h3)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        y_hat = h.squeeze(1)
        loss = self.AsymmetricHuberLoss(y_hat, y)
        return loss, y, y_hat

class ETAMetaLearner(object):
    def __init__(self, train_data, opt):
        super(ETAMetaLearner).__init__()
        self.model = trans_to_cuda(PETA(opt))
        self.opt = opt
        self.maml = l2l.algorithms.MAML(self.model, lr=opt.lr, allow_unused=True, first_order=True)
        self.optimizer = torch.optim.Adam(self.maml.parameters(), lr=opt.meta_lr, weight_decay=opt.l2)
        self.task_sampler = TaskBatchGenerator(train_data=train_data)
        self.max_task_data_num = opt.max_task_data_num
    def train_tasks(self):
        sampler = self.task_sampler.getTaskBatch()
        for batch in tqdm(sampler):
            self.optimizer.zero_grad()
            task_num = 0
            total_loss = 0.0

            for data in batch:
                support_data, query_data = data
                learner = self.maml.clone()
                learner.train()
                for epoch in range(1):
                    val_loss = self.train_valid_base_model(learner, support_data, query_data)
                    total_loss += val_loss
                    task_num += 1

            total_loss /= task_num
            total_loss.backward()
            self.optimizer.step()

    def train_valid_base_model(self, learner, support_data, query_data):
        support_loader = DataLoader(support_data, num_workers=0, batch_size=self.opt.batch_size, shuffle=True, pin_memory=False)
        for i, data in enumerate(support_loader):
            loss, targets, scores = forward(learner, data)
            learner.adapt(loss)
        val_loss = self.validation(learner, query_data)

        return val_loss

    def validation(self, learner, query_data):
        query_loader = DataLoader(query_data, num_workers=0, batch_size=self.opt.batch_size, shuffle=True, pin_memory=False)
        data_num = len(query_data)
        total_loss = 0.0

        for data in query_loader:
            loss, targets, scores = forward(learner, data)
            loss *= data_num / self.max_task_data_num
            total_loss += loss

        return total_loss
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable

def forward(model, data):
    id, feature, static, top_5, target = data

    id = trans_to_cuda(id).long()
    feature = trans_to_cuda(feature).float()
    static = trans_to_cuda(static).float()
    top_5 = trans_to_cuda(top_5).float()
    target = trans_to_cuda(target).float()
    loss, target, score = model(id, feature, static, top_5, target)
    return loss, target, score

def train_test(model, opt, train_data, test_data, isMetaLearning=False):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = DataLoader(train_data, num_workers=0 if not isMetaLearning else 8, batch_size=opt.batch_size, shuffle=True, pin_memory=False)
    for i, data in enumerate(tqdm(train_loader)):
        model.optimizer.zero_grad()
        loss, targets, scores = forward(model, data)
        loss.backward()
        model.optimizer.step()
        total_loss += float(loss.item())
    print('\tLoss:\t%.3f' % total_loss)

    MSE, MAE, MAPE = test(model, opt, test_data, ~isMetaLearning)

    return MSE, MAE, MAPE
def test(model, opt, test_data, isTestset=True):
    if not isTestset:
        num_workers = 2
    else:
        num_workers = 0
    model.eval()
    test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=opt.batch_size, shuffle=False, pin_memory=False)

    SE = []
    AE = []
    APE = []
    total_loss = 0.0

    for data in test_loader:
        loss, targets, scores = forward(model, data)
        total_loss += float(loss.item())
        # MSE & MAE
        se = SquareError(scores, targets)
        SE.append(se)
        ae = AbsolError(scores, targets)
        AE.append(ae)
        ape = AbsolPercentageError(scores, targets)
        APE.append(ape)

    MSE = float(torch.mean(torch.cat(SE, dim=0)))
    MAE = float(torch.mean(torch.cat(AE, dim=0)))
    MAPE = float(torch.mean(torch.cat(APE, dim=0)))
    if isTestset:
        print('Test predicting: ', datetime.datetime.now(), '========== Test Loss:\t%.3f' % total_loss)
    else:
        print('Valid predicting: ', datetime.datetime.now(), '========== Valid Loss:\t%.3f' % total_loss)

    return MSE, MAE, MAPE
