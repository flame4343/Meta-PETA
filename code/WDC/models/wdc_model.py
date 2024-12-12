import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, lr_scheduler
from torch.optim.lr_scheduler import LinearLR
from models.loss import MAPE, SMAPE, SatsfScore, MY_MAPE_FT


PAD_IDX = 0

class Wide(nn.Module):
    def __init__(self, wide_config, dropout=0.1):
        super(Wide, self).__init__()
        self.config = wide_config
        for feature in self.config["sparse"]:
            a = nn.Embedding(feature["size"], feature["dim"])
            nn.init.uniform_(a.weight)
            setattr(self, "embedding_%s" % feature["name"], a)
        self.flatten = nn.Flatten(1)
        self.linear1 = nn.Linear(self.feature_dim + 56 + 32, 256)
        self.linear2 = nn.Linear(256, 256)
    @property
    def feature_dim(self) -> int:
        sparse_dim = sum([feature["dim"]
                          for feature in self.config["sparse"]])
        return sparse_dim

    def forward(self, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
                categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
                logistic, cnn_rnn):
        sparse_features = []
        wide_spec_features = [categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
                categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class]
        for index, feature in enumerate(self.config["sparse"]):
            embed = getattr(self, "embedding_%s" % feature["name"])(wide_spec_features[index])
            sparse_features.append(self.flatten(embed))
        sparse_feature = torch.cat(sparse_features, -1)
        logistic = self.flatten(logistic)
        features = torch.cat([sparse_feature, logistic, cnn_rnn], 1)
        out = self.linear1(features)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        return out

class Deep(nn.Module):
    def __init__(self, dropout=0.1):
        super(Deep, self).__init__()
        self.flatten = nn.Flatten(1)
        self.linear1 = nn.Linear(47, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, 256)

    def forward(self, dense):
        dense = self.flatten(dense)
        out = self.linear1(dense)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        out = F.relu(out)
        return out

class LuongAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(LuongAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, query, key, value, v_mask=None):
        scores = torch.bmm(query, key.transpose(1, 2))
        if v_mask is not None:
            scores_mask = torch.unsqueeze(v_mask, axis=-2)
            padding_mask = torch.logical_not(scores_mask)
            scores -= 1.0e9 * padding_mask.float()
        weights = self.softmax(scores)
        result = torch.bmm(weights, value)
        return result, weights

class CNNAtt(nn.Module):
    def __init__(self, pretrain_path, link_num, embedding_dim=20, drop_out=0.1):
        super().__init__()
        self.link_embedding = nn.Embedding(link_num + 1, embedding_dim, padding_idx=0)
        if pretrain_path:
            nn.init.uniform_(self.link_embedding.weight[1:])
        else:
            nn.init.trunc_normal_(self.link_embedding.weight[1:], std=0.02, a=-0.04, b=0.04) # mask_zero=True
        self.link_sstatus_embedding = nn.Embedding(5, embedding_dim) #  mask_zero=False
        nn.init.uniform_(self.link_sstatus_embedding.weight)
        self.link_speed_limit_embedding = nn.Embedding(9, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.link_speed_limit_embedding.weight[1:])    # mask_zero=True
        self.stime_bn = nn.BatchNorm1d(1, momentum=0.01, eps=1e-3, affine=True)
        self.len_sSpeed_bn = nn.BatchNorm1d(2, momentum=0.01, eps=1e-3, affine=True)
        self.linear_link = nn.Linear(embedding_dim * 3 + 1 + 2, 32)
        # CNN
        self.conv = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding='same')
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        # attention
        self.attention = LuongAttention(32, 32, 32)

    def forward(self, train_rnn):
        # NOTE link id must be mapping float64
        link_id = train_rnn[:, :, 0].to(torch.long)
        link_start_time = train_rnn[:, :, 1:2].to(torch.float32)
        link_start_status = train_rnn[:, :, 2].to(torch.int32)
        link_length_sSpeed = train_rnn[:, :, 3:5].to(torch.float32)
        link_speed_limit_class = train_rnn[:, :, 5].to(torch.int32)
        link_id_embed = self.link_embedding(link_id)
        link_start_status_ebd = self.link_sstatus_embedding(link_start_status)
        link_speed_limit_class_ebd = self.link_speed_limit_embedding(link_speed_limit_class)
        link_start_time = torch.permute(link_start_time, (0, 2, 1))
        start_time_bn = self.stime_bn(link_start_time)
        start_time_bn = torch.permute(start_time_bn, (0, 2, 1))

        link_length_sSpeed = torch.permute(link_length_sSpeed, (0, 2, 1))
        length_sSpeed_bn = self.len_sSpeed_bn(link_length_sSpeed)
        length_sSpeed_bn = torch.permute(length_sSpeed_bn, (0, 2, 1))

        sparse_feature = torch.cat((link_id_embed, start_time_bn, link_start_status_ebd, length_sSpeed_bn, link_speed_limit_class_ebd), -1)

        seq = self.linear_link(sparse_feature)
        seq = F.relu(seq)
        value_mask = torch.logical_not(link_speed_limit_class == PAD_IDX)
        linear_mask = value_mask.to(torch.int)
        seq = linear_mask.unsqueeze(-1) * seq
        seq = torch.permute(seq, (0, 2, 1))
        cnn_rnn = self.conv(seq)
        cnn_rnn = torch.permute(cnn_rnn, (0, 2, 1))
        # For the attention mask
        attn_output, attn_output_weights = self.attention(cnn_rnn, cnn_rnn, cnn_rnn, value_mask)  # 使用Luong Attention
        # sum mask
        attn_output_mask = torch.unsqueeze(value_mask.float(), -1)
        attn_output = attn_output * attn_output_mask
        cnn_rnn = torch.sum(attn_output, axis=-2, keepdims=False)
        return cnn_rnn

class WDC(nn.Module):
    def __init__(self, model_config, embedding_dim: int = 20, dropout=0.1):
        super(WDC, self).__init__()
        wide_config = model_config["wide"]
        self.link_sum = model_config["link_num"]
        self.batch_size = model_config["batch_size"]
        self.pretrain_path = model_config["pretrain_path"]
        self.qeta_flag = model_config["qeta"]
        self.wide = Wide(wide_config, dropout)
        self.deep = Deep(dropout)
        self.cnn_rnn = CNNAtt(self.pretrain_path, self.link_sum, embedding_dim, dropout)
        self.mlp1 = nn.Linear(544, 128)
        self.mlp2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1, bias=False)
        if self.pretrain_path:
            nn.init.xavier_uniform_(self.output.weight)
        else:
            nn.init.xavier_normal_(self.output.weight)
        self.mape_loss = MAPE()
        self.smape_loss = SMAPE()
        self.my_mape_loss = MY_MAPE_FT()
        self.satsf_loss = SatsfScore()

    def forward(self, data, target):
        [dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, train_rnn], label = data, target
        out_deep = self.deep(dense)
        out_cnn_rnn = self.cnn_rnn(train_rnn)
        out_wide = self.wide(categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, out_cnn_rnn)
        features = torch.cat([out_deep, out_wide], 1)
        out = self.mlp1(features)
        out = F.relu(out)
        out = self.mlp2(out)
        out = F.relu(out)
        pred = self.output(out)
        mape = self.mape_loss(pred, label)
        smape = self.smape_loss(pred, label)
        my_mape = self.my_mape_loss(pred, label)
        myscore = self.satsf_loss(pred, label)
        return mape, smape, my_mape, myscore, pred
    def save_checkpoint(self, state, dirname, fmt='epoch{}-step{}-loss{:.4f}-mape{:.4f}-smape{:.4f}-my_mape{:.4f}-satsf{:.4f}-checkpoint.pth.tar', topk=1):
        epoch, steps, loss, mape, smape, my_mape, satsf = state['epoch'], state['steps'], state['loss'], state['mape'], state['smape'], state['my_mape'], state['satsf']
        save_name = os.path.join(dirname, fmt.format(epoch, steps, loss, mape, smape, my_mape, satsf))
        torch.save(state, save_name)
        return save_name
    def config_optimizers(self, model, args):
        if hasattr(model, "module"):
            parmeters = model.module.parameters()
        else:
            parmeters = model.parameters()
        optimizer = Adam(parmeters, lr=args.lr, weight_decay=args.weight_decay)
        if self.pretrain_path:
            scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=100)
        else:
            scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=15)
        return optimizer, scheduler
