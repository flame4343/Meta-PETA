import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import PolynomialFeatures
from torch import tensor
import numpy as np
import pandas as pd
import json
import os
import time
from tqdm import tqdm


CATEGORICAL_COLUMNS = ['C_distance_class', 'C_weekday_class', 'C_if_busytime_class', 'C_slice_id_class', 'C_city_class',
                      'C_day_before_2_type', 'C_day_before_1_type', 'C_day_type', 'C_day_after_1_type', 'C_day_after_2_type']
CONTINUOUS_COLUMNS = ['feature1','feature2','feature3',]
FEATURE_SELECT = ['C_distance_class', 'C_weekday_class', 'C_if_busytime_class', 'C_slice_id_class', 'C_city_class',
                  'C_day_before_2_type', 'C_day_before_1_type', 'C_day_type', 'C_day_after_1_type', 'C_day_after_2_type'
                  'feature1','feature2','feature3',
                  'ata']
def load_json(file_path):
    with open(file_path, "r", encoding='UTF-8') as json_file:
        data = json.load(json_file)
    return data
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
def link2zero(d, a):
    a_link = a[:, :, 0]
    a_other = a[:, :, 1:]
    a_link = np.array(a_link, dtype=int)
    indexer = np.array([int(d.get(str(i), 0)) for i in range(a_link.min(), a_link.max() + 1)], dtype=np.long)
    c_link = indexer[(a_link - a_link.min())][:, :, None]
    a_new = np.concatenate([c_link, a_other], axis=2)
    return a_new
def get_index(data_type, dataset_city, is_distributed, _new_data_flag):
    file_step_num = 1
    file_start_index = 0
    file_end_index = 1
    if data_type == 'test':
        if dataset_city == 'nationwide':
            file_start_index = 98
            file_end_index = 99
    elif is_distributed:
        if dataset_city == 'nationwide':
            if data_type == 'train':
                file_end_index = 90
                file_step_num = 1
            elif data_type == 'val':
                file_start_index = 95
                file_end_index = 96

    return file_start_index, file_end_index, file_step_num
def gene_process(deepfm_file_path, link_dict, train_rnn, data_type, pretrain_path, qeta_data_flag, read_trace_data, mean, std, city_dict, start_time):
    train_rnn = link2zero(link_dict, train_rnn)
    read_trace_data = pd.DataFrame(read_trace_data)
    read_trace_data = read_trace_data[FEATURE_SELECT]
    read_trace_data['zsl_order_link_time_skew'] = 0
    read_trace_data['C_city_class'] = read_trace_data['C_city_class'].map(city_dict)
    read_trace_data = read_trace_data.fillna({'C_city_class': 370})
    read_trace_data['C_city_class'] = read_trace_data['C_city_class'].astype(int)
    categ_distance_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[0]]], dtype='int32')
    categ_weekday_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[1]]], dtype='int32')
    categ_if_busytime_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[2]]], dtype='int32')
    categ_slice_id_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[3]]], dtype='int32')
    categ_city_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[4]]], dtype='int32')
    categ_day_before2_type_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[5]]], dtype='int32')
    categ_day_before1_type_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[6]]], dtype='int32')
    categ_day_type_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[7]]], dtype='int32')
    categ_day_after1_type_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[8]]], dtype='int32')
    categ_day_after2_type_class = np.array(read_trace_data[[CATEGORICAL_COLUMNS[9]]], dtype='int32')
    x_train_conti_t = np.array(read_trace_data[CONTINUOUS_COLUMNS], dtype='float64')
    x_train_conti_t = np.array((x_train_conti_t - mean) / std, dtype='float32')
    x_train_conti_t = np.expand_dims(x_train_conti_t, 2)
    x_train_conti_t[np.isnan(x_train_conti_t)] = 0
    dense = x_train_conti_t
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    logistic = np.expand_dims(np.array(poly.fit_transform(read_trace_data[CATEGORICAL_COLUMNS]), dtype='int32'), 2)
    label_all = np.expand_dims(np.array(read_trace_data['ata'], dtype='float32'), 1)
    end_time = time.time()
    print(data_type, "data set load consume time: ", int(end_time - start_time) / 60, 'min', " data_count:", len(label_all))
    return categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
        categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
        dense, logistic, train_rnn, label_all

def load_data(deepfm_file_path, link_file_path, data_type, link_dict, city_dict, mean, std, start_index, file_step_num, qeta_data_flag, pretrain_path):
    start_time = time.time()
    print("data load begins: ", data_type, "data set.", "start_index:", start_index)
    if start_index == -1:
        if deepfm_file_path.endswith('.csv'):
            read_trace_data = pd.read_csv(deepfm_file_path, encoding='utf-8')
            train_rnn = np.array(np.load(link_file_path)['data'], dtype='float64')
        else:
            read_trace_data = pd.read_csv(deepfm_file_path + data_type + '.csv', encoding='utf-8')
            train_rnn = np.array(np.load(link_file_path + data_type + '.npz')['data'], dtype='float64')
        return gene_process(deepfm_file_path, link_dict, train_rnn, data_type, pretrain_path, qeta_data_flag, read_trace_data, mean, std, city_dict, start_time)
    else:
        file_start_index = start_index
        file_end_index = start_index + file_step_num
        x_train_categ_distance_class, x_train_categ_weekday_class, x_train_categ_if_busytime_class, x_train_categ_slice_id_class, x_train_categ_city_class = [[] for x in range(6)]
        x_train_categ_day_before2_type_class, x_train_categ_day_before1_type_class, x_train_categ_day_type_class, x_train_categ_day_after1_type_class, x_train_categ_day_after2_type_class = [[] for x in range(5)]
        x_train_conti, x_train_categ_poly, train_rnn_datas, label_all_l = [[] for x in range(4)]
        for f_index in tqdm(range(file_start_index, file_end_index)):
            read_trace_data = pd.read_csv(deepfm_file_path + str(f_index) + '.csv', encoding='utf-8')
            read_link_data = np.array(np.load(link_file_path + str(f_index) + '.npz')['data'], dtype='float64')
            t_data = gene_process(deepfm_file_path, link_dict, read_link_data, data_type, pretrain_path, qeta_data_flag, read_trace_data, mean, std, city_dict, start_time)
            x_train_categ_distance_class.append(t_data[0])
            x_train_categ_weekday_class.append(t_data[1])
            x_train_categ_if_busytime_class.append(t_data[2])
            x_train_categ_slice_id_class.append(t_data[3])
            x_train_categ_city_class.append(t_data[4])
            x_train_categ_day_before2_type_class.append(t_data[5])
            x_train_categ_day_before1_type_class.append(t_data[6])
            x_train_categ_day_type_class.append(t_data[7])
            x_train_categ_day_after1_type_class.append(t_data[8])
            x_train_categ_day_after2_type_class.append(t_data[9])
            x_train_conti.append(t_data[10])
            x_train_categ_poly.append(t_data[11])
            train_rnn_datas.append(t_data[12])
            label_all_l.append(t_data[13])
        categ_distance_class = np.concatenate(x_train_categ_distance_class, axis=0)
        categ_weekday_class = np.concatenate(x_train_categ_weekday_class, axis=0)
        categ_if_busytime_class = np.concatenate(x_train_categ_if_busytime_class, axis=0)
        categ_slice_id_class = np.concatenate(x_train_categ_slice_id_class, axis=0)
        categ_city_class = np.concatenate(x_train_categ_city_class, axis=0)
        categ_day_before2_type_class = np.concatenate(x_train_categ_day_before2_type_class, axis=0)
        categ_day_before1_type_class = np.concatenate(x_train_categ_day_before1_type_class, axis=0)
        categ_day_type_class = np.concatenate(x_train_categ_day_type_class, axis=0)
        categ_day_after1_type_class = np.concatenate(x_train_categ_day_after1_type_class, axis=0)
        categ_day_after2_type_class = np.concatenate(x_train_categ_day_after2_type_class, axis=0)
        dense = np.concatenate(x_train_conti, axis=0)
        logistic = np.concatenate(x_train_categ_poly)
        train_rnn = np.concatenate(train_rnn_datas,axis=0)
        label_all = np.concatenate(label_all_l, axis=0)
        return categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            dense, logistic, train_rnn, label_all
class ETAWDRDataset(Dataset):
    def __init__(self, args, data_type):
        self.data_type = data_type
        self.pretrain_path = args.pretrain_path
        self.qeta = args.qeta
        _data_path = args.data_dir
        _config_data_path = args.config_data_path
        _dataset_city = args.dataset_city
        _is_distributed = args.is_distributed
        self.file_start_index = -1
        self.file_end_index = -1
        self.file_step_num = -1
        self.qeta_data_flag = False
        _new_data_flag = False
        if _dataset_city == 'nationwide':
            self.link_num = 100000000
            self.link_dict = load_json(_config_data_path + 'link_dict.json')
            self.deepfm_file_path = os.path.join(_data_path, 'deepFM_')
            self.link_file_path = os.path.join(_data_path, 'link_')
            self.file_start_index, self.file_end_index, self.file_step_num = get_index(self.data_type, _dataset_city, _is_distributed, _new_data_flag)
        print("starting load data ", _data_path, self.data_type, "data set.")
        self.read_lines_seek()
        if self.file_start_index > -1:
            self.start_index = self.file_start_index
            self.cache_file_index = self.file_start_index
            self.single_file_len = int(self.total_lines / (self.file_end_index - self.file_start_index)) * self.file_step_num
        else:
            self.start_index = -1
            self.cache_file_index = 0
            self.single_file_len = self.total_lines
        print(self.data_type, " single_file_len: ", self.single_file_len)
        self.city_dict = load_json(_config_data_path + 'city_dict.json')
        self.mean = np.loadtxt(_config_data_path + 'feature_mean.csv', delimiter=',')
        self.std = np.loadtxt(_config_data_path + 'feature_std.csv', delimiter=',')
        self.all_data = load_data(self.deepfm_file_path, self.link_file_path, self.data_type, self.link_dict, self.city_dict, self.mean, self.std, self.start_index, self.file_step_num, self.qeta_data_flag, self.pretrain_path)
    def read_lines_seek(self):
        self.line_dict = {}
        first_csv_file = pd.read_csv(self.deepfm_file_path)
        length = len(first_csv_file)
        for i in range(length):
            self.line_dict[str(i)] = int(i)
        self.total_lines = len(self.line_dict)
        print("total_lines:", self.data_type, self.total_lines)
    def __len__(self):
        return self.total_lines
    def __getitem__(self, idx):
        offset = self.line_dict[str(idx)]
        f_index = offset // self.single_file_len
        cur_offset = offset % self.single_file_len
        if self.cache_file_index != f_index:
            self.cache_file_index = f_index
            self.start_index = self.start_index + self.file_step_num
            if self.start_index < self.file_end_index:
                self.all_data = load_data(self.deepfm_file_path, self.link_file_path, self.data_type, self.link_dict, self.city_dict, self.mean, self.std,  self.start_index, self.file_step_num, self.qeta_data_flag, self.pretrain_path)
            else:
                self.start_index = 0
                self.all_data = load_data(self.deepfm_file_path, self.link_file_path, self.data_type, self.link_dict, self.city_dict, self.mean, self.std,  self.start_index, self.file_step_num, self.qeta_data_flag, self.pretrain_path)
        categ_distance_class = self.all_data[0][cur_offset]
        categ_weekday_class = self.all_data[1][cur_offset]
        categ_if_busytime_class = self.all_data[2][cur_offset]
        categ_slice_id_class = self.all_data[3][cur_offset]
        categ_city_class = self.all_data[4][cur_offset]
        categ_day_before2_type_class = self.all_data[5][cur_offset]
        categ_day_before1_type_class = self.all_data[6][cur_offset]
        categ_day_type_class = self.all_data[7][cur_offset]
        categ_day_after1_type_class = self.all_data[8][cur_offset]
        categ_day_after2_type_class = self.all_data[9][cur_offset]
        dense = self.all_data[10][cur_offset]
        logistic = self.all_data[11][cur_offset]
        train_rnn = self.all_data[12][cur_offset]
        label = self.all_data[13][cur_offset]
        # CONTINUOUS_COLUMNS
        dense = torch.from_numpy(dense)
        # CATEGORICAL_COLUMNS
        categ_distance_class = tensor(categ_distance_class)
        categ_weekday_class = tensor(categ_weekday_class)
        categ_if_busytime_class = tensor(categ_if_busytime_class)
        categ_slice_id_class = tensor(categ_slice_id_class)
        categ_city_class = tensor(categ_city_class)
        categ_day_before2_type_class = tensor(categ_day_before2_type_class)
        categ_day_before1_type_class = tensor(categ_day_before1_type_class)
        categ_day_type_class = tensor(categ_day_type_class)
        categ_day_after1_type_class = tensor(categ_day_after1_type_class)
        categ_day_after2_type_class = tensor(categ_day_after2_type_class)
        logistic = torch.from_numpy(logistic)
        # rnn features
        train_rnn = torch.from_numpy(train_rnn)
        # ata
        label = tensor(label)
        return dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, train_rnn, label
    def generate_config(self):
        basic_info = {"link_num": self.link_num}
        wide_config = {
            "dense": {"size": 7},
            "sparse": [
                {"col": 0, "name": "input_2_distance_class", "size": 5, "dim": 5},
                {"col": 1, "name": "input_2_weekday_class", "size": 7, "dim": 7},
                {"col": 2, "name": "input_2_if_busytime_class", "size": 2, "dim": 2},
                {"col": 3, "name": "input_2_slice_id_class", "size": 72, "dim": 72},
                {"col": 4, "name": "input_2_city_class", "size": 371, "dim": 371},
                {"col": 5, "name": "input_2_day_before2_type_class", "size": 4, "dim": 4},
                {"col": 6, "name": "input_2_day_before1_type_class", "size": 4, "dim": 4},
                {"col": 7, "name": "input_2_day_type_class", "size": 4, "dim": 4},
                {"col": 8, "name": "input_2_day_after1_type_class", "size": 4, "dim": 4},
                {"col": 9, "name": "input_2_day_after2_type_class", "size": 4, "dim": 4}
            ],
        }
        return basic_info, wide_config

