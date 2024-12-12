import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []
    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)
    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)
    if len(batch[0]) == 10:
        output.append(torch.stack([x[-3] for x in batch]))
    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))
    return output

def collate_batch(batch):
    dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
        categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
        logistic, train_rnn, label = zip(*batch)
    dense = pad_sequence(dense, batch_first=True)
    categ_distance_class = pad_sequence(categ_distance_class, batch_first=True)
    categ_weekday_class = pad_sequence(categ_weekday_class, batch_first=True)
    categ_if_busytime_class = pad_sequence(categ_if_busytime_class, batch_first=True)
    categ_slice_id_class = pad_sequence(categ_slice_id_class, batch_first=True)
    categ_city_class = pad_sequence(categ_city_class, batch_first=True)
    categ_day_before2_type_class = pad_sequence(categ_day_before2_type_class, batch_first=True)
    categ_day_before1_type_class = pad_sequence(categ_day_before1_type_class, batch_first=True)
    categ_day_type_class = pad_sequence(categ_day_type_class, batch_first=True)
    categ_day_after1_type_class = pad_sequence(categ_day_after1_type_class, batch_first=True)
    categ_day_after2_type_class = pad_sequence(categ_day_after2_type_class, batch_first=True)
    logistic = pad_sequence(logistic, batch_first=True)
    train_rnn = pad_sequence(train_rnn, batch_first=True)
    label_all = pad_sequence(label, batch_first=True)

    return dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
        categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
        logistic, train_rnn, label_all
