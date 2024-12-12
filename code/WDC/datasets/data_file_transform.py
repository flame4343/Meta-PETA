import pandas as pd
import os
import time
from tqdm import tqdm

def data_offset_gene(data_path:str, data_type:str):
    start_time = time.time()
    print("data load begins: ", data_type, "data set.")
    if data_type == 'train':
        file_start_index = 0
        file_end_index = 6
    elif data_type == 'val':
        file_start_index = 90
        file_end_index = 91
    deepfm_file_path = os.path.join(data_path, 'deepFM_')
    read_trace_data = pd.read_csv(deepfm_file_path + str(file_start_index) + '.csv')
    single_data_len = len(read_trace_data)
    number_of_lines = (file_end_index - file_start_index) * single_data_len
    line = 0
    output_file = deepfm_file_path + str(file_start_index)
    of = open(f"{output_file}_offset.json", "a")
    for f_index in tqdm(range(file_start_index, file_end_index)):
        large_file_path = deepfm_file_path + str(f_index) + '.csv'
        offset_dict = {}
        with open(large_file_path, 'rb') as f:
            for i in range(single_data_len):
                if line % 1000000 == 0:
                    print(line)
                f.seek(line)
                offset = f.tell()
                offset_dict[line] = offset
                f.readline()
                content = f"{line} {offset}\n"
                of.write(content)
                line = line + 1
    assert number_of_lines == line, "number_of_lines is wrong, plz check."
    of.close()
    end_time = time.time()
    print(data_type, "data set load consume time: ", int(end_time - start_time) / 60, 'min', " data_count:", number_of_lines)


if __name__=='__main__':
    large_file_path = './data/'
    data_offset_gene(large_file_path, "train")
    data_offset_gene(large_file_path, "val")