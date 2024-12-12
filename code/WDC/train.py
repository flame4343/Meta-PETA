import argparse
import glob
import math
import os
import time
import traceback
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from utils.random_seed import set_seed
from utils.heap import Heap
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.multiprocessing as mp
from tqdm import tqdm
from datasets.eta_wdr_dataset import ETAWDRDataset, FINETUNE_TEST_SET
from datasets.truncate_dataset import TruncateDataset
from datasets.collate_functions import collate_batch
from utils.seq_sampler_dist import My_sampler
from torch.utils.data import DataLoader, DistributedSampler
from utils.seq_sampler import SequentialDistributedSampler as seq_sampler
from models.wdr_model import wdc
from torch.utils.tensorboard import SummaryWriter
from indices_calc.satisfaction_score_fun import satisfaction_score_calc
from indices_calc.smape_indices import calc_indices

def get_parser():
    parser = argparse.ArgumentParser(description="Training eta wdc regression model")
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--config_data_path", type=str, required=True, help="bert config dir")
    parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="loss type")
    parser.add_argument("--val_check_interval", type=float, default=1.0, help="interval to do the validation")
    parser.add_argument("--is_distributed", action='store_true', default=False)
    parser.add_argument("--use_rnn", action='store_true', default=False)
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument("--max_epoch", type=int, default=100, help="max epochs to iterate")
    parser.add_argument("--start_epoch", type=int, default=0, help="the start number of epoch")
    parser.add_argument('--default_root_dir', default="", help="path to save training process")
    parser.add_argument('--gradient_clip_val', default=0., type=float, help="the gradient clip value")
    parser.add_argument('--evaluate', action='store_true', default=False, help='whether to do the evaluation')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:1234', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', help='distributed backend')
    parser.add_argument("--ngpus_per_node", type=int, default=2, help="number of GPUs per node for distributed training")
    parser.add_argument('--world_size', type=int, default=-1, help='number of nodes for distributed training')
    parser.add_argument('--nodes', type=int, default=1, help="number of nodes for distributed training")
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--topk', type=int, default=1, help='save topk checkpoint accoording to the test result of dev set')
    parser.add_argument('--offset', type=int, default=0, help='')
    parser.add_argument('--seed', type=int, default=None, help='seed for initializing training')
    parser.add_argument('--gpu', type=int, default=None, help='gpu id for training')
    parser.add_argument("--polydecay_ratio", type=float, default=4, help="ratio for polydecay learing rate scheduler.")
    parser.add_argument("--lr_scheduler", type=str, default="linear", help="lr scheduler")
    parser.add_argument("--save_path", type=str, default="", help="path to save evaluation result")
    parser.add_argument("--pretrain_path", type=str, default='', help="load didi pretrain model for finetune")
    parser.add_argument("--dataset_city", type=str, default='beijing', help="beijing dataset or nationwide dataset for training model")
    parser.add_argument('--qeta', action='store_true', default=False, help='whether qeta dataset for training model or prediction')
    parser.add_argument('--patience', type=int, default=7, help='early stops the training ig validation loss does not improve after this given patience')

    return parser

def get_dataloader(args, data_type, limit=None):
    dataset = ETAWDRDataset(args, data_type)
    print("get_dataloader dataset length: ", data_type, len(dataset))
    if limit is not None:
        dataset = TruncateDataset(dataset, limit)

    if data_type == "train" and args.is_distributed:
        if args.pretrain_path:
            sampler = DistributedSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, drop_last=True, shuffle=False)
    elif data_type == "test" and args.is_distributed:
        sampler = seq_sampler(dataset, args.batch_size)
    else:
        sampler = None
    if data_type == "train" and sampler is not None:
        return dataset, DataLoader(dataset=dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          collate_fn=collate_batch,
                          pin_memory=True,
                          sampler=sampler), sampler

    return dataset, DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      shuffle=True if data_type == "train" else False,
                      collate_fn=collate_batch,
                      sampler=sampler), sampler

def dict2device(data, device):
    data_tensors = []
    if device.type!='cpu':
        for item in data:
            if torch.is_tensor(item):
                data_tensors.append(item.to(device))
        return data_tensors
    else:
        return data_tensors

def pick_best_model(model, epoch, loss, mape_loss, smape_loss, my_mape_loss, satsf_loss, optimizer, iters, dir_name):
    if hasattr(model, "module"):
        save_path = model.module.save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.module.state_dict(),
            "loss": loss.item(),
            "mape": mape_loss,
            "smape": smape_loss,
            "my_mape": my_mape_loss,
            "satsf": satsf_loss,
            "optimizer": optimizer.state_dict(),
            "steps": iters
        }, dir_name)
    else:
        save_path = model.save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "loss": loss.item(),
            "mape": mape_loss,
            "smape": smape_loss,
            "my_mape": my_mape_loss,
            "satsf": satsf_loss,
            "optimizer": optimizer.state_dict(),
            "steps": iters
        }, dir_name)
    return save_path

def early_stopping(patience, score, best_score, counter):
    early_stopping_flag = False
    if best_score == 0:
        best_score = score
    elif score < best_score:
        counter += 1
        if counter >= patience:
            print("Early stopping", score, best_score, counter)
            early_stopping_flag = True
    else:
        best_score = score
        counter = 0
    return best_score, counter, early_stopping_flag

# Everything about the model training.
def train_wdc(data_loader, val_loader, model, optimizer, scheduler, sampler, args):
    queue = Heap(args.topk)
    dev_result = dict()
    device = torch.device("cuda:{}".format(args.local_rank) if args.local_rank > -1 else "cpu")
    total_iters_per_epoch = len(data_loader)
    if args.dataset_city == 'nationwide' and not args.pretrain_path:
        validate_step = 2000
        #validate_step = int(total_iters_per_epoch * 0.1)
    else:
        validate_step = int(total_iters_per_epoch * args.val_check_interval)
    #validate_step = int(total_iters_per_epoch * args.val_check_interval)
    print("validate_step:", len(data_loader), args.val_check_interval, validate_step, total_iters_per_epoch)
    output_infos_flag = (args.is_distributed and dist.get_rank() == 0) or not args.is_distributed
    if output_infos_flag:
        dir_name = args.default_root_dir
        os.makedirs(dir_name, exist_ok=True)
    writer = SummaryWriter(args.default_root_dir + '_train_tensorboard')
    best_score = 0
    counter = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        # set the random seed
        if args.pretrain_path:
            if sampler is not None:
                sampler.set_epoch(epoch)
        else:
            if args.seed is not None:
                set_seed(args.seed + epoch)
        model.train()
        iters = 0
        if output_infos_flag:
            pbar = tqdm(total=len(data_loader), ncols=0, desc=f"epoch{epoch + 1} Train", unit=" step")
        for dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, train_rnn, target in data_loader:
            iters += 1
            data = [dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, train_rnn]
            if args.is_distributed:
                data, target = dict2device(data, device), target.to(device)
            mape, loss, my_mape, satsf, _ = model(data, target)
            optimizer.zero_grad()
            if args.pretrain_path:
                satsf.backward()
            else:
                loss.backward()
            if args.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            optimizer.step()
            ts_index = epoch + iters / total_iters_per_epoch
            if output_infos_flag:
                pbar.update(1)
                pbar.set_postfix(mape="{:5f}".format(mape.item()), loss="{:5f}".format(loss.item()), my_mape="{:5f}".format(my_mape.item()), satsf="{:5f}".format(satsf.item()), step=iters)
                writer.add_scalars('train_loss', {'mape': mape.item(), 'loss': loss.item(), 'my_mape': my_mape.item(), 'satsf': satsf.item()}, ts_index)
            if iters % validate_step == 0:
                mape_loss, smape_loss, smape_loss_qs, my_mape_loss, my_mape_loss_qs, satsf_loss, satsf_loss_qs = validation(val_loader, model, args, device, writer, ts_index, output_infos_flag)
                if args.pretrain_path:
                    is_best = queue.is_insert(satsf_loss_qs)
                    score = satsf_loss_qs
                else:
                    is_best = queue.is_insert(my_mape_loss_qs)
                    score = my_mape_loss_qs
                if is_best and output_infos_flag:
                    print("top {} model saved.".format(args.topk))
                    try:
                        to_be_sub = queue.insert(my_mape_loss_qs)
                        if to_be_sub != -1:
                            path = dev_result[to_be_sub]
                            os.remove(path)
                        dev_result[my_mape_loss_qs] = pick_best_model(model, epoch, loss, mape_loss, smape_loss, my_mape_loss, satsf_loss, optimizer, iters, dir_name)
                    except:
                        traceback.print_exc()
        scheduler.step()
        if output_infos_flag:
            pbar.close()
    if output_infos_flag:
        writer.close()

def validation(data_loader, model, args, device, writer, ts_index, output_infos_flag):
    model.eval()
    if output_infos_flag:
        pbar = tqdm(total=len(data_loader), ncols=0, desc="Valid", unit=" trace")
    mape_loss = 0.
    running_loss = 0.
    my_mape_loss = 0.
    satsf_loss = 0.
    iters = 0

    for dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, train_rnn, target in data_loader:
        iters += 1
        data = [dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, train_rnn]
        if args.is_distributed:
            data, target = dict2device(data, device), target.to(device)
        with torch.no_grad():
            mape, loss, my_mape, satsf, pred = model(data, target)
        mape_loss += mape.item()
        running_loss += loss.item()
        my_mape_loss += my_mape.item()
        satsf_loss += satsf.item()
    mape_loss = round(mape_loss / iters, 4)
    smape_loss= round(running_loss / iters, 4)
    my_mape_loss = round(my_mape_loss / iters, 4)
    satsf_loss = round(satsf_loss / iters, 4)
    if output_infos_flag:
        writer.add_scalars('val_loss', {'mape': mape_loss, 'smape': smape_loss, 'my_mape': my_mape_loss, 'satsf': satsf_loss}, ts_index)
        pbar.set_postfix(mape=f"{mape_loss:.6f}", loss=f"{smape_loss:.6f}", my_mape=f"{my_mape_loss:.6f}", satsf=f"{satsf_loss:.6f}", step=ts_index)
        pbar.close()
    model.train()
    return mape_loss, smape_loss, -smape_loss, my_mape_loss, -my_mape_loss, satsf_loss, -satsf_loss

def evaluation(data_loader, model, args, device, test_dataroot):
    pred_start_time = time.time()
    model.eval()
    pbar = tqdm(total=len(data_loader), ncols=0, desc="test", unit=" trace")
    predict_results = []
    iters = -1
    for dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, train_rnn, target in data_loader:
        iters += 1
        data = [dense, categ_distance_class, categ_weekday_class, categ_if_busytime_class, categ_slice_id_class, categ_city_class, \
            categ_day_before2_type_class, categ_day_before1_type_class, categ_day_type_class, categ_day_after1_type_class, categ_day_after2_type_class, \
            logistic, train_rnn]
        if args.is_distributed:
            data, target = dict2device(data, device), target.to(device)
        with torch.no_grad():
            mape, loss, my_mape, satsf, pred = model(data, target)
        pbar.update(1)
        predict_results.append(pred.tolist())
    predict = np.concatenate(predict_results, axis=0)
    test_result = pd.DataFrame(predict, columns=['result'])
    df2 = pd.read_csv(test_dataroot)
    df2 = df2[:len(predict)]
    test_result['slice_id'] = df2['slice_id']
    test_result['simple_eta'] = df2['simple_eta']
    test_result['distance'] = df2['distance']
    test_result['ata'] = df2['ata']
    test_result['status_rate'] = 1 - df2['zsl_link_current_status0_countrates']
    test_result['predict'] =test_result['result']
    test_result['result'] =[math.ceil(x) for x in test_result['result'] / 60]
    test_result['simple_eta'] = [math.ceil(x) for x in test_result['simple_eta'] / 60]
    if not args.qeta:
        test_result['ata'] =[math.ceil(x) for x in test_result['ata']/60]
    simple_eta_satsf_score, predict_satsf_score, df = satisfaction_score_calc(test_result)
    df.to_csv('result.csv', index=None)
    calc_indices(test_result)
    pbar.close()
    pred_end_time = time.time()
    print("test data set predict and calc the indices consume time: ", int(pred_end_time - pred_start_time) / 60, 'min', " data_count:", len_results)

# every process works for doing the model train.
def main_workers(local_rank, ngpus_per_node, args):
    args.local_rank = local_rank
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    else:
        args.rank = args.rank * ngpus_per_node + args.local_rank
    # set rank backend and port for communications between rank
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    print('rank and local_rank', args.rank, args.local_rank, ' use multi-gpus...')
    # Init the dataset and dataloader
    train_dataset, train_loader, sampler = get_dataloader(args, data_type='train')
    val_dataset, val_loader, _ = get_dataloader(args, data_type='val')
    basic_info, wide_config = train_dataset.generate_config()
    model_config = args.__dict__.copy()
    model_config["wide"] = wide_config
    model_config["link_num"] = basic_info["link_num"]
    # Init the wdc model
    model = wdc(model_config)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.local_rank)
    # Init for load pretrain model, finetune params
    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path)
        model.load_state_dict(checkpoint['state_dict'])
        # only finetune the last three mlp layers
        for params in model.wide.parameters():
            params.requires_grad = False
        for params in model.deep.parameters():
            params.requires_grad = False
        for params in model.cnn_rnn.parameters():
            params.requires_grad = False
    # Start the ddp and train the model.
    model = model.to("cuda:{}".format(args.local_rank))
    ddp_model = ddp(model, find_unused_parameters=False, device_ids=[args.local_rank])
    optimizer, scheduler = model.config_optimizers(ddp_model, args)
    train_wdc(train_loader, val_loader, ddp_model, optimizer, scheduler, sampler, args)

def main():
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    if args.evaluate:
        data_type = 'test'
        test_dataset, data_loader, _ = get_dataloader(args, data_type='test')
        basic_info, wide_config = test_dataset.generate_config()
        model_config = args.__dict__.copy()
        model_config["wide"] = wide_config
        model_config["link_num"] = basic_info["link_num"]
        model = wdc(model_config)
        # test file path and name.
        if test_dataset.deepfm_file_path.endswith(".csv"):
            test_dataroot = test_dataset.deepfm_file_path
        elif args.qeta:
            test_dataroot = test_dataset.deepfm_file_path + data_type + '.csv'
        else:
            test_dataroot = test_dataset.deepfm_file_path + str(test_dataset.file_start_index) + '.csv'
        # It works for predicting many models but only load the predict data once
        if os.path.isdir(args.resume):
            if "finetune" in args.resume:
                model_names = glob.glob(os.path.join(args.resume, "*smape13*"))
                if len(model_names) == 0:
                   model_names = glob.glob(os.path.join(args.resume, "*smape12*"))
            else:
                model_names = glob.glob(os.path.join(args.resume, "*"))
        else:
            model_names = args.resume.split('#')
        print("model_names length: ", len(model_names))
        for model_name in model_names:
            try:
                checkpoint = torch.load(model_name, map_location="cpu")
                print("predict model_name:", model_name)
                model.load_state_dict(checkpoint['state_dict'])
                device = torch.device("cpu")
                model = model.to(device)
                evaluation(data_loader, model, args, device, test_dataroot)
            except:
                traceback.print_exc()
        return

    if args.seed is not None:
        set_seed(args.seed)

    if args.is_distributed:
    
        ngpus_per_node = torch.cuda.device_count()
        print("ngpus_per_node: ", ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.nodes
        mp.spawn(main_workers, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train_data_type = 'train'
        val_data_type = 'val'
        train_dataset, train_loader, sampler = get_dataloader(args, data_type=train_data_type)
        val_dataset, val_loader, _ = get_dataloader(args, data_type=val_data_type)
        basic_info, wide_config = train_dataset.generate_config()
        model_config = args.__dict__.copy()
        model_config["wide"] = wide_config
        model_config["link_num"] = basic_info["link_num"]
        model = wdc(model_config)
        optimizer, scheduler = model.config_optimizers(model, args)
        train_wdc(train_loader, val_loader, model, optimizer, scheduler, sampler, args)
    end_time = time.time()
    print("all over time consume:", int(end_time - start_time) / 60, 'min')

if __name__ == "__main__":
    main()
