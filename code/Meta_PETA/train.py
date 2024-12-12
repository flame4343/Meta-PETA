import torch
import time
import argparse
from utils.random import *
from datasets.eta_dataset import *
from models.model import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ETA', help='ETA')
parser.add_argument('--model_path', default='best_model', help='ETA')
parser.add_argument('--car_num', type=int, default=10000)
parser.add_argument('--static_feature_num', type=int, default=6)
parser.add_argument('--embedding_size', type=int, default=10)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--hidden_size2', type=int, default=32)
parser.add_argument('--target_feature_num', type=int, default=128)
parser.add_argument('--top_5_feature_num', type=int, default=129)
parser.add_argument('--heads_num', type=int, default=4)
parser.add_argument('--Asymmetric_delta', type=int, default=1)
parser.add_argument('--Asymmetric_alpha', type=int, default=0.5)
parser.add_argument('--l2', type=float, default=1e-4, help='l2 penalty')
parser.add_argument('--meta_lr', type=float, default=0.001, help='lr')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--max_task_data_num', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--fine_tune_patience', type=int, default=10)



def ETA_meta_learning(ETAmaml, valid_data):
    # Meta learning
    print('-------------------------------------------------------')
    print("Meta-Learning...")
    start = time.time()
    best_result = [99999, 99999, 99999]
    best_epoch = [0, 0, 0]

    bad_counter = 0
    for epoch in range(50):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        print('start training: ', datetime.datetime.now())
        ETAmaml.train_tasks()
        MSE, MAE, MAPE = test(ETAmaml.maml, opt, valid_data, isTestset=False)
        RMSE = math.sqrt(MSE)
        flag = 0
        if RMSE <= best_result[0]:
            best_result[0] = RMSE
            best_epoch[0] = epoch
            flag = 1
        if MAE <= best_result[1]:
            best_result[1] = MAE
            best_epoch[1] = epoch
            flag = 1
        if MAPE <= best_result[2]:
            best_result[2] = MAPE
            best_epoch[2] = epoch
            flag = 1
            torch.save(ETAmaml.model.state_dict(), f"{opt.model_path}/{opt.dataset}.pth")
        print('Current Result:')
        print('\tRMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' % (math.sqrt(MSE), MAE, MAPE))
        print('Best Result:')
        print('\tRMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f\tEpoch:\t%d, %d, %d' %
              (best_result[0], best_result[1], best_result[2], best_epoch[0], best_epoch[1], best_epoch[2]))
        if flag != 1:
            bad_counter += 1
            if bad_counter >= 5:
                break
        else:
            bad_counter = 0
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

def fine_tuning_user(train_data, valid_data, test_data):
    model = PETA(opt)
    model.load_state_dict(torch.load(f"{opt.model_path}/{opt.dataset}.pth"))
    model = trans_to_cuda(model)

    start = time.time()
    best_result = [99999, 99999, 99999]
    best_epoch = [0, 0, 0]
    bad_counter = 0
    test_score = [0, 0, 0]


    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        MSE, MAE, MAPE = train_test(model, opt, train_data, valid_data)
        # RMSE = math.sqrt(MSE)
        flag_mape = 0
        flag_patience = 0
        if MSE <= best_result[0]:
            best_result[0] = MSE
            best_epoch[0] = epoch
            flag_patience = 1
        if MAE <= best_result[1]:
            best_result[1] = MAE
            best_epoch[1] = epoch
            flag_patience = 1
        if MAPE <= best_result[2]:
            best_result[2] = MAPE
            best_epoch[2] = epoch
            flag_mape = 1
            flag_patience = 1
        print('Current Result:')
        print('\tRMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' % (MSE, MAE, MAPE))
        print('Best Result:')
        print('\tRMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f\tEpoch:\t%d, %d, %d' %
              (best_result[0], best_result[1], best_result[2], best_epoch[0], best_epoch[1], best_epoch[2]))
        if flag_patience != 1:
            bad_counter += 1
            if bad_counter >= opt.fine_tune_patience:
                break
        else:
            if flag_mape == 1:
                t_MSE, t_MAE, t_MAPE = test(model, opt, test_data)
                test_score[0], test_score[1], test_score[2] = t_MSE, t_MAE, t_MAPE
                print('Test Result:')
                print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' % (test_score[0], test_score[1], test_score[2]))
    print('-------------------------------------------------------')
    print('Test Result:')
    print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' % (test_score[0], test_score[1], test_score[2]))
    end = time.time()
    print("Run time: %f s" % (end - start))
    # print(test_score)
    # quit()
    return test_score




opt = parser.parse_args()

def main():

    init_seed(42)
    train_data = load_json(f"./data/{opt.dataset}/train_meta.json")
    valid_data = load_json(f"./data/{opt.dataset}/valid_meta.json")


    for i in range(len(train_data)):
        train_data[i][0] = ETADataset(train_data[i][0], opt)
        train_data[i][1] = ETADataset(train_data[i][1], opt)
    valid_data = ETADataset(valid_data, opt)


    train_data_fine_tune = load_json(f"./data/{opt.dataset}/train_fine_tune.json")
    test_data = load_json(f"./data/{opt.dataset}/test_meta.json")

    for u in train_data_fine_tune:
        train_data_fine_tune[u][0] = ETADataset(train_data_fine_tune[u][0], opt)
        train_data_fine_tune[u][1] = ETADataset(train_data_fine_tune[u][1], opt)
    for u in test_data:
        test_data[u] = ETADataset(test_data[u], opt)

    all_user_scores = [[], [], []]
    for u in test_data:
        print('-------------------------------------------------------')
        print(f"Fine tuning on user {u}...")
        train_data, valid_data = train_data_fine_tune[u]
        user_test_data = test_data[u]
        user_test_score = fine_tuning_user(train_data, valid_data, user_test_data)
        for i in range(len(all_user_scores)):
            all_user_scores[i].append(user_test_score[i])

    print('-------------------------------------------------------')
    print('All User Test Result:')
    print('\tRMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' % (math.sqrt(np.mean(all_user_scores[0])), np.mean(all_user_scores[1]), np.mean(all_user_scores[2])))


if __name__ == "__main__":
    main()
