import torch
import time
import argparse
import math
import datetime
import numpy as np
from utils.random import *
from datasets.eta_dataset import *
from models.model import *

# Argument parser for configuring the model
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ETA', help='Dataset name')
parser.add_argument('--model_path', default='best_model', help='Path to save the trained model')
parser.add_argument('--car_num', type=int, default=10000, help='Number of cars in the dataset')
parser.add_argument('--static_feature_num', type=int, default=6, help='Number of static features')
parser.add_argument('--embedding_size', type=int, default=10, help='Embedding size for features')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size for the model')
parser.add_argument('--hidden_size2', type=int, default=32, help='Secondary hidden layer size')
parser.add_argument('--target_feature_num', type=int, default=128, help='Number of target features')
parser.add_argument('--top_5_feature_num', type=int, default=129, help='Number of top 5 features')
parser.add_argument('--heads_num', type=int, default=4, help='Number of attention heads')
parser.add_argument('--Asymmetric_delta', type=int, default=1, help='Asymmetric delta value')
parser.add_argument('--Asymmetric_alpha', type=float, default=0.5, help='Asymmetric alpha value')
parser.add_argument('--l2', type=float, default=1e-4, help='L2 regularization penalty')
parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the model')
parser.add_argument('--max_task_data_num', type=int, default=50, help='Maximum number of tasks for meta learning')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--epoch', type=int, default=5, help='Number of fine-tuning epochs')
parser.add_argument('--fine_tune_patience', type=int, default=10, help='Patience for fine-tuning early stopping')

# Meta-learning function
def ETA_meta_learning(ETAmaml, valid_data):
    """
    Perform meta-learning to train the model using training tasks and validation data.
    """
    print('-------------------------------------------------------')
    print("Meta-Learning...")
    start = time.time()

    # Best results trackers
    best_result = [99999, 99999, 99999]  # RMSE, MAE, MAPE
    best_epoch = [0, 0, 0]  # Epochs corresponding to the best results
    bad_counter = 0

    for epoch in range(50):
        print('-------------------------------------------------------')
        print(f'Epoch: {epoch}')
        print(f'Start training: {datetime.datetime.now()}')

        ETAmaml.train_tasks()  # Train tasks using meta-learning
        MSE, MAE, MAPE = test(ETAmaml.maml, opt, valid_data, isTestset=False)  # Evaluate the model
        RMSE = math.sqrt(MSE)

        # Update best results
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

            # Save the best model
            torch.save(ETAmaml.model.state_dict(), f"{opt.model_path}/{opt.dataset}.pth")

        print('Current Results:')
        print(f'\tRMSE:\t{RMSE:.4f}\tMAE:\t{MAE:.4f}\tMAPE:\t{MAPE:.4f}')
        print('Best Results:')
        print(f'\tRMSE:\t{best_result[0]:.4f}\tMAE:\t{best_result[1]:.4f}\tMAPE:\t{best_result[2]:.4f}\tEpoch:\t{best_epoch[0]}, {best_epoch[1]}, {best_epoch[2]}')

        if flag != 1:
            bad_counter += 1
            if bad_counter >= 5:  # Early stopping
                break
        else:
            bad_counter = 0

    print('-------------------------------------------------------')
    end = time.time()
    print(f"Meta-learning runtime: {end - start:.2f} seconds")

# Fine-tuning function
def fine_tuning_user(train_data, valid_data, test_data):
    """
    Fine-tune the model for a specific user using their training, validation, and test data.
    """
    model = PETA(opt)
    model.load_state_dict(torch.load(f"{opt.model_path}/{opt.dataset}.pth"))  # Load the pre-trained model
    model = trans_to_cuda(model)

    start = time.time()
    best_result = [99999, 99999, 99999]
    best_epoch = [0, 0, 0]
    bad_counter = 0
    test_score = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print(f'Epoch: {epoch}')

        MSE, MAE, MAPE = train_test(model, opt, train_data, valid_data)  # Train and validate the model

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

        print('Current Results:')
        print(f'\tMSE:\t{MSE:.4f}\tMAE:\t{MAE:.4f}\tMAPE:\t{MAPE:.4f}')
        print('Best Results:')
        print(f'\tMSE:\t{best_result[0]:.4f}\tMAE:\t{best_result[1]:.4f}\tMAPE:\t{best_result[2]:.4f}\tEpoch:\t{best_epoch[0]}, {best_epoch[1]}, {best_epoch[2]}')

        if flag_patience != 1:
            bad_counter += 1
            if bad_counter >= opt.fine_tune_patience:  # Early stopping
                break
        else:
            if flag_mape == 1:
                t_MSE, t_MAE, t_MAPE = test(model, opt, test_data)  # Test the model on user test data
                test_score[0], test_score[1], test_score[2] = t_MSE, t_MAE, t_MAPE
                print('Test Results:')
                print(f'\tMSE:\t{t_MSE:.4f}\tMAE:\t{t_MAE:.4f}\tMAPE:\t{t_MAPE:.4f}')

    print('-------------------------------------------------------')
    print('Final Test Results:')
    print(f'\tMSE:\t{test_score[0]:.4f}\tMAE:\t{test_score[1]:.4f}\tMAPE:\t{test_score[2]:.4f}')
    end = time.time()
    print(f"Fine-tuning runtime: {end - start:.2f} seconds")
    return test_score

# Main function
def main():
    """
    Main function to orchestrate meta-learning and fine-tuning processes.
    """
    init_seed(42)  # Set random seed for reproducibility

    # Load training and validation data for meta-learning
    train_data = load_json(f"./data/{opt.dataset}/train_meta.json")
    valid_data = load_json(f"./data/{opt.dataset}/valid_meta.json")

    # Prepare datasets
    for i in range(len(train_data)):
        train_data[i][0] = ETADataset(train_data[i][0], opt)
        train_data[i][1] = ETADataset(train_data[i][1], opt)
    valid_data = ETADataset(valid_data, opt)

    # Initialize meta-learner and perform meta-learning
    maml = ETAMetaLearner(train_data, opt)
    ETA_meta_learning(maml, valid_data)

    # Load fine-tuning and test data
    train_data_fine_tune = load_json(f"./data/{opt.dataset}/train_fine_tune.json")
    test_data = load_json(f"./data/{opt.dataset}/test_meta.json")

    for u in train_data_fine_tune:
        train_data_fine_tune[u][0] = ETADataset(train_data_fine_tune[u][0], opt)
        train_data_fine_tune[u][1] = ETADataset(train_data_fine_tune[u][1], opt)
    for u in test_data:
        test_data[u] = ETADataset(test_data[u], opt)

    # Fine-tuning for each user
    all_user_scores = [[], [], []]
    for u in test_data:
        print('-------------------------------------------------------')
        print(f"Fine-tuning on user {u}...")
        train_data, valid_data = train_data_fine_tune[u]
        user_test_data = test_data[u]
        user_test_score = fine_tuning_user(train_data, valid_data, user_test_data)
        for i in range(len(all_user_scores)):
            all_user_scores[i].append(user_test_score[i])

    # Report overall results
    print('-------------------------------------------------------')
    print('All User Test Results:')
    print(f'\tRMSE:\t{math.sqrt(np.mean(all_user_scores[0])):.4f}\tMAE:\t{np.mean(all_user_scores[1]):.4f}\tMAPE:\t{np.mean(all_user_scores[2]):.4f}')

if __name__ == "__main__":
    opt = parser.parse_args()
    main()
