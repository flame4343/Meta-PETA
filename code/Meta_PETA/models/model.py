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

# Use GPU if available, else use CPU
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

class PETA(nn.Module):
    def __init__(self, opt):
        super(PETA, self).__init__()
        self.opt = opt

        # Embedding layers
        self.id_embedding = nn.Embedding(opt.car_num, opt.embedding_size)
        self.static_feature_transform = nn.Linear(opt.static_feature_num, opt.embedding_size, bias=False)
        self.static_layer_norm = nn.LayerNorm(normalized_shape=2 * opt.embedding_size)

        # Neural network layers
        self.feature_transform = nn.Linear(opt.target_feature_num, opt.hidden_size, bias=False)
        self.top_5_feature_transform = nn.Linear(opt.top_5_feature_num, opt.hidden_size, bias=False)

        self.fc0 = nn.Linear(2 * opt.embedding_size + opt.hidden_size, opt.hidden_size)
        self.fc1 = nn.Linear(opt.hidden_size, opt.hidden_size2)
        self.fc2 = nn.Linear(opt.hidden_size2, 1)

        # Multihead Self-Attention (MSA)
        self.multihead_self_attention = nn.MultiheadAttention(embed_dim=opt.target_feature_num, num_heads=opt.heads_num, batch_first=True)
        self.preference_learning_fc = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.preference_learning_layer_norm = nn.LayerNorm(normalized_shape=opt.hidden_size)

        # Multihead Cross-Attention (MCA)
        self.multihead_cross_attention = nn.MultiheadAttention(embed_dim=opt.hidden_size, num_heads=opt.heads_num, batch_first=True)
        self.behavior_prediction_fc = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.behavior_prediction_layer_norm = nn.LayerNorm(normalized_shape=opt.hidden_size)

        # Activation functions and gates
        self.relu = nn.ReLU()
        self.gate_layer = nn.Linear(2 * opt.hidden_size, opt.hidden_size, bias=False)

        # Loss function parameters
        self.delta = opt.Asymmetric_delta
        self.alpha = opt.Asymmetric_alpha

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with uniform distribution
        for weight in self.parameters():
            if len(weight.shape) == 2:  # Only for linear layers
                stdv = 1.0 / math.sqrt(weight.size(1))
                weight.data.uniform_(-stdv, stdv)

    def MAELoss(self, predictions, targets):
        # Mean Absolute Error (MAE) Loss
        absolute_error = torch.abs(predictions - targets)
        return torch.mean(absolute_error)

    def AsymmetricHuberLoss(self, predictions, targets):
        # Asymmetric Huber Loss function
        error = predictions - targets
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.full_like(abs_error, self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear

        weight_positive = self.alpha
        weight_negative = 1 - self.alpha

        # Apply different weights to positive and negative errors
        loss = torch.where(error > 0, loss * weight_positive, loss * weight_negative)
        return torch.mean(loss)

    def min_max_normalize(self, tensor):
        # Min-max normalization for tensor
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)

    def forward(self, id, feature, static, top_5, target):
        # Forward pass
        id_embedding = self.id_embedding(id)
        static_transformed = self.static_feature_transform(static)
        static_combined = torch.cat([id_embedding, static_transformed], dim=1)
        static_combined = self.static_layer_norm(static_combined)

        feature_transformed = self.feature_transform(feature)
        top_5_transformed = self.top_5_feature_transform(top_5)

        # Multihead Self-Attention (MSA) for preference learning
        query, key, value = top_5_transformed, top_5_transformed, top_5_transformed
        preference_learning, _ = self.multihead_self_attention(query, key, value)
        preference_learning = preference_learning + top_5_transformed
        preference_learning_fc = self.relu(self.preference_learning_fc(preference_learning))
        preference_learning_combined = preference_learning_fc + preference_learning
        preference_learning_normalized = self.preference_learning_layer_norm(preference_learning_combined)

        # Multihead Cross-Attention (MCA) for behavior prediction
        query = feature_transformed.unsqueeze(1)
        key, value = preference_learning_normalized, preference_learning_normalized
        behavior_prediction, _ = self.multihead_cross_attention(query, key, value)
        behavior_prediction = behavior_prediction + query
        behavior_prediction_fc = self.relu(self.behavior_prediction_fc(behavior_prediction))
        behavior_prediction_combined = behavior_prediction_fc + behavior_prediction
        behavior_prediction_normalized = self.behavior_prediction_layer_norm(behavior_prediction_combined)
        behavior_prediction_normalized = behavior_prediction_normalized.squeeze(1)

        # Final predictions
        combined_features = torch.cat([static_combined, behavior_prediction_normalized], dim=1)
        fc0_output = self.relu(self.fc0(combined_features))
        gate_output = torch.relu(self.gate_layer(torch.cat([feature_transformed, fc0_output], dim=-1)))
        combined_output = feature_transformed * gate_output + fc0_output * (1 - gate_output)

        fc1_output = self.relu(self.fc1(combined_output))
        predictions = self.relu(self.fc2(fc1_output)).squeeze(1)

        # Calculate loss
        loss = self.AsymmetricHuberLoss(predictions, target)

        return loss, target, predictions

class ETAMetaLearner(object):
    def __init__(self, train_data, opt):
        super(ETAMetaLearner).__init__()
        # Initialize model and hyperparameters
        self.model = trans_to_cuda(PETA(opt))
        self.opt = opt

        # Use MAML (Model-Agnostic Meta-Learning) with Learn2Learn
        self.maml = l2l.algorithms.MAML(self.model, lr=opt.lr, allow_unused=True, first_order=True)

        # Optimizer for meta-learner
        self.optimizer = torch.optim.Adam(self.maml.parameters(), lr=opt.meta_lr, weight_decay=opt.l2)

        # Task sampler to generate batches of tasks
        self.task_sampler = TaskBatchGenerator(train_data=train_data)

        # Maximum number of task data points
        self.max_task_data_num = opt.max_task_data_num

    def train_tasks(self):
        # Train model on sampled tasks
        sampler = self.task_sampler.getTaskBatch()  # Generate a batch of tasks
        for task_batch in tqdm(sampler):  # Iterate through batches
            self.optimizer.zero_grad()  # Reset gradients
            task_count = 0
            total_loss = 0.0

            # Iterate through tasks in the batch
            for task_data in task_batch:
                support_data, query_data = task_data
                learner = self.maml.clone()  # Clone the MAML model
                learner.train()

                for _ in range(1):  # Training loop (1 epoch)
                    task_loss = self.train_valid_base_model(learner, support_data, query_data)
                    total_loss += task_loss
                    task_count += 1

            # Calculate average loss for all tasks
            total_loss /= task_count
            total_loss.backward()  # Backpropagation
            self.optimizer.step()  # Update meta-learner parameters

    def train_valid_base_model(self, learner, support_data, query_data):
        # Train and validate the base model on support and query data
        support_loader = DataLoader(
            support_data,
            num_workers=0,
            batch_size=self.opt.batch_size,
            shuffle=True,
            pin_memory=False,
        )

        # Adapt the model using support set
        for _, batch_data in enumerate(support_loader):
            loss, targets, scores = forward(learner, batch_data)
            learner.adapt(loss)  # Perform model adaptation using loss

        # Validate the adapted model on the query set
        val_loss = self.validation(learner, query_data)
        return val_loss

    def validation(self, learner, query_data):
        # Validate the adapted model using query data
        query_loader = DataLoader(
            query_data,
            num_workers=0,
            batch_size=self.opt.batch_size,
            shuffle=True,
            pin_memory=False,
        )
        data_count = len(query_data)
        total_loss = 0.0

        # Evaluate the model on query data
        for batch_data in query_loader:
            loss, targets, scores = forward(learner, batch_data)
            # Adjust loss for the number of query data points
            loss *= data_count / self.max_task_data_num
            total_loss += loss

        return total_loss

# Utility function to move tensors to CUDA if available
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable

# Forward pass through the model
def forward(model, data):
    # Unpack the input data
    id_tensor, feature_tensor, static_tensor, top_5_tensor, target_tensor = data

    # Move data to GPU if available and convert to the appropriate data types
    id_tensor = trans_to_cuda(id_tensor).long()
    feature_tensor = trans_to_cuda(feature_tensor).float()
    static_tensor = trans_to_cuda(static_tensor).float()
    top_5_tensor = trans_to_cuda(top_5_tensor).float()
    target_tensor = trans_to_cuda(target_tensor).float()

    # Pass the data through the model
    loss, target, prediction = model(id_tensor, feature_tensor, static_tensor, top_5_tensor, target_tensor)
    return loss, target, prediction

# Train and test the model
def train_test(model, opt, train_data, test_data, is_meta_learning=False):
    print('Start training: ', datetime.datetime.now())
    model.train()  # Set model to training mode
    total_loss = 0.0

    # Create a DataLoader for the training data
    train_loader = DataLoader(
        train_data,
        num_workers=0 if not is_meta_learning else 8,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=False,
    )

    # Train the model on the training set
    for _, batch_data in enumerate(tqdm(train_loader)):
        model.optimizer.zero_grad()  # Reset gradients
        loss, targets, predictions = forward(model, batch_data)
        loss.backward()  # Backpropagation
        model.optimizer.step()  # Update model parameters
        total_loss += float(loss.item())  # Accumulate total loss

    print('\tLoss:\t%.3f' % total_loss)

    # Test the model on the test set
    mse, mae, mape = test(model, opt, test_data, not is_meta_learning)

    return mse, mae, mape
def test(model, opt, test_data, isTestset=True):
    # Determine number of workers for DataLoader
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
