import itertools

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt
import torch

# use nvidia gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        # Use sigmoid activations for f and g.                              #
        encoding = torch.sigmoid(self.g(inputs))
        decoding = torch.sigmoid(self.h(encoding))
        return decoding


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    valid_acc_lst = []
    train_loss_lst = []

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0).to(device)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            # nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            # target[0][nan_mask] = output[0][nan_mask]

            # Mask target using pytorch ops so tensors moved onto gpu
            valid_entries_mask = ~torch.isnan(train_data[user_id].unsqueeze(0))
            target[~valid_entries_mask] = output[~valid_entries_mask]

            # compute regularized loss
            weight_norm = model.get_weight_norm()
            reg_loss = (lamb / 2) * weight_norm

            loss = torch.sum((output - target) ** 2.) + reg_loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        train_loss_lst.append(train_loss)
        valid_acc_lst.append(valid_acc)
    return train_loss_lst, valid_acc_lst


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    print(f"Using {device}")

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    zero_train_matrix = zero_train_matrix.to(device)
    train_matrix = train_matrix.to(device)

    # hyperparam sets (ignore lambda for now)
    hp_grid = {
        'k': [10, 50, 100, 200, 500],
        'lr': [0.001, 0.01, 0.1],
        'num_epoch': [50, 100, 200, 400],
        'lamb': [0],
    }
    num_questions = train_matrix.shape[1]
    best_perf = float('inf')
    best_hp = None

    # get all hyperparameter combinations (k, lr, num_epoch)
    keys, values = zip(*hp_grid.items())
    hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # part (b) train autoencoder and tune hyperparams
    for hp in hp_combinations:
        model = AutoEncoder(num_questions, hp['k']).to(device)
        train(model, hp['lr'], hp['lamb'], train_matrix, zero_train_matrix,
              valid_data, hp['num_epoch'])
        perf = evaluate(model, zero_train_matrix, valid_data)
        if perf > best_perf:
            best_perf = perf
            best_hp = hp
        print(f"perf: {perf}, params: {hp}")
    print(f"best accuracy {best_perf} with params {best_hp}")

    # part (c) plot training and validation objectives with best hp
    model = AutoEncoder(num_questions, best_hp['k']).to(device)
    train(model, best_hp['lr'], best_hp['lamb'], train_matrix,
          zero_train_matrix,
          valid_data, best_hp['num_epoch'])
    train_loss, valid_acc = train(model, best_hp['lr'], best_hp['lamb'],
                                  train_matrix, zero_train_matrix,
                                  valid_data, best_hp['num_epoch'])

    # plotting training loss wrt epoch
    epochs = range(1, best_hp['num_epoch'] + 1)
    plt.plot(epochs, train_loss, label='Training loss')
    plt.title('Training Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')

    # plotting validation accuracy wrt epoch
    epochs = range(1, best_hp['num_epoch'] + 1)
    plt.plot(epochs, valid_acc, label='Validation accuracy')
    plt.title('Validation Accuracy by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')

    # part (c): compute test accuracy
    model = AutoEncoder(num_questions, best_hp['k']).to(device)
    train(model, best_hp['lr'], best_hp['lamb'], train_matrix,
          zero_train_matrix,
          valid_data, best_hp['num_epoch'])
    test_perf = evaluate(model, zero_train_matrix, test_data)
    print(f"test accuracy {test_perf}")

    # part (d) evaluate model with lambda
    lamb_set = [0.001, 0.01, 0.1, 1, 1.1, 1.2, 1.5]
    new_hp = best_hp
    best_lamb_perf = float('inf')
    best_lamb = None
    for lamb in lamb_set:
        new_hp['lamb'] = lamb
        model = AutoEncoder(num_questions, new_hp['k']).to(device)
        train(model, new_hp['lr'], new_hp['lamb'], train_matrix,
              zero_train_matrix,
              valid_data, new_hp['num_epoch'])
        perf = evaluate(model, zero_train_matrix, valid_data)
        if perf > best_lamb_perf:
            best_lamb_perf = perf
            best_lamb = lamb
        print(f"perf: {perf}, lambda: {lamb}")
    print(f"best accuracy: {best_lamb_perf} with lambda: {best_lamb}")

    # part (d): compute test accuracy with regularizer
    best_hp['lamb'] = best_lamb
    model = AutoEncoder(num_questions, best_hp['k']).to(device)
    train(model, best_hp['lr'], best_hp['lamb'], train_matrix,
          zero_train_matrix,
          valid_data, best_hp['num_epoch'])
    test_perf = evaluate(model, zero_train_matrix, test_data)
    print(f"test accuracy {test_perf} with lambda: {best_lamb}")


if __name__ == "__main__":
    main()
