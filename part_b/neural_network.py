from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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

        # input->g->h->f
        # apply sigmoid to the linear function g with input as inputs, the output is h
        h = F.sigmoid(self.g(inputs))
        # apply sigmoid to the linear function h with input as h, the output is f
        f = F.sigmoid(self.h(h))

        out = f
        return out


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

    #####################################################################
    # Additional code for plotting:
    # Define lists to store the training and validation objectives.
    train_accuracy = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []
    # load train data
    train_data_2 = load_train_csv('../data')


    #####################################################################

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # get weight norm
            weight_norm = model.get_weight_norm()
            regularizer_loss = (lamb / 2) * weight_norm

            loss = torch.sum((output - target) ** 2.) + regularizer_loss # add regularizer
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        #####################################################################
        # Additional code for plotting:
        train_losses.append((epoch, train_loss))
        # Evaluate the model on the training and validation set.
        train_acc = evaluate(model, zero_train_data, train_data_2)
        train_accuracy.append((epoch, train_acc))
        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_accuracies.append((epoch, valid_acc))
        #####################################################################
        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        
        # additional return values for plotting
    return valid_losses, train_losses, train_accuracy, valid_accuracies
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


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
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # C)
    # Iterate over different hyperparameters and train the model.
    # Set model hyperparameters.
    # ks = [10, 50, 100, 200, 500]

    # # Set optimization hyperparameters.
    # lrs = [0.001, 0.005, 0.01, 0.02] 
    # nums_epoch = [20, 40, 80, 100]
    # lamb = 0 # not used for now
    
    # current_accuracy = 0
    # best_k = 0
    # best_epo = 0
    # best_lr = 0

    # for k in ks:
    #     for lr in lrs:
    #         for num_epoch in nums_epoch:
    #             model = AutoEncoder(train_matrix.shape[1], k=k)
    #             # model.to(device)
    #             train(model, lr, lamb, train_matrix, zero_train_matrix,
    #                 valid_data, num_epoch)
    #             accuracy = evaluate(model, zero_train_matrix, valid_data)
    #             print(f'k={k}, lr={lr}, num_epoch={num_epoch}, accuracy={accuracy}')
    #             if accuracy > current_accuracy:
    #                 current_accuracy = accuracy
    #                 best_k = k
    #                 best_epo = num_epoch
    #                 best_lr = lr
    
    # print(f'best k={best_k}, best lr={best_lr}, best num_epoch={best_epo}')

    print("Best k: 10, Best lr: 0.01, Best num_epoch: 100 (Part C)")
    # print("Running the model with the best hyperparameters...")

    # # Choose k = 10, lr = 0.01, num_epoch = 100
    # k = 10
    # lr = 0.01
    # num_epoch = 100
    # lamb = 0
    # model = AutoEncoder(train_matrix.shape[1], k=k)
    # valid_losses, train_losses, train_accuracy, valid_accuracies = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # score_with_validation = evaluate(model, zero_train_matrix, valid_data)
    # test_acc = evaluate(model, zero_train_matrix, test_data)
    # print("------------------------------------")
    # print("Results:")
    # print("Test Accuracy: ", test_acc)
    # print("Validation Accuracy: ", score_with_validation)
    # plot the training and validation objectives as a function of epoch
    # 1: train_losses vs epoch: train_losses is a list of tuples (epoch, train_loss)
    # plot_data = np.array(train_losses)
    # epochs, losses = zip(*plot_data)
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
    # plt.title('Training Losses vs Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.grid(True)
    # plt.xticks(epochs)  # Ensure all epochs are shown
    # plt.tight_layout()
    # plt.show()

    # # 2: train_accuracy vs epoch: train_accuracy is a list of tuples (epoch, train_acc)
    # plot_data = np.array(train_accuracy)
    # epochs, acc = zip(*plot_data)
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, acc, marker='o', linestyle='-', color='b')
    # plt.title('Training Accuracy vs Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Accuracy')
    # plt.grid(True)
    # plt.xticks(epochs)  # Ensure all epochs are shown
    # plt.tight_layout()
    # plt.show()

    # # 3: valid_accuracy vs epoch: valid_accuracies is a list of tuples (epoch, valid_acc)
    # plot_data = np.array(valid_accuracies)
    # epochs, acc = zip(*plot_data)
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, acc, marker='o', linestyle='-', color='b')
    # plt.title('Validation Accuracy vs Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Accuracy')
    # plt.grid(True)
    # plt.xticks(epochs)  # Ensure all epochs are shown
    # plt.tight_layout()
    # plt.show()

    #####################################################################
    #                       Results (Without L^2):                      #
    #                Test Accuracy:  0.6923511148744003                 #
    #            Validation Accuracy:  0.6860005644933672               #   
    #####################################################################

    #####################################################################
    #                    Part E Tuning                                  # 
    #                   Best lambda:  0.001                             #                                     
    #####################################################################
    print("Best k: 10, Best lr: 0.01, Best num_epoch: 100, Best lambda: 0.001 (Part E)")
    print("Running the model with the best lambda = 0.001...")
    # Choose k = 10, lr = 0.01, num_epoch = 100
    k = 10
    lr = 0.01
    num_epoch = 100
    lamb = 0.001
    model = AutoEncoder(train_matrix.shape[1], k=k)
    valid_losses, train_losses, train_accuracy, valid_accuracies = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    valid_acc= evaluate(model, zero_train_matrix, valid_data)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    # current_accuracy = 0
    # best_lamb = 0
    # for l in lamb:
    #     model = AutoEncoder(train_matrix.shape[1], k=k)
    #     train(model, lr, l, train_matrix, zero_train_matrix, valid_data, num_epoch)
    #     accuracy = evaluate(model, zero_train_matrix, valid_data)
    #     print(f'lamb={l}, accuracy={accuracy}')
    #     if accuracy > current_accuracy:
    #         current_accuracy = accuracy
    #         best_lamb = l
    print("------------------------------------")
    print("Results:")
    print("Test Accuracy: ", test_acc)
    print("Validation Accuracy: ", valid_acc)

    #####################################################################
    #                       Results (With L^2):                         #
    #                Test Accuracy:  0.6833192209991532                 #
    #            Validation Accuracy:  0.687694044594976                #   
    #####################################################################











    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
