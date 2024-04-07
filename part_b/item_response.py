from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    # get data from dictionary
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    # calculate log likelihood

    for i in range(len(user_id)):
        u = user_id[i]
        q = question_id[i]
        cij = is_correct[i]
        x = (theta[u] - beta[q])
        log_lklihood += cij * x - np.log(1.0 + np.exp(x))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # get data from dictionary
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]

    # update theta and beta
    for i in range(len(user_id)):
        u = user_id[i]
        q = question_id[i]
        cij = is_correct[i]
        x = (theta[u] - beta[q])
        # cij - sigmoid(x) is the gradient with respect to theta
        theta[u] += lr * (cij - sigmoid(x))
        # -cij + sigmoid(x) is the gradient with respect to beta
        beta[q] += lr * (sigmoid(x) - cij)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(np.max(data["user_id"]) + 1)
    beta = np.zeros(np.max(data["question_id"]) + 1)

    val_acc_lst = []
    train_likelihood = []
    val_likelihood = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)

        train_likelihood.append(neg_lld)
        val_likelihood.append(neg_log_likelihood(val_data, theta=theta, beta=beta))

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_likelihood, val_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    import matplotlib.pyplot as plt

    iterations = 30
    lr = 0.018
    theta, beta, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, lr, iterations)

    # 30, 0.02 -> test acc = 0.7078, val acc = 0.7073
    # 50, 0.008 -> Test accuracy:  0.7044877222692634 Validation accuracy:  0.7054755856618685
    # 40, 0.01 -> Test accuracy:  0.7044877222692634 Validation accuracy:  0.7063223257126728
    # 35, 0.01: Test accuracy:  0.7047699689528648 Validation accuracy:  0.7067456957380751
    # 30, 0.01: Test accuracy:  0.7050522156364663 Validation accuracy:  0.7070279424216765

    """     # find the best hyperparameters
    best_val_acc = 0
    best_lr = 0
    best_iterations = 0
    for lr in np.linspace(0.01, 0.03, 10):
        for iterations in range(20, 40, 5):
            theta, beta, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, lr, iterations)
            test_acc = evaluate(test_data, theta, beta)
            val_acc = evaluate(val_data, theta, beta)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_iterations = iterations

    print("Best validation accuracy: ", best_val_acc)
    print("Best learning rate: ", best_lr)
    print("Best number of iterations: ", best_iterations)

    Best validation accuracy:  0.707874682472481
    Best learning rate:  0.018888888888888886
    Best number of iterations:  30 """




    # visualize the hypertuning process:
    """ iteration_lable = np.arange(1, iterations + 1)
    plt.plot(iteration_lable, val_acc_lst, label='validation accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.title('validation accuracy vs iteration')
    plt.legend()
    plt.show()

    # visualize the likelihood for training and validation data seperately
    plt.plot(iteration_lable, train_likelihood, label='training likelihood')
    plt.xlabel('iteration')
    plt.ylabel('likelihood')
    plt.title('training likelihood vs iteration')
    plt.legend()
    plt.show()

    plt.plot(iteration_lable, val_likelihood, label='validation likelihood')
    plt.xlabel('iteration')
    plt.ylabel('likelihood')
    plt.title('validation likelihood vs iteration')
    plt.legend()
    plt.show() """ 

    # evaluate the model on test data
    test_acc = evaluate(test_data, theta, beta)
    # evaluate the model on validation data
    val_acc = evaluate(val_data, theta, beta)
    # print both test and validation accuracy
    print("Test accuracy: ", test_acc, "\n")
    print("Validation accuracy: ", val_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    
    # choose j1, j2 ,j3
    j1 = 1      # the first question
    j2 = 1000    
    j3 = 1500   # the 1500th question

    # calculate the probability of correct answer for each user
    theta = np.sort(theta)
    prob_j1 = sigmoid(theta - beta[j1])
    prob_j2 = sigmoid(theta - beta[j2])
    prob_j3 = sigmoid(theta - beta[j3])

    # print(beta[j1], beta[j2], beta[j3])

    plt.plot(theta, prob_j1, color='red', label='question 1 (1, J1)')
    plt.plot(theta, prob_j2, color='green', label='question 2 (1000, J2)')
    plt.plot(theta, prob_j3, color='blue', label='question 3 (1500, J3)')
    plt.xlabel('theta')
    plt.ylabel('Probability of correct answer')
    plt.title('Probability of correct answer vs theta')
    plt.legend()
    plt.show()


    



    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
