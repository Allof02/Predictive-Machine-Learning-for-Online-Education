from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))



def neg_log_likelihood(data, theta, beta, alpha, gamma):
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
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']

    for i, q in enumerate(user_id):
        theta_i = theta[q] 
        beta_j = beta[question_id[i]] 
        gamma_j = gamma[question_id[i]]
        alpha_j = alpha[question_id[i]]
        cij = is_correct[i]

        if gamma_j <= 0:
            gamma_j = 0
        
        else: 
            gamma_j = 0.99999

        exp = np.exp(alpha_j * (theta_i - beta_j))
        #print(gamma_j)
        log_lklihood += cij * np.log(gamma_j + exp) + (1 - cij) * np.log(1 - gamma_j) - np.log(1 + exp) # 3.6.1 in the report
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, gamma, alpha):
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
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']

    for i, q in enumerate(user_id):
        theta_i = theta[q]
        beta_j = beta[question_id[i]]
        gamma_j = gamma[question_id[i]]
        alpha_j = alpha[question_id[i]]
        alpha_beta_theta = alpha_j * (theta_i - beta_j)
        exp = np.exp(alpha_beta_theta)
        sigmoid_with_A_T_B = sigmoid(alpha_j * (theta_i - beta_j))
        sigmoid2 = exp / (gamma_j + exp)
        cij = is_correct[i]

        theta[q] += lr * (cij * alpha_j * exp /(gamma_j + exp)  - alpha_j * sigmoid_with_A_T_B) # 3.6.2 in the report
        beta[question_id[i]] += lr * (-alpha_j * cij * sigmoid2 + alpha_j * sigmoid_with_A_T_B) # 3.6.3 in the report
        alpha[question_id[i]] += lr * (cij * (theta_i - beta_j) * sigmoid2 - (theta_i - beta_j) * sigmoid_with_A_T_B) # 3.6.4 in the report
        #gamma[question_id[i]] += lr * (cij / (gamma_j + exp) - (1 - cij) / (1 - gamma_j)) # 3.6.5 in the report
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha, gamma



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
    theta = np.zeros(max(data["user_id"]) + 1)
    beta = np.zeros(max(data["question_id"]) + 1)
    alpha = np.ones(max(data["question_id"]) + 1)
    #initialize gamma to all 0.5
    #gamma = np.ones(max(data["question_id"]) + 1) * 0.5
    # initialize gamma to all 0
    gamma = np.zeros(max(data["question_id"]) + 1)



    val_acc_lst = []
    train_likelihood = []
    val_likelihood = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha, gamma=gamma)

        train_likelihood.append(neg_lld)
        val_likelihood.append(neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha, gamma=gamma))

        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, gamma=gamma)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha, gamma = update_theta_beta(data, lr, theta, beta, gamma, alpha)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_likelihood, val_likelihood, alpha, gamma


def evaluate(data, theta, beta, alpha, gamma):
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
        
        theta_i = theta[u]
        beta_j = beta[q]
        alpha_j = alpha[q]
        gamma_j = gamma[q]

        exp1 = np.exp(alpha_j * (theta_i - beta_j))

        # P(C=1|Theta, Beta, Alpha, Gamma)
        p_a = (gamma_j + exp1) / (1 + exp1)
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

    iterations = 45
    lr = 0.005

    # 0.01, 35 -> Validation accuracy: 0.707874682472481
    # 0.005, 35 -> Validation accuracy: 0.7074513124470787
    # 0.005, 40 -> Validation accuracy: 0.7075924357888794
    # 0.005, 45 -> Test accuracy:  0.7056167090036692 Validation accuracy:  0.7082980524978831

    theta, beta, val_acc_lst, train_likelihood, val_likelihood, alpha, gamma = \
        irt(train_data, val_data, lr, iterations)

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
    iteration_lable = np.arange(1, iterations + 1)
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
    plt.show() 

    # evaluate the model on test data
    test_acc = evaluate(test_data, theta, beta, alpha, gamma)
    # evaluate the model on validation data
    val_acc = evaluate(val_data, theta, beta, alpha, gamma)
    # print both test and validation accuracy
    print("Test accuracy: ", test_acc, "\n")
    print("Validation accuracy: ", val_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    
    """ # choose j1, j2 ,j3
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
    plt.show() """


    



    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
