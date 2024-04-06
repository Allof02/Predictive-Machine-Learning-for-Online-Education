from utils import *

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return np.exp(x)/(1 + np.exp(x))

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
    log_lklihood = 0
    
    # get data from dictionary
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    
    for i in range(len(user_id)):
        u = user_id[i]
        q = question_id[i]
        cij = is_correct[i]
        x = alpha[q] * (theta[u] - beta[q])
        p = gamma[q] + (1-gamma[q])*sigmoid(x)
        
        # Apply epsilon to prevent taking logarithm of zero or one
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        # Compute the log likelihood for the given data point
        log_lklihood += cij*np.log(p) + (1 - cij)*np.log(1 - p)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha, gamma):
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
        x = alpha[q]*(theta[u] - beta[q])
        y = gamma[q] + (1 - gamma[q])*(np.exp(x)/(1+np.exp(x)))
        pij = np.exp(x)/(1+np.exp(x))
        theta1 = (1-gamma[q])*pij*(1-pij)*alpha[q]
        beta1 = -(1-gamma[q])*pij*(1-pij)*alpha[q]
        alpha1 = (1-gamma[q])*(pij*(theta[u]-beta[q])/(1+np.exp(x)))
        
        # cij - sigmoid(x) is the gradient with respect to theta
        theta[u] += lr * (cij*alpha[q]*np.exp(x)/(gamma[q] + np.exp(x)) - (alpha[q]*np.exp(x)/(1+np.exp(x))))
        # -cij + sigmoid(x) is the gradient with respect to beta
        alpha[q] += lr * (cij * (theta[u] - beta[q]) *(np.exp(x) / (gamma[q] + np.exp(x))) - (theta[u] - beta[q]) * sigmoid(x))
        gamma[q] += lr*(cij/(gamma[q] + np.exp(x)) - (1-cij)/(1-(gamma[q])))
        beta[q] += lr * ((-cij*alpha[q]*np.exp(x)/(gamma[q] + np.exp(x))) + (alpha[q]*np.exp(x)/(1+np.exp(x))))
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
    theta = np.zeros(np.max(data["user_id"]) + 1)
    beta = np.zeros(np.max(data["question_id"]) + 1)
    alpha = np.zeros(np.max(data["question_id"]) + 1)
    gamma = np.zeros(np.max(data["question_id"]) + 1)

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
        theta, beta, alpha, gamma = update_theta_beta(data, lr, theta, beta, alpha, gamma)
    return theta, beta, alpha, gamma, val_acc_lst, train_likelihood, val_likelihood


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
        x = alpha[q] * (theta[u] - beta[q])
        prob_correct = gamma[q] + (1 - gamma[q])*np.exp(x)/(1+np.exp(x))
        pred.append(prob_correct >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv(r"C:\Users\harvi\Desktop\CSC311\project-starter-files-karathah\data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse(r"C:\Users\harvi\Desktop\CSC311\project-starter-files-karathah\data")
    val_data = load_valid_csv(r"C:\Users\harvi\Desktop\CSC311\project-starter-files-karathah\data")
    test_data = load_public_test_csv(r"C:\Users\harvi\Desktop\CSC311\project-starter-files-karathah\data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iterations = 30
    lr = 0.008
    theta, beta, alpha, gamma, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, lr, iterations)
    
    # 30, 0.02 -> test acc = 0.7078, val acc = 0.7073
    # 50, 0.008 -> Test accuracy:  0.7044877222692634 Validation accuracy:  0.7054755856618685
    # 40, 0.01 -> Test accuracy:  0.7044877222692634 Validation accuracy:  0.7063223257126728
    # 35, 0.01: Test accuracy:  0.7047699689528648 Validation accuracy:  0.7067456957380751
    # 30, 0.01: Test accuracy:  0.7050522156364663 Validation accuracy:  0.7070279424216765
    # 40, 0.04: Test accuracy:  0.7075924357888794 Validation accuracy:  0.70575783234547
    # 20, 0.04: Test accuracy:  0.707310189105278 Validation accuracy:  0.7068868190798758
    # 40, 0.015: Test accuracy:  0.7053344623200677 Validation accuracy:  0.7075924357888794
    # 50, 0.02: Test accuracy:  0.7047699689528648 Validation accuracy:  0.7067456957380751
    
    """
    best_val_acc = 0
    best_lr = 0
    best_iterations = 0
    
    for lr in np.linspace(0.01, 0.03, 10):
        for iterations in range(20, 40, 5):
            theta, beta, alpha, gamma, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, lr, iterations)
            test_acc = evaluate(test_data, theta, beta, alpha, gamma)
            val_acc = evaluate(val_data, theta, beta, alpha, gamma)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_iterations = iterations
    
    print("Best validation accuracy: ", best_val_acc)
    print("Best learning rate: ", best_lr)
    print("Best number of iterations: ", best_iterations)
    
    """
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

    ######################################################################
    
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
