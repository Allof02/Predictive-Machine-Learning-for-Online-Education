from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # Since the matrix is students as rows and questions as columns, and knnimputer is by default row-based,
    # transpose the matrix first to do item-based knn:
    matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    # TBH, im not sure why i need to transpose the matrix back to get the correct accuracy (i used mat first but seems to be wrong so i used mat.T here now)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # user-based knn
    kl = [1, 6, 11, 16, 21]
    acc_user = []
    for k in kl:
        acc_user.append(knn_impute_by_user(sparse_matrix, val_data, k))
    

    # find the best k
    best_k = kl[acc_user.index(max(acc_user))]
    # highest accuracy
    highest_acc_user = max(acc_user)
    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    # print the best k and the corresponding highest accuracy
    print("Best k (user-based): {}".format(best_k))
    print("Highest accuracy: {}".format(highest_acc_user))
    print("Test accuracy: {}".format(test_acc))

    # plot
    plt.plot(kl, acc_user)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. k (User-based)")
    plt.show()

    # item-based knn
    acc_item = []
    for k in kl:
        acc_item.append(knn_impute_by_item(sparse_matrix, val_data, k))
    
    # find the best k
    best_k = kl[acc_item.index(max(acc_item))]
    # highest accuracy
    highest_acc_item = max(acc_item)
    test_acc = knn_impute_by_item(sparse_matrix, test_data, best_k)

    # print the best k and the corresponding highest accuracy
    print("Best k (item-based): {}".format(best_k))
    print("Highest accuracy: {}".format(highest_acc_item))
    print("Test accuracy: {}".format(test_acc))

    # plot
    plt.plot(kl, acc_item)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. k (Item-based)")
    plt.show()

    # part c and d are in the report

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
