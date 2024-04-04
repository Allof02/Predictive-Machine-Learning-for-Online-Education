from sklearn.impute import KNNImputer
from utils import *


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
    acc = None
    matrix_T = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat_T = nbrs.fit_transform(matrix_T)
    mat = mat_T.T  # Transpose back to original shape
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse(r"C:\Users\harvi\Desktop\CSC311\project-starter-files-karathah\data").toarray()
    val_data = load_valid_csv(r"C:\Users\harvi\Desktop\CSC311\project-starter-files-karathah\data")
    test_data = load_public_test_csv(r"C:\Users\harvi\Desktop\CSC311\project-starter-files-karathah\data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    
    k_values = [1, 6, 11, 16, 21, 26]
    val_accuracies_user = []
    val_accuracies_item = []

    for k in k_values:
        print("Running user-based collaborative filtering for k =", k)
        acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
        val_accuracies_user.append(acc_user)

        print("Running item-based collaborative filtering for k =", k)
        acc_item = knn_impute_by_item(sparse_matrix, val_data, k)
        val_accuracies_item.append(acc_item)

    # Plotting validation accuracies
    import matplotlib.pyplot as plt

    plt.plot(k_values, val_accuracies_user, label="User-based CF")
    plt.plot(k_values, val_accuracies_item, label="Item-based CF")
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. k for User-based and Item-based CF")
    plt.legend()
    plt.show()

    # Selecting k with best performance
    best_k_user = k_values[np.argmax(val_accuracies_user)]
    best_k_item = k_values[np.argmax(val_accuracies_item)]

    print("Best k for user-based CF:", best_k_user)
    print("Best k for item-based CF:", best_k_item)

    # Report test accuracy for best k
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)

    print("Test Accuracy for user-based CF with best k:", test_acc_user)
    print("Test Accuracy for item-based CF with best k:", test_acc_item)
    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    #pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
