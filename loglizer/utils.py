"""
The utility functions of loglizer

Authors: 
    LogPAI Team

"""

# from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def metrics(y_prob, y_true):
    """ 
    Calculate evaluation metrics for precision, recall, f1, AUC and AUPR.

    Arguments
    ---------
        y_prob: ndarray, the predicted probability list
        y_true: ndarray, the ground truth label list

    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
        auc: float, AUC value
        aupr: float, AUPR value
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_pred = np.round(y_prob)  # convert probabilities to class predictions

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum(y_true == 0) - TN
    FN = np.sum(y_true == 1) - TP
    precision = 100 * TP / (TP + FP + 1e-8)
    recall = 100 * TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    auc = 100 * roc_auc_score(y_true, y_prob)
    aupr = 100 * average_precision_score(y_true, y_prob)

    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    auc = round(auc, 4)
    aupr = round(aupr, 4)

    print(f"Confusion Matrix: TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    return precision, recall, f1, auc, aupr



if __name__ == "__main__":
    print(metrics(np.array([1, 1, 1, 0, 0, 0]), np.array([1, 0, 1, 1, 0, 0])))

