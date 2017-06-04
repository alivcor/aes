import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import DeepScore_Metrics
import time
import datetime
import EventIssuer


def qw_kappa(y_true, y_pred):
    print y_true, y_pred
    print y_true.shape[0], y_pred.shape[0]
    assert(y_true.shape[0] == y_pred.shape[0])

    decoded_y_true = np.zeros(y_true.shape[0])
    decoded_y_pred = np.zeros(y_pred.shape[0])
    for i in range(0, y_true.shape[0]):
        decoded_y_true[i] = decodeOneHot(y_true[i])
        decoded_y_pred[i] = decodeOneHot(y_pred[i])
    qwk_value = DeepScore_Metrics.quadratic_weighted_kappa(decoded_y_true, decoded_y_pred)
    return qwk_value



def decodeOneHot(rating_array):
    return rating_array.index(1)



def analyze(model, logfn):

