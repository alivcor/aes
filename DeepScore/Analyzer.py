import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import DeepScore_Metrics
import time
import datetime
import EventIssuer


class AnalyzerObject:
    def __init__(self, model, log_fn, dev_X, dev_Y, epoch_num, batch_size):
        self.model = model
        self.log_fn = log_fn
        self.dev_X = dev_X
        self.dev_Y = dev_Y
        self.epoch = epoch_num
        self.batch_size = batch_size
        self.dev_loss = 0
        self.dev_metric = 0
        self.qwk = -1
        self.lwk = -1

    def analyze(self):
        EventIssuer.issueMessage("Analyzer is evaluating over validation set..", self.log_fn)
        EventIssuer.issueMessage("Batch Size is "+ str(self.batch_size), self.log_fn)
        self.dev_loss, self.dev_metric = self.model.evaluate(self.dev_X, self.dev_Y, batch_size=self.batch_size, verbose=1)
        dev_pred = self.model.predict(self.dev_X, batch_size=self.batch_size).squeeze()
        self.qwk, self.lwk = calculate_kappa(self.dev_Y, dev_pred)
        print ""
        EventIssuer.issueMessage("[VALIDATION] : QWK : " + str(self.qwk) + " | LWK : " + str(self.lwk), self.log_fn)




def calculate_kappa(y_true, y_pred):
    # print y_true.shape[0], y_pred.shape[0]
    assert(y_true.shape[0] == y_pred.shape[0])

    decoded_y_true = np.zeros(y_true.shape[0])
    decoded_y_pred = np.zeros(y_pred.shape[0])
    for i in range(0, y_true.shape[0]):
        decoded_y_true[i] = decodeOneHot(y_true[i])
        decoded_y_pred[i] = decodeOneHot(y_pred[i])
    decoded_y_pred = [int(i) for i in decoded_y_pred]
    decoded_y_true = [int(i) for i in decoded_y_true]
    # print decoded_y_true
    # print decoded_y_pred
    qwk_value = DeepScore_Metrics.quadratic_weighted_kappa(decoded_y_true, decoded_y_pred, 0, 12)
    lwk_value = DeepScore_Metrics.linear_weighted_kappa(decoded_y_true, decoded_y_pred, 0, 12)
    return qwk_value, lwk_value



def decodeOneHot(rating_array):
    return np.argmax(rating_array)





