import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import DeepScore_Metrics
import time
import datetime
import EventIssuer


def analyze(model, logfn):

