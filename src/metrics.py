import numpy as np

def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])

    min_size = np.min([len(act_set), len(pred_set)])

    result = len(act_set & pred_set) / float(min_size)
    return result