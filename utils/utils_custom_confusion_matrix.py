import torch
import numpy as np


def custom_confusion_matrix(actual, predicted):

    classes_actual = np.unique(actual)
    classes_predicted = np.unique(predicted)

    confusion_mtx = np.empty((len(classes_actual), len(classes_predicted)), dtype=int)

    for i, a in enumerate(classes_actual):
        for j, p in enumerate(classes_predicted):
            confusion_mtx[i, j] = np.where((actual == a) * (predicted == p))[0].shape[0]

    norm_sum = np.empty((len(classes_actual),), dtype=int)

    for i, a in enumerate(classes_actual):
        norm_sum[i] = sum(confusion_mtx[i, :])

    return confusion_mtx, norm_sum


def normalize_custom_confusion_matrix(confusion_mtx, norm_sum):
    new_cfm = []
    for i in range(len(norm_sum)):
        new_cfm.append(confusion_mtx[i, :] / norm_sum[i])
    return (
        torch.tensor(np.array(new_cfm))
        .reshape((confusion_mtx.shape[0], confusion_mtx.shape[1]))
        .tolist()
    )