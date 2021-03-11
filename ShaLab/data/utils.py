import numpy as np
import torch


def insert_go_token(batch, go_token_id):
    if isinstance(batch, np.ndarray):
        data = batch.transpose()
    else:
        data = batch.cpu().numpy().transpose()
    result = np.zeros((batch.size(1), batch.size(0) + 1), dtype=np.int64)
    for i in range(batch.size(1)):
        word_indexes = np.nonzero(data[i])[0]
        result[i][0] = go_token_id
        result[i][1:word_indexes.size + 1] = data[i][word_indexes]
    return torch.from_numpy(result.transpose())


def append_eos_token(batch, eos_token_id):
    if isinstance(batch, np.ndarray):
        data = batch.transpose()
    else:
        data = batch.cpu().numpy().transpose()
    result = np.zeros((batch.size(1), batch.size(0) + 1), dtype=np.int64)
    for i in range(batch.size(1)):
        word_indexes = np.nonzero(data[i])[0]
        result[i][:word_indexes.size] = data[i][word_indexes]
        result[i][word_indexes.size] = eos_token_id
    return torch.from_numpy(result.transpose())
