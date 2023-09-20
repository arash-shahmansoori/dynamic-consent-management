import torch
import numpy as np

# Functions for selection of buckets with/without overlaps
def cor_seq_counter_list(N, s, stride):

    x = []
    y = []

    for i in range(0, N - 1, stride):
        for j in range(0, N):
            if j >= i and j < i + s:
                x.append(j)
        if len(x) == s:
            y.append(x)
            x = []
    return y


def cor_seq_counter(N, s, stride):

    x = []
    y = []

    for i in range(0, N - 1, stride):
        for j in range(0, N):
            if j >= i and j < i + s:
                x.append(j)
        if len(x) == s:
            y.append(torch.tensor(x))
            x = []
    return torch.stack(y).shape[0], torch.stack(y)


def unreg_spks_per_bkts(spks_in_buckets, unreg_spks):
    z_raw, z_nonempty = [], []
    for s in spks_in_buckets:
        new_s = []
        for u in s:
            if u not in unreg_spks:
                new_s.append(u)
        if len(new_s) != 0:
            z_nonempty.append(new_s)
        z_raw.append(new_s)

    return z_raw, z_nonempty


def count_similar_elements(x0):
    # Computes the number of repetiotion of each element in a list
    # and returns the result as a dictionary with the list unique elements as the keys

    xx = torch.tensor(x0).to(torch.int64).tolist()
    x_ = torch.tensor(xx).unique().to(torch.int64).tolist()

    counter = {i_: 0 for i_ in x_}
    for j in x_:
        for i in xx:
            if i == j:
                counter[j] += 1

    return counter


def progressive_indx_normalization(spk, base_spk_per_bkt, bucket_id, device):

    spk_per_bkt = (bucket_id + 1) * base_spk_per_bkt

    for j, i in enumerate(sorted(spk.tolist())):
        # indx = np.where(np.array(spk) >= 40)
        if i >= spk_per_bkt:
            # if len(indx[0] != 0):
            # new_j = j % (indx[0][-1] + 1)

            spk[j] = spk[0] % base_spk_per_bkt + spk_per_bkt
    spk = spk - spk[0]
    return spk.to(device)


def create_calibrated_length(x, y, latent_dim):

    total_input_length = len(x.view(-1))
    num_spks = len(y.unique())

    remaining = total_input_length // latent_dim
    num_utts = remaining // num_spks

    length_calibrated = num_spks * num_utts * latent_dim

    return length_calibrated, num_spks, num_utts


def Progressive_normalized_label(x, bucket_id):
    z = x
    if bucket_id == 0:
        return z
    else:
        z += bucket_id * len(x.unique()) - 1
        return z


def label_normalizer_per_bucket(spk):

    num_utts = len(spk) // len(spk.unique())

    normalized_spk = []
    for i in range(len(spk.unique())):
        normalized_spk.append(num_utts * [i])
    spk_normalized = torch.tensor(normalized_spk).view(-1)

    return spk_normalized


def normalize_per_bkt_labels(spk_per_bkt):

    num_utts = len(spk_per_bkt) // len(spk_per_bkt.unique())
    _spk_per_bkt = []
    for i in range(len(spk_per_bkt.unique())):
        _spk_per_bkt.append(num_utts * [i])

    return torch.tensor(_spk_per_bkt).view(-1)


def label_normalizer_progressive(spk_next_normalized, spk_normalized, spk_init):

    if len(spk_init) != 0:
        spk_next_normalized_scaled = normalize_per_bkt_labels(spk_next_normalized)
        spk_next_progressive_normalized = spk_next_normalized_scaled + len(
            torch.cat(spk_init, dim=0).unique()
        )
    elif len(spk_init) == 0:
        spk_next_normalized_scaled = normalize_per_bkt_labels(spk_next_normalized)
        spk_next_progressive_normalized = spk_next_normalized_scaled

    if len(spk_normalized) != 0:
        spk_init.append(spk_next_progressive_normalized)
    else:
        spk_next_normalized_scaled = normalize_per_bkt_labels(spk_next_normalized)
        spk_init.append(spk_next_normalized_scaled)

    return spk_next_progressive_normalized


def customized_labels_unreg_unsup(updated_outputs, outputs):

    z = []

    for i, j in zip(outputs, updated_outputs):
        if len(i) == len(j):
            z.append(i)
        else:
            z.append(j + [x for x in i if x not in j])

    return z


def customized_labels_unreg_unsup_bkt(updated_outputs_bkt, outputs_bkt):

    if len(updated_outputs_bkt) == len(outputs_bkt):
        z_new = outputs_bkt
        z_removed = []
    else:
        z_new = updated_outputs_bkt
        z_removed = [z for z in outputs_bkt if z not in z_new]

    return z_new, z_removed
