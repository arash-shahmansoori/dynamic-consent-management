import torch


def filter_spk_indx(spks):
    count = {str(int(k)): 1 for _, k in enumerate(spks.unique())}
    indx = {str(int(k)): [] for _, k in enumerate(spks.unique())}
    indx_unique = {}
    for i, v in enumerate(spks):
        if i > 0:
            if spks[i] == spks[i - 1]:
                count[str(int(v))] += 1
                indx[str(int(v))].append(i - 1)
                indx[str(int(v))].append(i)

                indx_unique[str(int(v))] = torch.tensor(indx[str(int(v))]).unique()

    count_list = []
    for _, v in count.items():
        count_list.append(v)

    count_min = torch.tensor(count_list).min().item()

    filtered_indx = []
    for spk, _ in indx_unique.items():
        filtered_indx.append(indx_unique[spk][:count_min])

    filtered_indx_list = torch.cat(filtered_indx).view(-1).tolist()

    return filtered_indx_list
