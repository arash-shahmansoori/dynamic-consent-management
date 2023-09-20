import os
import json
import torch

from pathlib import Path
from utils import (
    cor_seq_counter_list,
    save_as_json,
    create_filenames_bkts_json,
)


def create_unique_opt_bkts_spks(
    dvectors,
    args,
    hparams,
    dataset_validation,
    dataset_validation_prev_other,
    dataset_validation_other,
    device,
    compute_opt_bkt_final,
    unique_opt_seq_final,
):
    """Compute unique set of optimal buckets and corresponding indices according to the
    Euclidean distance.

    """

    round_num = hparams.round_num

    print(f"The optimal bucket(s) to be found for DP round:{round_num}")

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    labels = [i for i in range(args.n_speakers)]
    labels_other = [i for i in range(args.n_speakers_other)]

    outputs_val = cor_seq_counter_list(
        len(labels), args.spk_per_bucket, args.spk_per_bucket
    )

    # Number of new registrations steps (set to #spk_per_bucket for each round)
    _bunch_reg_steps = args.n_speakers // args.spk_per_bucket + args.spk_per_bucket

    file_name_indx_pop_dict, file_name_indx_bktopt_dict = create_filenames_bkts_json(
        args,
        _bunch_reg_steps,
    )

    removed_indices, opt_buckets = [], []
    # Initialize the list of the new speakers' IDs to be registered
    for _, _, files in os.walk(result_dir_acc_val_path):
        for file in files:
            for c_ in range(_bunch_reg_steps):
                if file == file_name_indx_pop_dict[c_][0] and c_ <= round_num:

                    with open(
                        Path(result_dir_acc_val_path, file_name_indx_pop_dict[c_][0]),
                        "r",
                    ) as spk_indx_removed_:
                        spk_indx_removed = json.load(spk_indx_removed_)
                        removed_indices.append(spk_indx_removed)

                    with open(
                        Path(
                            result_dir_acc_val_path,
                            file_name_indx_bktopt_dict[c_][0],
                        ),
                        "r",
                    ) as bkt_selected:
                        bkt_selected_ = json.load(bkt_selected)
                        opt_buckets.append(bkt_selected_)

    removed_indices_flattened = [s for t in removed_indices for s in t]
    opt_buckets_flattened = [s for t in opt_buckets for s in t]

    _id_chosen = []
    for j_, i_ in enumerate(labels_other):
        if j_ not in removed_indices_flattened:
            _id_chosen.append(i_)
        else:
            _id_chosen.append(None)

    indx_selected_list_ = []
    indx_selected_ = {i_: [] for _, i_ in enumerate(_id_chosen)}
    for _, i_ in enumerate(_id_chosen):
        if i_ != None:
            dists_cl_other, n_x, n_y = compute_opt_bkt_final(
                dvectors,
                i_,
                round_num,
                hparams.num_of_buckets,
                outputs_val,
                removed_indices_flattened,
                opt_buckets_flattened,
                dataset_validation,
                dataset_validation_prev_other,
                dataset_validation_other,
                device,
                args,
            )

            dists_cl_other_list = []
            for i in range(hparams.num_of_buckets):
                if n_x[i] and n_y[i]:
                    dists_cl_other_list.append(
                        torch.tensor(dists_cl_other[i])
                        .view(1, n_x[i][0], n_y[i][0])
                        .mean(dim=1)
                        .view(-1)
                        .tolist()
                    )

            val_selectedb_list = []
            for i_, _ in enumerate(dists_cl_other_list):
                if n_y[i_]:

                    val_selectedb_, _ = torch.min(
                        # torch.tensor(dists_cl_other_list[i_]).view(1, n_y[i_][0]),
                        torch.tensor(dists_cl_other_list[i_]).view(
                            1, len(dists_cl_other_list[i_])
                        ),
                        dim=1,
                    )
                    val_selectedb_list.append(val_selectedb_)
            _, indx_selectedc_ = torch.min(
                torch.tensor(val_selectedb_list).view(-1), dim=0
            )

            # List of optimal buckets
            indx_selected_list_.append(indx_selectedc_)

        elif i_ == None:
            # List of optimal buckets for seq_reg
            indx_selected_list_.append(torch.tensor(-1))

    indx_selected_ = torch.stack(indx_selected_list_, dim=0).view(-1).tolist()
    indx_after, _, X_after = unique_opt_seq_final(indx_selected_)

    Xp_ = [xp for xp in X_after if xp >= 0]

    save_as_json(
        result_dir_acc_val_path, file_name_indx_pop_dict[round_num][0], indx_after
    )
    save_as_json(result_dir_acc_val_path, file_name_indx_bktopt_dict[round_num][0], Xp_)

    opt_bkt_list = torch.tensor(Xp_).view(-1).tolist()
    indx_opt_bkt_list = torch.tensor(indx_after).view(-1).tolist()

    return opt_bkt_list, indx_opt_bkt_list


def create_unique_opt_bkt_spks_existing(args, round_num):

    """Compute unique set of optimal buckets and corresponding indices according to the
    existing optimal set of initial buckets.

    """

    # round_num = hparams.round_num

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Number of new registrations steps (set to #spk_per_bucket for each round)
    _bunch_reg_steps = args.n_speakers // args.spk_per_bucket + args.spk_per_bucket

    file_name_indx_pop_dict, file_name_indx_bktopt_dict = create_filenames_bkts_json(
        args,
        _bunch_reg_steps,
    )

    opt_bkt_list, indx_opt_bkt_list = [], []
    removed_indices, opt_buckets = [], []
    # Initialize the list of the new speakers' IDs to be registered
    for _, _, files in os.walk(result_dir_acc_val_path):
        for file in files:
            for c_ in range(_bunch_reg_steps):
                if file == file_name_indx_pop_dict[c_][0] and c_ == round_num:

                    with open(
                        Path(result_dir_acc_val_path, file_name_indx_pop_dict[c_][0]),
                        "r",
                    ) as spk_indx_removed_:
                        spk_indx_removed = json.load(spk_indx_removed_)
                        removed_indices.append(spk_indx_removed)

                    with open(
                        Path(
                            result_dir_acc_val_path,
                            file_name_indx_bktopt_dict[c_][0],
                        ),
                        "r",
                    ) as bkt_selected:
                        bkt_selected_ = json.load(bkt_selected)
                        opt_buckets.append(bkt_selected_)

    opt_buckets_flattened = [s for t in opt_buckets for s in t]
    removed_indices_flattened = [s for t in removed_indices for s in t]

    opt_bkt_list = opt_buckets_flattened
    indx_opt_bkt_list = removed_indices_flattened

    return opt_bkt_list, indx_opt_bkt_list


def create_unique_opt_bkt_spks_sofar(args, round_num):

    """Compute unique set of optimal buckets and corresponding indices according to the
    existing optimal set of initial buckets from the previous rounds.
    """

    # round_num = hparams.round_num

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Number of new registrations steps (set to #spk_per_bucket for each round)
    _bunch_reg_steps = args.n_speakers // args.spk_per_bucket + args.spk_per_bucket

    file_name_indx_pop_dict, file_name_indx_bktopt_dict = create_filenames_bkts_json(
        args,
        _bunch_reg_steps,
    )

    opt_bkt_list, indx_opt_bkt_list = [], []
    removed_indices, opt_buckets = [], []
    # Initialize the list of the new speakers' IDs to be registered
    for _, _, files in os.walk(result_dir_acc_val_path):
        for file in files:
            for c_ in range(_bunch_reg_steps):
                if file == file_name_indx_pop_dict[c_][0] and c_ < round_num:

                    with open(
                        Path(result_dir_acc_val_path, file_name_indx_pop_dict[c_][0]),
                        "r",
                    ) as spk_indx_removed_:
                        spk_indx_removed = json.load(spk_indx_removed_)
                        removed_indices.append(spk_indx_removed)

                    with open(
                        Path(
                            result_dir_acc_val_path,
                            file_name_indx_bktopt_dict[c_][0],
                        ),
                        "r",
                    ) as bkt_selected:
                        bkt_selected_ = json.load(bkt_selected)
                        opt_buckets.append(bkt_selected_)

    opt_buckets_flattened = [s for t in opt_buckets for s in t]
    removed_indices_flattened = [s for t in removed_indices for s in t]

    opt_bkt_list = opt_buckets_flattened
    indx_opt_bkt_list = removed_indices_flattened

    return opt_bkt_list, indx_opt_bkt_list