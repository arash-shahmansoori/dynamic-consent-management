import torch
import numpy as np

from torch.utils.data import DataLoader


def train_selective_sofar_reg_inductive_bias_unsup_causal_scratch(
    epoch,
    indx_opt_unique_bkt_sofar,
    spk_selected_strategy_sofar,
    early_stopping,
    hparams,
    args,
    outputs,
    indx_selected_sofar,
    device,
    agent_method,
    total_num_spk_per_bkt,
    kwargs_training_val,
):

    sub_lbs_other = np.array(indx_opt_unique_bkt_sofar)[
        spk_selected_strategy_sofar
    ].tolist()
    sub_lbs_old = outputs[indx_selected_sofar]

    sub_dataset = kwargs_training_val["SubDatasetGdrSpk"](
        kwargs_training_val["dataset"],
        sub_lbs_old,
    )
    train_sub_loader = DataLoader(
        sub_dataset,
        batch_size=len(sub_lbs_old),
        shuffle=False,
        collate_fn=kwargs_training_val["collateGdrSpkr"],
        drop_last=True,
    )

    sub_dataset_other = kwargs_training_val["SubDatasetGdrSpk"](
        kwargs_training_val["dataset_other"], sub_lbs_other
    )
    train_sub_loader_other = DataLoader(
        sub_dataset_other,
        batch_size=len(sub_lbs_other),
        shuffle=False,
        collate_fn=kwargs_training_val["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch = next(iter(train_sub_loader))
    mel_db_batch_other = next(iter(train_sub_loader_other))

    x, _, spk = mel_db_batch
    x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk = spk.to(device)

    x_other, _, spk_other = mel_db_batch_other
    x_other = x_other.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk_other = spk_other.to(device)

    x_cat = torch.cat((x, x_other), dim=0)
    spk_cat = torch.cat(
        (spk, spk_other + args.n_speakers), dim=0
    )  # shift the labels for new registrations

    # Create input data
    input_data = {"x": x_cat, "y": spk_cat}

    if early_stopping.early_stop:
        return input_data

    # Re-train the d-vector with the new registered speaker(s) so far
    agent_method[hparams.train_dvec_mode](
        kwargs_training_val["dvectors"][indx_selected_sofar],
        kwargs_training_val["opt_dvec_type"],
        kwargs_training_val["contrastive_loss"][indx_selected_sofar],
        indx_selected_sofar,
        input_data,
        epoch,
        kwargs_training_val["filename_dvec_reg"][indx_selected_sofar],
        kwargs_training_val["filename_dvec_dir_reg"][indx_selected_sofar],
    )

    return input_data
