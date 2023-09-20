import torch

from torch.utils.data import DataLoader


def train_selective_unreg_inductive_bias_unsup_v2(
    epoch,
    early_stopping,
    agent_method,
    hparams,
    args,
    updated_outputs,
    indx_selected,
    device,
    kwargs_training_val,
):

    sub_lbs = updated_outputs[indx_selected]
    sub_lbs_wittness = [10, 11, 12]

    sub_dataset = kwargs_training_val["SubDatasetGdrSpk"](
        kwargs_training_val["dataset"],
        sub_lbs,
    )
    train_sub_loader = DataLoader(
        sub_dataset,
        batch_size=len(sub_lbs),
        shuffle=False,
        collate_fn=kwargs_training_val["collateGdrSpkr"],
        drop_last=True,
    )

    sub_dataset_wittness = kwargs_training_val["SubDatasetGdrSpk"](
        kwargs_training_val["dataset_other"],
        sub_lbs_wittness,
    )
    train_sub_loader_wittness = DataLoader(
        sub_dataset_wittness,
        batch_size=len(sub_lbs_wittness),
        shuffle=False,
        collate_fn=kwargs_training_val["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch = next(iter(train_sub_loader))
    mel_db_batch_wittness = next(iter(train_sub_loader_wittness))

    x, _, spk = mel_db_batch
    x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk = spk.to(device)

    x_wittness, _, spk_wittness = mel_db_batch_wittness
    x_wittness = x_wittness.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk_wittness = spk_wittness.to(device)

    x_cat = torch.cat((x, 10000 * x_wittness), dim=0)
    spk_cat = torch.cat((spk, spk_wittness), dim=0)

    # Create input data
    input_data = {"x": x_cat, "y": spk_cat}

    if early_stopping.early_stop:
        return input_data

    # Re-train the d-vector with the new registered bucket
    agent_method[hparams.train_dvec_mode](
        kwargs_training_val["dvectors"][indx_selected],
        kwargs_training_val["opt_dvec_type"],
        kwargs_training_val["contrastive_loss"][indx_selected],
        indx_selected,
        input_data,
        epoch,
        kwargs_training_val["filename_dvec_unreg"][indx_selected],
        kwargs_training_val["filename_dvec_dir_unreg"][indx_selected],
    )

    return input_data
