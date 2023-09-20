from torch.utils.data import DataLoader


def train_selective_unreg_inductive_bias_unsup(
    epoch,
    early_stopping,
    agent_method,
    hparams,
    args,
    outputs,
    indx_selected,
    device,
    kwargs_training_val,
):

    sub_lbs = outputs[indx_selected]

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

    mel_db_batch = next(iter(train_sub_loader))

    x, _, spk = mel_db_batch
    x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk = spk.to(device)

    # Create input data
    input_data = {"x": x, "y": spk}

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
