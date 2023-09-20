from torch.utils.data import DataLoader


def train_reg_unsup_old_scratch(
    epoch,
    early_stopping,
    hparams,
    args,
    outputs,
    bucket_id,
    device,
    agent_method,
    total_num_spk_per_bkt,
    kwargs_training_val,
):

    sub_lbs_old = outputs[bucket_id]

    sub_dataset = kwargs_training_val["SubDatasetGdrSpk"](
        kwargs_training_val["dataset"], sub_lbs_old
    )
    train_sub_loader = DataLoader(
        sub_dataset,
        batch_size=len(sub_lbs_old),
        shuffle=False,
        collate_fn=kwargs_training_val["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch = next(iter(train_sub_loader))
    x, _, spk = mel_db_batch
    x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk = spk.to(device)

    input_data = {"x": x, "y": spk}

    if early_stopping.early_stop:
        return input_data

    # Re-train the d-vector with the new registered bucket
    agent_method[hparams.train_dvec_mode](
        kwargs_training_val["dvectors"][bucket_id],
        kwargs_training_val["opt_dvec_type"],
        kwargs_training_val["contrastive_loss"][bucket_id],
        bucket_id,
        input_data,
        epoch,
        kwargs_training_val["filename_dvec_reg"][bucket_id],
        kwargs_training_val["filename_dvec_dir_reg"][bucket_id],
    )

    return input_data
