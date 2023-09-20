from torch.utils.data import DataLoader


def reg_sup_old(
    early_stopping_status,
    args,
    outputs,
    bucket_id,
    device,
    kwargs_training_val,
):

    early_stopping_status.append(True)

    sub_lbs_old = outputs[bucket_id]

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

    mel_db_batch = next(iter(train_sub_loader))

    x, _, spk = mel_db_batch
    x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk = spk.to(device)

    input_data = {"x": x, "y": spk}

    return input_data
