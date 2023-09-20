from torch.utils.data import DataLoader


def train_selective_inductive_bias_unsup_vox(
    args,
    outputs,
    device,
    agent_method,
    train_dvec_mode,
    early_stop_status,
    early_stopping,
    early_stopping_bkt,
    bucket_id,
    epoch,
    logger,
    kwargs_training_val,
):

    sub_labels = outputs[bucket_id]

    sub_dataset = kwargs_training_val["SubDatasetSpk"](
        kwargs_training_val["dataset"],
        sub_labels,
    )
    train_sub_loader = DataLoader(
        sub_dataset,
        batch_size=len(sub_labels),
        shuffle=False,
        collate_fn=kwargs_training_val["collateSpkr"],
        drop_last=True,
    )

    mel_db_batch = next(iter(train_sub_loader))

    x, spk = mel_db_batch

    if args.delta and args.delta_delta:
        feat_dim_processed = args.feature_dim * 3
    elif args.delta:
        feat_dim_processed = args.feature_dim * 2
    else:
        feat_dim_processed = args.feature_dim

    x = x.reshape(-1, args.seg_len, feat_dim_processed).to(device)
    spk = spk.to(device)

    # Create the input data
    input_data = {"x": x, "y": spk}

    if early_stop_status.early_stop:
        logger.info(f"Training of the bucket:{bucket_id} completed.")
        early_stopping.append(early_stop_status.early_stop)
        early_stopping_bkt[bucket_id] = early_stop_status.early_stop

        props = {
            "x": x,
            "y": spk,
            "bucket_id_selected": bucket_id,
            "early_stopping": early_stopping,
            "early_stopping_bkt": early_stopping_bkt,
        }

        return props
    else:
        early_stopping.append(early_stop_status.early_stop)
        early_stopping_bkt[bucket_id] = early_stop_status.early_stop

    # Train the d-vectors per bucket
    agent_method[train_dvec_mode](
        kwargs_training_val["dvectors"][bucket_id],
        kwargs_training_val["opt_dvec_type"],
        kwargs_training_val["contrastive_loss"][bucket_id],
        bucket_id,
        input_data,
        epoch,
        kwargs_training_val["filename_dvec"][bucket_id],
        kwargs_training_val["filename_dvec_dir"][bucket_id],
    )

    props = {
        "x": x,
        "y": spk,
        "bucket_id_selected": bucket_id,
        "early_stopping": early_stopping,
        "early_stopping_bkt": early_stopping_bkt,
    }

    return props
