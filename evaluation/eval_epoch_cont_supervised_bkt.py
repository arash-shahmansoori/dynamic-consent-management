from torch.utils.data import DataLoader


def eval_per_epoch_per_bucket_contrastive_supervised(
    args,
    device,
    outputs,
    buckets,
    pred_indx,
    gtruth_indx,
    **kwargs_validation,
):

    for _, bkt_id in enumerate(buckets):

        kwargs_validation["dvectors"][bkt_id].eval()

        sub_lbs_current_validation = outputs[bkt_id]

        sub_dataset_current_validation = kwargs_validation["SubDatasetGdrSpk"](
            kwargs_validation["dataset_val"], sub_lbs_current_validation
        )
        validation_loader_current = DataLoader(
            sub_dataset_current_validation,
            batch_size=len(sub_lbs_current_validation),
            collate_fn=kwargs_validation["collateGdrSpkr"],
            drop_last=True,
        )

        mel_db_batch_validation = next(iter(validation_loader_current))

        x_val, _, spk_val = mel_db_batch_validation
        x_val = x_val.reshape(-1, args.seg_len, args.feature_dim)
        x_val, spk_val = x_val.to(device), spk_val.to(device)

        xe_val = kwargs_validation["dvectors"][bkt_id](x_val).detach()

        # Store contrastive embeddings for validation
        x_val_buffer = xe_val.view(-1, args.dim_emb)
        t_val_buffer = spk_val.view(-1).to(device)

        # Compute predicted index
        predicted_index, ground_truth_indx = kwargs_validation["agent"].predict_index(
            kwargs_validation["classifier"],
            kwargs_validation["classifier_ma"],
            x_val_buffer,
            t_val_buffer,
        )

        pred_indx[bkt_id].append(predicted_index.tolist())
        gtruth_indx[bkt_id].append(ground_truth_indx.tolist())

    out = {
        "pred_indx": pred_indx,
        "gtruth_indx": gtruth_indx,
    }

    return out
