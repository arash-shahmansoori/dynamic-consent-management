from torch.utils.data import DataLoader


def eval_per_epoch_per_bucket_contrastive_updated_supervised(
    args,
    device,
    outputs_updated,
    buckets,
    pred_indx,
    gtruth_indx,
    epoch,
    **kwargs_validation,
):

    for _, bkt_id in enumerate(buckets):

        kwargs_validation["dvectors"][bkt_id].eval()

        sub_lbs_current_validation = outputs_updated[bkt_id]

        sub_dataset_current_validation = kwargs_validation["SubDatasetGdrSpk"](
            kwargs_validation["dataset_val"],
            sub_lbs_current_validation,
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
        t_val_buffer = spk_val.view(-1)

        # Compute predicted index
        predicted_index, ground_truth_indx = kwargs_validation["agent"].predict_index(
            kwargs_validation["classifier"],
            kwargs_validation["classifier_ma"],
            x_val_buffer,
            t_val_buffer,
        )

        # Compute performance measures of the classifier
        acc, loss = kwargs_validation["agent"].accuracy_loss(
            kwargs_validation["classifier"],
            kwargs_validation["classifier_ma"],
            kwargs_validation["ce_loss"],
            x_val_buffer,
            t_val_buffer,
        )

        # if args.log_training:

        #     loss_display = loss.item()
        #     acc_display = acc.item()

        #     epoch_display = f"Train Epoch: {epoch}| "
        #     if bkt_id == 0:
        #         bucket_display = f"Bucket:{0}| "
        #     else:
        #         bucket_display = f"Bucket:{bkt_id}| "

        #     val_loss_display = f"Loss:{loss_display:0.3f}| "
        #     val_acc_display = f"Acc:{acc_display:0.3f}| "

        #     print(epoch_display, bucket_display, val_loss_display, val_acc_display)

        pred_indx[bkt_id].append(predicted_index.tolist())
        gtruth_indx[bkt_id].append(ground_truth_indx.tolist())

    out = {
        "pred_indx": pred_indx,
        "gtruth_indx": gtruth_indx,
    }

    return out
