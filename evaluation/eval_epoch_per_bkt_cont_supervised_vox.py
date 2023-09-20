from torch.utils.data import DataLoader


def eval_per_epoch_per_bkt_contrastive_supervised_vox(
    args,
    device,
    outputs,
    buckets,
    val_loss,
    val_acc,
    epoch,
    **kwargs_validation,
):
    for _, bkt_id in enumerate(buckets):
        # kwargs_validation["dvectors"][bkt_id].eval()
        # kwargs_validation["contrastive_loss"][bkt_id].eval()

        sub_lbs_current_validation = outputs[bkt_id]

        sub_dataset_current_validation = kwargs_validation["SubDatasetSpk"](
            kwargs_validation["dataset_val"],
            sub_lbs_current_validation,
        )
        validation_loader_current = DataLoader(
            sub_dataset_current_validation,
            batch_size=len(sub_lbs_current_validation),
            collate_fn=kwargs_validation["collateSpkr"],
            drop_last=True,
        )

        mel_db_batch_validation = next(iter(validation_loader_current))

        x_val, spk_val = mel_db_batch_validation

        spk_val_filtered = spk_val
        x_val_filtered = x_val

        if args.delta and args.delta_delta:
            feat_dim_processed = args.feature_dim * 3
        elif args.delta:
            feat_dim_processed = args.feature_dim * 2
        else:
            feat_dim_processed = args.feature_dim

        x_val_filtered = x_val_filtered.reshape(-1, args.seg_len, feat_dim_processed)
        x_val_filtered, spk_val_filtered = x_val_filtered.to(
            device
        ), spk_val_filtered.to(device)

        xe_val = kwargs_validation["dvectors"][bkt_id](x_val_filtered).detach()

        loss = kwargs_validation["contrastive_loss"][bkt_id](
            xe_val.view(args.spk_per_bucket, -1, args.dim_emb)
        )

        cos_sim_matrix = kwargs_validation["contrastive_loss"][
            bkt_id
        ].compute_similarity_matrix(xe_val.view(args.spk_per_bucket, -1, args.dim_emb))
        acc = kwargs_validation["contrastive_loss"][bkt_id].calc_acc(cos_sim_matrix)

        if args.log_validation:
            loss_display = loss.item()
            acc_display = acc.item()

            epoch_display = f"Eval Epoch: {epoch}| "
            bucket_display = f"Bucket:{bkt_id}| "

            val_loss_display = f"valLoss:{loss_display:0.3f}| "
            val_acc_display = f"valAcc:{acc_display:0.3f}| "

            print(epoch_display, bucket_display, val_loss_display, val_acc_display)

        val_loss[bkt_id].append(loss.item())
        val_acc[bkt_id].append(acc.item())

    out = {"val_loss": val_loss, "val_acc": val_acc}

    return out
