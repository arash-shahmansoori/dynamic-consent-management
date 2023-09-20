from torch.utils.data import DataLoader
from utils.utils_filter_labels import filter_spk_indx


def eval_scratch_per_epoch_unsup_vox(
    args,
    device,
    model_dvec_latent,
    criterion,
    val_loss_storage,
    val_acc_storage,
    eval_metrics,
    epoch,
    **kwargs_validation,
):
    model_dvec_latent.eval()

    validation_loader_current = DataLoader(
        kwargs_validation["dataset_val"],
        batch_size=args.n_speakers,
        collate_fn=kwargs_validation["collateSpkr"],
        drop_last=True,
    )

    mel_db_batch_validation = next(iter(validation_loader_current))

    x_val, spk_val = mel_db_batch_validation

    # spk_val_filtered = spk_val[filter_spk_indx(spk_val)]
    # x_val_filtered = x_val[filter_spk_indx(spk_val)]

    spk_val_filtered = spk_val
    x_val_filtered = x_val

    if args.delta and args.delta_delta:
        feat_dim_processed = args.feature_dim * 3
    elif args.delta:
        feat_dim_processed = args.feature_dim * 2
    else:
        feat_dim_processed = args.feature_dim

    x_val_filtered = x_val_filtered.reshape(-1, args.seg_len, feat_dim_processed)
    x_val_filtered, spk_val_filtered = x_val_filtered.to(device), spk_val_filtered.to(
        device
    )

    input_data = {"x_val": x_val_filtered, "y_val": spk_val_filtered}

    val_acc, val_loss = eval_metrics(
        args,
        model_dvec_latent,
        input_data,
        criterion,
    )

    loss_cont_display = val_loss
    acc_cont_display = val_acc.item()

    if args.log_validation:
        epoch_display = f"Eval Epoch: {epoch}| "
        bucket_display = f"Bucket:{0}| "

        val_loss_display = f"Loss:{loss_cont_display:0.3f}| "
        val_acc_display = f"Acc:{acc_cont_display:0.3f}| "

        print(
            epoch_display,
            bucket_display,
            val_loss_display,
            val_acc_display,
        )

    val_acc_storage.append(val_acc.item())
    val_loss_storage.append(val_loss)

    out = {
        "val_acc": val_acc_storage,
        "val_loss": val_loss_storage,
    }

    return out
