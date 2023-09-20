from torch.utils.data import DataLoader


def eval_scratch_per_epoch_unsup(
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
        collate_fn=kwargs_validation["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch_validation = next(iter(validation_loader_current))

    x_val, _, spk_val = mel_db_batch_validation
    x_val = x_val.reshape(-1, args.seg_len, args.feature_dim)
    x_val, spk_val = x_val.to(device), spk_val.to(device)

    input_data = {"x_val": x_val, "y_val": spk_val}

    val_acc, val_loss = eval_metrics(
        args,
        model_dvec_latent,
        input_data,
        criterion,
    )

    if args.log_validation:
        loss_cont_display = val_loss
        acc_cont_display = val_acc.item()

        epoch_display = f"Eval Epoch: {epoch}| "
        bucket_display = f"Bucket:{0}| "

        val_loss_display = f"Loss:{loss_cont_display:0.3f}| "
        val_acc_display = f"Acc:{acc_cont_display:0.3f}| "

        print(epoch_display, bucket_display, val_loss_display, val_acc_display)

    val_acc_storage.append(val_acc.item())
    val_loss_storage.append(val_loss)

    out = {"val_acc": val_acc_storage, "val_loss": val_loss_storage}

    return out
