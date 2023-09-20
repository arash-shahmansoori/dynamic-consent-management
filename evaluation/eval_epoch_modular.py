import torch
from torch.utils.data import DataLoader


def eval_per_epoch_modular(
    args,
    device,
    outputs,
    buckets,
    logger,
    epoch,
    val_acc,
    val_loss,
    **kwargs_validation,
):

    xe_val_list, spk_val_list = [], []
    for _, bucket_id in enumerate(buckets):

        sub_lbs_current_validation = outputs[bucket_id]

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

        xe_val = kwargs_validation["dvectors"][bucket_id](x_val).detach()

        # Store contrastive embeddings for validation
        xe_val_list.append(xe_val)
        spk_val_list.append(spk_val)

        x_val_buffer = torch.stack(xe_val_list, dim=0).view(-1, args.dim_emb)
        t_val_buffer = torch.stack(spk_val_list, dim=0).view(-1)

        # Compute performance measures of the classifier
        acc, loss = kwargs_validation["agent"].accuracy_loss(
            kwargs_validation["classifier"],
            kwargs_validation["ce_loss"],
            x_val_buffer,
            t_val_buffer,
        )

        # Earling stopping updates
        # if args.early_stopping:
        #     kwargs_validation["early_stopping"](val_loss)

        val_acc[bucket_id].append(acc.item())
        val_loss[bucket_id].append(loss.item())

        if args.log_training:
            logger.info(
                "Train Epoch: {}| Buckets:{}| Acc:{:0.3f}| Loss:{:0.3f}".format(
                    epoch,
                    bucket_id,
                    acc,
                    loss,
                )
            )

    out = {"val_acc": val_acc, "val_loss": val_loss}

    return out
