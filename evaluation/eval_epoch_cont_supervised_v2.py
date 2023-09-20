import torch
from torch.utils.data import DataLoader


def eval_per_epoch_progressive_contrastive_supervised_v2(
    args,
    device,
    outputs,
    buckets,
    val_loss,
    val_acc,
    logger,
    epoch,
    **kwargs_validation,
):

    xe_val_list, spk_val_list = [], []
    for _, bkt_id in enumerate(buckets):

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
        xe_val_list.append(xe_val)
        spk_val_list.append(spk_val)

        x_val_buffer = torch.stack(xe_val_list, dim=0).view(-1, args.dim_emb)
        t_val_buffer = torch.stack(spk_val_list, dim=0).view(-1)

        # Compute performance measures of the classifier
        acc, loss = kwargs_validation["agent"].accuracy_loss(
            kwargs_validation["dvec_latent"],
            kwargs_validation["dvec_latent_ma"],
            kwargs_validation["contrastive_loss_latent"],
            x_val_buffer,
            t_val_buffer,
        )

        if args.log_training:
            logger.info(
                "Train Epoch: {}| Buckets:{}| Loss:{:0.3f}| Acc:{:0.3f}|".format(
                    epoch,
                    bkt_id,
                    loss.item(),
                    acc.item(),
                ),
            )

        val_loss[bkt_id].append(loss.item())
        val_acc[bkt_id].append(acc.item())

    out = {"val_loss": val_loss, "val_acc": val_acc}

    return out
