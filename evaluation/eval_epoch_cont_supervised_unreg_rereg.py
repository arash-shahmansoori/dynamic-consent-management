import torch
from torch.utils.data import DataLoader


def eval_per_epoch_progressive_contrastive_supervised_unreg_rereg(
    args,
    device,
    outputs,
    unreg_bkts_storage,
    buckets,
    val_loss,
    val_acc,
    epoch,
    **kwargs_validation,
):

    xe_val_storage, spk_val_storage = [], []
    for _, bkt_id in enumerate(buckets):

        kwargs_validation["dvectors"][bkt_id].eval()

        sub_lbs_current_validation = outputs[bkt_id]

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

        if bkt_id in unreg_bkts_storage:

            x_val_unreg_buffer = xe_val.view(-1, args.dim_emb)
            t_val_unreg_buffer = spk_val.view(-1)

            # Compute performance measures of the classifier
            acc, loss = kwargs_validation["agent"].accuracy_loss(
                kwargs_validation["classifier"],
                kwargs_validation["classifier_ma"],
                kwargs_validation["ce_loss"],
                x_val_unreg_buffer,
                t_val_unreg_buffer,
            )

            val_loss[bkt_id].append(loss.item())
            val_acc[bkt_id].append(acc.item())

            bucket_display = f"Bucket:{bkt_id}| "

        else:

            # Store contrastive embeddings for validation
            xe_val_storage.append(xe_val)
            spk_val_storage.append(spk_val)

            x_val_buffer = torch.cat(xe_val_storage, dim=0).view(-1, args.dim_emb)
            t_val_buffer = torch.cat(spk_val_storage, dim=0).view(-1)

            # Compute performance measures of the classifier
            acc, loss = kwargs_validation["agent"].accuracy_loss(
                kwargs_validation["classifier"],
                kwargs_validation["classifier_ma"],
                kwargs_validation["ce_loss"],
                x_val_buffer,
                t_val_buffer,
            )

            val_loss[bkt_id].append(loss.item())
            val_acc[bkt_id].append(acc.item())

            bucket_display = f"Bucket:[{0}, {bkt_id}]\{unreg_bkts_storage}| "

        if args.log_training:

            loss_display = loss.item()
            acc_display = acc.item()

            epoch_display = f"Train Epoch: {epoch}| "

            val_loss_display = f"Loss:{loss_display:0.3f}| "
            val_acc_display = f"Acc:{acc_display:0.3f}| "

            print(epoch_display, bucket_display, val_loss_display, val_acc_display)

    out = {"val_loss": val_loss, "val_acc": val_acc}

    return out
