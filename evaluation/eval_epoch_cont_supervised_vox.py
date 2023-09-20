import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils_filter_labels import filter_spk_indx


# def set_n_training(m):
#     if isinstance(m, nn.LayerNorm):
#         m.train()


def eval_per_epoch_progressive_contrastive_supervised_vox(
    args,
    device,
    outputs,
    buckets,
    val_loss,
    val_acc,
    epoch,
    **kwargs_validation,
):
    xe_val_list, spk_val_list = [], []
    for _, bkt_id in enumerate(buckets):
        kwargs_validation["dvectors"][bkt_id].eval()
        # kwargs_validation["dvectors"][bkt_id].apply(set_n_training)

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
        x_val_filtered, spk_val_filtered = x_val_filtered.to(
            device
        ), spk_val_filtered.to(device)

        xe_val = kwargs_validation["dvectors"][bkt_id](x_val_filtered).detach()

        # Store contrastive embeddings for validation
        xe_val_list.append(xe_val)
        spk_val_list.append(spk_val_filtered)

        x_val_buffer = torch.cat(xe_val_list, dim=0).view(-1, args.dim_emb)
        t_val_buffer = torch.cat(spk_val_list, dim=0).view(-1)

        # Compute performance measures of the classifier
        acc, loss = kwargs_validation["agent"].accuracy_loss(
            kwargs_validation["classifier"],
            kwargs_validation["classifier_ma"],
            kwargs_validation["ce_loss"],
            x_val_buffer,
            t_val_buffer,
        )

        if args.log_validation:
            loss_display = loss.item()
            acc_display = acc.item()

            epoch_display = f"Eval Epoch: {epoch}| "
            if bkt_id == 0:
                bucket_display = f"Bucket:{0}| "
            else:
                bucket_display = f"Bucket:[{0}, {bkt_id}]| "

            val_loss_display = f"valLoss:{loss_display:0.3f}| "
            val_acc_display = f"valAcc:{acc_display:0.3f}| "

            print(epoch_display, bucket_display, val_loss_display, val_acc_display)

        val_loss[bkt_id].append(loss.item())
        val_acc[bkt_id].append(acc.item())

    out = {"val_loss": val_loss, "val_acc": val_acc}

    return out
