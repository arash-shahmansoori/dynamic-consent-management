import torch

from torch.utils.data import DataLoader
from utils import normalize_per_bkt_labels, customized_labels_unreg_unsup_bkt


def eval_per_epoch_per_bucket_contrastive_unsupervised_unreg(
    hparams,
    args,
    device,
    updated_outputs,
    outputs,
    buckets,
    val_loss,
    val_acc,
    epoch,
    **kwargs_validation,
):

    for _, bkt_id in enumerate(buckets):

        kwargs_validation["dvectors"][bkt_id].eval()

        if len(updated_outputs[bkt_id]) != 0:
            updated_outputs_bkt = updated_outputs[bkt_id]
            outputs_bkt = outputs[bkt_id]
            updated_out_bkt, removed_out_bkt = customized_labels_unreg_unsup_bkt(
                updated_outputs_bkt,
                outputs_bkt,
            )
            sub_lbs_current_validation_updated = updated_out_bkt
        elif len(updated_outputs[bkt_id]) == 0:
            outputs_bkt = outputs[bkt_id]
            sub_lbs_current_validation_updated = outputs_bkt

        sub_dataset_current_validation_updated = kwargs_validation["SubDatasetGdrSpk"](
            kwargs_validation["dataset_val"],
            sub_lbs_current_validation_updated,
        )
        validation_loader_current_updated = DataLoader(
            sub_dataset_current_validation_updated,
            batch_size=len(sub_lbs_current_validation_updated),
            collate_fn=kwargs_validation["collateGdrSpkr"],
            drop_last=True,
        )

        mel_db_batch_validation_updated = next(iter(validation_loader_current_updated))

        x_val_updated, _, spk_val_updated = mel_db_batch_validation_updated
        x_val_updated = x_val_updated.reshape(-1, args.seg_len, args.feature_dim)
        x_val_updated, spk_val_updated = (
            x_val_updated.to(device),
            spk_val_updated.to(device),
        )

        if len(removed_out_bkt) != 0 and len(updated_outputs[bkt_id]) != 0:
            sub_lbs_current_validation_removed = removed_out_bkt

            sub_dataset_current_validation_removed = kwargs_validation[
                "SubDatasetGdrSpk"
            ](
                kwargs_validation["dataset_val"],
                sub_lbs_current_validation_removed,
            )
            validation_loader_current_removed = DataLoader(
                sub_dataset_current_validation_removed,
                batch_size=len(sub_lbs_current_validation_removed),
                collate_fn=kwargs_validation["collateGdrSpkr"],
                drop_last=True,
            )

            mel_db_batch_validation_removed = next(
                iter(validation_loader_current_removed)
            )

            x_val_removed, _, spk_val_removed = mel_db_batch_validation_removed
            x_val_removed = x_val_removed.reshape(-1, args.seg_len, args.feature_dim)
            x_val_removed, spk_val_removed = (
                x_val_removed.to(device),
                spk_val_removed.to(device),
            )

            x_val = torch.cat((x_val_updated, x_val_removed))
            spk_val = torch.cat((spk_val_updated, spk_val_removed))
        else:
            x_val = x_val_updated
            spk_val = spk_val_updated

        spk_val_normalized = normalize_per_bkt_labels(spk_val.view(-1))
        xe_val = kwargs_validation["dvectors"][bkt_id](x_val).detach()

        if hparams.ma_mode == "swa":
            xe_val_out = kwargs_validation["dvec_latent_ma"](
                xe_val.view(-1, args.dim_emb)
            ).detach()

        else:
            xe_val_out = kwargs_validation["dvec_latent"](
                xe_val.view(-1, args.dim_emb)
            ).detach()

        loss_cont = kwargs_validation["contrastive_loss_latent"](
            xe_val_out.view(len(spk_val.unique(dim=0)), -1, args.latent_dim)
        )

        cos_sim_matrix = kwargs_validation[
            "contrastive_loss_latent"
        ].compute_similarity_matrix(
            xe_val_out.view(len(spk_val.unique(dim=0)), -1, args.latent_dim)
        )

        acc_cont = kwargs_validation["contrastive_loss_latent"].calc_acc(
            cos_sim_matrix,
            spk_val_normalized,
            len(spk_val.unique(dim=0)),
        )

        if args.log_training:

            loss_cont_display = loss_cont.item() / len(spk_val_normalized)
            acc_cont_display = acc_cont.item()

            epoch_display = f"Train Epoch: {epoch}| "
            if bkt_id == 0:
                bucket_display = f"Bucket:{0}| "
            else:
                bucket_display = f"Bucket:{bkt_id}| "

            val_loss_display = f"Loss:{loss_cont_display:0.3f}| "
            val_acc_display = f"Acc:{acc_cont_display:0.3f}| "

            print(epoch_display, bucket_display, val_loss_display, val_acc_display)

        val_loss[bkt_id].append(loss_cont.item() / len(spk_val_normalized))
        val_acc[bkt_id].append(acc_cont.item())

    out = {
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    return out
