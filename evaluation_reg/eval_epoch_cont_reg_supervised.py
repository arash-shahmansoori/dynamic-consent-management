import torch

from .eval_reg_overall_epoch_bkt import eval_reg_overall_per_round_per_epoch_per_bkt


def eval_reg_progressive_per_round_per_epoch_sup(
    args,
    device,
    outputs,
    buckets,
    opt_unique_bkt_sofar,
    indx_opt_unique_bkt_sofar,
    opt_unique_bkt,
    indx_opt_unique_bkt,
    epoch,
    val_acc_opt_bkt,
    **kwargs_validation,
):
    # Create validation data for evaluations
    xe_val_storage, spk_val_storage = [], []

    for _, bucket_id in enumerate(buckets):

        kwargs_validation["dvectors"][bucket_id].eval()

        eval_out = eval_reg_overall_per_round_per_epoch_per_bkt(
            args,
            device,
            outputs,
            bucket_id,
            opt_unique_bkt_sofar,
            indx_opt_unique_bkt_sofar,
            opt_unique_bkt,
            indx_opt_unique_bkt,
            **kwargs_validation,
        )

        # Store contrastive embeddings for validation
        xe_val_storage.append(eval_out["xe_val_cat"])
        spk_val_storage.append(eval_out["spk_val_cat"])

        x_val_buffer = torch.cat(xe_val_storage, dim=0).view(-1, args.dim_emb)
        t_val_buffer = torch.cat(spk_val_storage, dim=0).view(-1)

        val_acc, val_loss = kwargs_validation["agent"].accuracy_loss(
            kwargs_validation["classifier"],
            kwargs_validation["classifier_ma"],
            kwargs_validation["ce_loss"],
            x_val_buffer,
            t_val_buffer,
        )

        if args.log_training:

            loss_display = val_loss.item()
            acc_display = val_acc.item()

            epoch_display = f"Train Epoch: {epoch}| "
            if bucket_id == 0:
                bucket_display = f"Bucket:{0}| "
            else:
                bucket_display = f"Bucket:[{0}, {bucket_id}]| "

            val_loss_display = f"Loss:{loss_display:0.3f}| "
            val_acc_display = f"Acc:{acc_display:0.3f}| "

            print(epoch_display, bucket_display, val_loss_display, val_acc_display)

        val_acc_opt_bkt[bucket_id].append(val_acc.item())

    return val_acc_opt_bkt