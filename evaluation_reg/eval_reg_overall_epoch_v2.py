import torch

from .eval_reg_overall_epoch_bkt import eval_reg_overall_per_round_per_epoch_per_bkt


def eval_reg_overall_per_round_per_epoch_v2(
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
    val_acc_opt_bkt_old,
    **kwargs_validation,
):
    # Create validation data for evaluations
    xe_val_list, spk_val_list = [], []
    xe_val_list_old, spk_val_list_old = [], []

    for _, bucket_id in enumerate(buckets):

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

        indx_selected_spk_overall = eval_out["indx_selected_new_spks_overall"]
        indx_selected = eval_out["indx_selected"]
        _indx_selected_spk = -1

        # Store contrastive embeddings for validation
        xe_val_list_old.append(eval_out["xe_val_old"])
        spk_val_list_old.append(eval_out["spk_val_old"])

        x_val_buffer_old = torch.cat(xe_val_list_old, dim=0).view(-1, args.dim_emb)
        t_val_buffer_old = torch.cat(spk_val_list_old, dim=0).view(-1)

        val_acc_old, val_loss_old = kwargs_validation["agent"].accuracy_loss(
            kwargs_validation["dvec_latent"],
            kwargs_validation["dvec_latent_ma"],
            kwargs_validation["contrastive_loss_latent"],
            x_val_buffer_old,
            t_val_buffer_old,
        )

        val_acc_opt_bkt_old[f"{_indx_selected_spk}_{bucket_id}"].append(
            val_acc_old.item()
        )

        if len(eval_out["xe_val_new"]):

            xe_val_list.append(eval_out["xe_val_cat"])
            spk_val_list.append(eval_out["spk_val_cat"])

            x_val_buffer = torch.cat(xe_val_list, dim=0).view(-1, args.dim_emb)
            t_val_buffer = torch.cat(spk_val_list, dim=0).view(-1)

            val_acc, val_loss = kwargs_validation["agent"].accuracy_loss(
                kwargs_validation["dvec_latent"],
                kwargs_validation["dvec_latent_ma"],
                kwargs_validation["contrastive_loss_latent"],
                x_val_buffer,
                t_val_buffer,
            )

            val_acc_opt_bkt[f"{indx_selected_spk_overall}_{indx_selected}"].append(
                val_acc.item()
            )

        if args.log_training:

            epoch_display = f"Train Epoch: {epoch}| "
            if bucket_id in opt_unique_bkt:
                bucket_display = f"Bkt-Opt-New:{bucket_id}| "
                val_acc_display = f"AccNew:{val_acc:0.3f}| "
            elif bucket_id in opt_unique_bkt_sofar:
                bucket_display = f"Bkt-Opt-Sofar:{bucket_id}| "
                val_acc_display = f"AccSofar:{val_acc:0.3f}| "
            else:
                bucket_display = f"Bkt-Opt-Old:{bucket_id}| "
                val_acc_display = f"AccOld:{val_acc_old:0.3f}| "

            print(epoch_display, bucket_display, val_acc_display)

    out = {
        "val_acc_old": val_acc_opt_bkt_old,
        "val_acc": val_acc_opt_bkt,
        "val_loss": val_loss if len(eval_out["xe_val_new"]) else val_loss_old,
    }

    return out