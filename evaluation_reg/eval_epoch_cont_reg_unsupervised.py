import torch

from utils import (
    label_normalizer_per_bucket,
    label_normalizer_progressive,
)
from .eval_reg_overall_epoch_bkt_unsup import (
    eval_reg_overall_per_round_per_epoch_per_bkt_unsup,
)


def eval_reg_progressive_per_round_per_epoch_unsup(
    hparams,
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

    xe_val_storage = []
    spk_val_storage = []

    spk_init = []
    spk_normalized_prev = torch.tensor([])

    kwargs_validation["dvec_latent"].eval()
    kwargs_validation["dvec_latent_ma"].eval()
    kwargs_validation["contrastive_loss_latent"].eval()

    for _, bucket_id in enumerate(buckets):

        kwargs_validation["dvectors"][bucket_id].eval()

        eval_out = eval_reg_overall_per_round_per_epoch_per_bkt_unsup(
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

        spk_normalized = label_normalizer_per_bucket(eval_out["spk_val_cat"])
        spk_normalized_progressive_per_bkt = label_normalizer_progressive(
            spk_normalized,
            spk_normalized_prev,
            spk_init,
        )
        spk_normalized_prev = spk_normalized_progressive_per_bkt

        xe_val_storage.append(eval_out["xe_val_cat"])
        spk_val_storage.append(spk_normalized_progressive_per_bkt)

        t_val_buffer = torch.cat(spk_val_storage, dim=0).reshape((-1))
        x_val_buffer = torch.cat(xe_val_storage, dim=0)[
            : len(t_val_buffer), : args.dim_emb
        ].reshape((len(t_val_buffer), args.dim_emb))

        if hparams.ma_mode == "swa":
            xe_val = kwargs_validation["dvec_latent_ma"](x_val_buffer).detach()
        else:
            xe_val = kwargs_validation["dvec_latent"](x_val_buffer).detach()

        cos_sim_matrix = kwargs_validation[
            "contrastive_loss_latent"
        ].compute_similarity_matrix(
            xe_val.reshape(
                (
                    len(t_val_buffer.unique(dim=0)),
                    len(t_val_buffer) // len(t_val_buffer.unique(dim=0)),
                    args.latent_dim,
                )
            )
        )

        cos_sim_matrix_combined = cos_sim_matrix
        spk_per_bucket_combined = len(t_val_buffer.unique(dim=0))

        val_loss_combined = kwargs_validation["contrastive_loss_latent"](
            xe_val.view(
                len(t_val_buffer.unique(dim=0)),
                len(t_val_buffer) // len(t_val_buffer.unique(dim=0)),
                args.latent_dim,
            )
        )

        val_acc_combined = kwargs_validation["contrastive_loss_latent"].calc_acc(
            cos_sim_matrix_combined,
            t_val_buffer,
            spk_per_bucket_combined,
        )

        if args.log_training:

            loss_cont_display = val_loss_combined.item() / len(t_val_buffer)

            epoch_display = f"Train Epoch: {epoch}| "
            if bucket_id == 0:
                bucket_display = f"Bucket:{0}| "
            else:
                bucket_display = f"Bucket:[{0}, {bucket_id}]| "

            val_loss_display = f"Loss:{loss_cont_display:0.3f}| "
            val_acc_display = f"Acc:{val_acc_combined:0.3f}| "

            print(epoch_display, bucket_display, val_loss_display, val_acc_display)

        val_acc_opt_bkt[bucket_id].append(val_acc_combined.item())

    return val_acc_opt_bkt