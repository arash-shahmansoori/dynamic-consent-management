import torch

from utils import (
    label_normalizer_per_bucket,
    label_normalizer_progressive,
    # progressive_indx_normalization,
    # Progressive_normalized_label,
)
from .eval_reg_overall_epoch_bkt_unsup import (
    eval_reg_overall_per_round_per_epoch_per_bkt_unsup,
)


def eval_reg_overall_per_round_per_epoch_unsup(
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
    # Create validation data for evaluations
    # xe_val_list, spk_val_list = [], []
    xe_val_list_old = []
    # spk_val_list_old = [], []

    spk_val_list_old = []
    # spk_val_list_old_ = []

    spk_init = []
    spk_normalized_prev = torch.tensor([])

    for _, bucket_id in enumerate(buckets):

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

        # indx_selected_spk_overall = eval_out["indx_selected_new_spks_overall"]
        # indx_selected = eval_out["indx_selected"]
        # _indx_selected_spk = -1

        # Store contrastive embeddings for validation
        # xe_val_list_old.append(eval_out["xe_val_cat"])
        # spk_val_list_old.append(eval_out["spk_val_cat"])

        # spk_normalized = progressive_indx_normalization(
        #     eval_out["spk_val_cat"].cpu(),
        #     args.spk_per_bucket,
        #     bucket_id,
        #     device,
        # )

        # spk_normalized_concat = Progressive_normalized_label(
        #     spk_normalized,
        #     bucket_id,
        # )

        spk_normalized = label_normalizer_per_bucket(eval_out["spk_val_cat"])
        spk_normalized_progressive_per_bkt = label_normalizer_progressive(
            spk_normalized,
            spk_normalized_prev,
            spk_init,
        )
        spk_normalized_prev = spk_normalized_progressive_per_bkt

        xe_val_list_old.append(eval_out["xe_val_cat"])
        spk_val_list_old.append(spk_normalized_progressive_per_bkt)

        # print(spk_normalized, eval_out["spk_val_cat"])

        # spk_val_list_old_.append(spk_normalized)

        t_val_buffer_old = torch.cat(spk_val_list_old, dim=0).reshape((-1))
        # t_val_buffer_old_ = torch.cat(spk_val_list_old_, dim=0).reshape((-1))

        # t_val_buffer_old = spk_normalized.reshape((-1))
        # x_val_buffer_old = eval_out["xe_val_cat"][
        #     : len(t_val_buffer_old), : args.dim_emb
        # ].reshape((len(t_val_buffer_old), args.dim_emb))

        # print(t_val_buffer_old)

        # print(len(t_val_buffer_old.unique(dim=0)), len(t_val_buffer_old_.unique(dim=0)))

        x_val_buffer_old = torch.cat(xe_val_list_old, dim=0)[
            : len(t_val_buffer_old), : args.dim_emb
        ].reshape((len(t_val_buffer_old), args.dim_emb))

        if hparams.ma_mode == "swa":
            xe_val_old = kwargs_validation["dvec_latent_ma"](x_val_buffer_old).detach()
        else:
            xe_val_old = kwargs_validation["dvec_latent"](x_val_buffer_old).detach()

        # val_loss_old = kwargs_validation["contrastive_loss_latent"](
        #     xe_val_old.reshape(
        #         (
        #             len(t_val_buffer_old.unique(dim=0)),
        #             len(t_val_buffer_old) // len(t_val_buffer_old.unique(dim=0)),
        #             args.latent_dim,
        #         )
        #     )
        # )

        cos_sim_matrix_old = kwargs_validation[
            "contrastive_loss_latent"
        ].compute_similarity_matrix(
            xe_val_old.reshape(
                (
                    len(t_val_buffer_old.unique(dim=0)),
                    len(t_val_buffer_old) // len(t_val_buffer_old.unique(dim=0)),
                    args.latent_dim,
                )
            )
        )

        cos_sim_matrix_combined = cos_sim_matrix_old
        batch_size_combined = x_val_buffer_old.shape[0]
        spk_per_bucket_combined = len(t_val_buffer_old.unique(dim=0))

        # val_acc_old = kwargs_validation["contrastive_loss_latent"].calc_acc(
        #     cos_sim_matrix_old,
        #     x_val_buffer_old.shape[0],
        #     t_val_buffer_old % spk_per_bucket_old,
        #     spk_per_bucket_old,
        # )

        # val_acc_opt_bkt_old[f"{_indx_selected_spk}_{bucket_id}"].append(
        #     val_acc_old.item()
        # )

        # if len(eval_out["xe_val_new"]):

        #     xe_val_list.append(eval_out["xe_val_cat"])
        #     spk_val_list.append(eval_out["spk_val_cat"])

        #     t_val_buffer = torch.cat(spk_val_list, dim=0).reshape((-1))
        #     x_val_buffer = torch.cat(xe_val_list, dim=0).reshape(
        #         (len(t_val_buffer), args.dim_emb)
        #     )

        #     spk_per_bucket_updated = len(t_val_buffer.unique(dim=0))

        #     if hparams.ma_mode == "swa":
        #         xe_val = kwargs_validation["dvec_latent_ma"](x_val_buffer).detach()
        #     else:
        #         xe_val = kwargs_validation["dvec_latent"](x_val_buffer).detach()

        #     val_loss = kwargs_validation["contrastive_loss_latent"](
        #         xe_val.reshape(
        #             (
        #                 spk_per_bucket_updated,
        #                 len(t_val_buffer) // spk_per_bucket_updated,
        #                 args.latent_dim,
        #             )
        #         )
        #     )

        #     cos_sim_matrix_updated = kwargs_validation[
        #         "contrastive_loss_latent"
        #     ].compute_similarity_matrix(
        #         xe_val.reshape(
        #             (
        #                 spk_per_bucket_updated,
        #                 len(t_val_buffer) // spk_per_bucket_updated,
        #                 args.latent_dim,
        #             )
        #         )
        #     )

        # t_val_buffer_combined = t_val_buffer
        # cos_sim_matrix_combined = cos_sim_matrix_updated
        # batch_size_combined = x_val_buffer.shape[0]
        # spk_per_bucket_combined = spk_per_bucket_updated

        #     val_acc = kwargs_validation["contrastive_loss_latent"].calc_acc(
        #         cos_sim_matrix_updated,
        #         x_val_buffer.shape[0],
        #         # t_val_buffer % (spk_per_bucket_updated),
        #         t_val_buffer,
        #         spk_per_bucket_updated,
        #     )

        #     val_acc_opt_bkt[f"{indx_selected_spk_overall}_{indx_selected}"].append(
        #         val_acc.item()
        #     )

        val_loss_combined = kwargs_validation["contrastive_loss_latent"](
            xe_val_old.view(
                len(t_val_buffer_old.unique(dim=0)),
                len(t_val_buffer_old) // len(t_val_buffer_old.unique(dim=0)),
                args.latent_dim,
            )
        )

        val_acc_combined = kwargs_validation["contrastive_loss_latent"].calc_acc(
            cos_sim_matrix_combined,
            batch_size_combined,
            t_val_buffer_old,
            spk_per_bucket_combined,
        )

        # print(torch.tensor(sorted(spk_normalized)))
        # print(bucket_id, val_acc_combined)

        if args.log_training:

            loss_cont_display = val_loss_combined.item() / len(t_val_buffer_old)

            epoch_display = f"Train Epoch: {epoch}| "
            if bucket_id == 0:
                bucket_display = f"Bucket:{0}| "
            else:
                bucket_display = f"Bucket:[{0}, {bucket_id}]| "

            val_loss_display = f"Loss:{loss_cont_display:0.3f}| "
            val_acc_display = f"Acc:{val_acc_combined:0.3f}| "

            # if bucket_id in opt_unique_bkt:
            #     bucket_display = f"Bkt-Opt-New:{bucket_id}| "
            #     val_acc_display = f"AccNew:{val_acc:0.3f}| "
            #     val_loss_display = f"ValLossNew:{val_loss:0.3f} "
            # elif bucket_id in opt_unique_bkt_sofar:
            #     bucket_display = f"Bkt-Opt-Sofar:{bucket_id}| "
            #     val_acc_display = f"AccSofar:{val_acc:0.3f}| "
            #     val_loss_display = f"ValLossSofar:{val_loss:0.3f} "
            # else:
            #     bucket_display = f"Bkt-Opt-Old:{bucket_id}| "
            #     val_acc_display = f"AccOld:{val_acc_old:0.3f}| "
            #     val_loss_display = f"ValLossOld:{val_loss_old:0.3f}"

            # print(epoch_display, bucket_display, val_acc_display, val_loss_display)

            print(epoch_display, bucket_display, val_loss_display, val_acc_display)

        val_acc_opt_bkt[bucket_id].append(val_acc_combined.item())

    # out = {
    #     "val_acc_old": val_acc_opt_bkt_old,
    #     "val_acc": val_acc_opt_bkt,
    #     # "val_loss": val_loss if len(eval_out["xe_val_new"]) else val_loss_old,
    # }

    return val_acc_opt_bkt