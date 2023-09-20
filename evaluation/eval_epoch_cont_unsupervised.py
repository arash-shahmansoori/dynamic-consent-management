import torch
from torch.utils.data import DataLoader


def eval_per_epoch_progressive_contrastive_unsupervised(
    hparams,
    args,
    device,
    outputs,
    buckets,
    val_loss_cont,
    val_acc_cont,
    epoch,
    **kwargs_validation,
):
    kwargs_validation["dvec_latent"].eval()
    kwargs_validation["dvec_latent_ma"].eval()
    kwargs_validation["contrastive_loss_latent"].eval()

    xe_val_list, spk_val_list = [], []
    for _, bkt_id in enumerate(buckets):

        kwargs_validation["dvectors"][bkt_id].eval()

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

        t_val_buffer = torch.stack(spk_val_list, dim=0).view(-1)
        spks_per_buckets_sofar = len(t_val_buffer.unique(dim=0))

        x_val_buffer = torch.stack(xe_val_list, dim=0).view(
            spks_per_buckets_sofar, -1, args.dim_emb
        )

        if hparams.ma_mode == "swa":
            xe_val_out = kwargs_validation["dvec_latent_ma"](
                x_val_buffer.view(-1, args.dim_emb)
            ).detach()

        else:
            xe_val_out = kwargs_validation["dvec_latent"](
                x_val_buffer.view(-1, args.dim_emb)
            ).detach()

        loss_cont = kwargs_validation["contrastive_loss_latent"](
            xe_val_out.view(spks_per_buckets_sofar, -1, args.latent_dim)
        )

        cos_sim_matrix = kwargs_validation[
            "contrastive_loss_latent"
        ].compute_similarity_matrix(
            xe_val_out.view(spks_per_buckets_sofar, -1, args.latent_dim)
        )

        # acc_cont = kwargs_validation["contrastive_loss_latent"].calc_acc(
        #     cos_sim_matrix,
        #     x_val_buffer.shape[0],
        #     t_val_buffer % (spks_per_buckets_sofar),
        #     spks_per_buckets_sofar,
        # )

        acc_cont = kwargs_validation["contrastive_loss_latent"].calc_acc(
            cos_sim_matrix,
            t_val_buffer,
            spks_per_buckets_sofar,
        )

        if args.log_training:
            loss_cont_display = loss_cont.item() / len(t_val_buffer)
            acc_cont_display = acc_cont.item()

            epoch_display = f"Train Epoch: {epoch}| "
            if bkt_id == 0:
                bucket_display = f"Bucket:{0}| "
            else:
                bucket_display = f"Bucket:[{0}, {bkt_id}]| "

            val_loss_display = f"Loss:{loss_cont_display:0.3f}| "
            val_acc_display = f"Acc:{acc_cont_display:0.3f}| "

            print(epoch_display, bucket_display, val_loss_display, val_acc_display)

        val_loss_cont[bkt_id].append(loss_cont.item() / len(t_val_buffer))
        val_acc_cont[bkt_id].append(acc_cont.item())

    out = {"val_loss_cont": val_loss_cont, "val_acc_cont": val_acc_cont}

    return out
