import torch
from torch.utils.data import DataLoader
from utils.utils_filter_labels import filter_spk_indx


def eval_per_epoch_progressive_contrastive_unsupervised_vox(
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

        t_val_buffer = torch.cat(spk_val_list, dim=0).view(-1)
        x_val_buffer = torch.cat(xe_val_list, dim=0).view(-1, args.dim_emb)

        if hparams.ma_mode == "swa":
            xe_out = kwargs_validation["dvec_latent_ma"](
                x_val_buffer.view(-1, args.dim_emb)
            ).detach()
        else:
            xe_out = kwargs_validation["dvec_latent"](
                x_val_buffer.view(-1, args.dim_emb)
            ).detach()

        spks_per_buckets_sofar = len(t_val_buffer.unique(dim=0))
        utt_per_spks = xe_out.shape[0] // spks_per_buckets_sofar

        xe_out_reformed = xe_out[: utt_per_spks * spks_per_buckets_sofar]
        t_val_buffer_reformed = t_val_buffer[: utt_per_spks * spks_per_buckets_sofar]

        x_out_reshaped = xe_out_reformed.reshape(
            (spks_per_buckets_sofar, utt_per_spks, args.latent_dim)
        )

        loss = kwargs_validation["contrastive_loss_latent"](x_out_reshaped)

        cos_sim_matrix = kwargs_validation[
            "contrastive_loss_latent"
        ].compute_similarity_matrix(x_out_reshaped)

        acc = kwargs_validation["contrastive_loss_latent"].calc_acc(
            cos_sim_matrix,
            t_val_buffer_reformed,
            spks_per_buckets_sofar,
        )

        if args.log_validation:
            loss_cont_display = loss.item()
            acc_cont_display = acc.item()

            epoch_display = f"Eval Epoch: {epoch}| "
            if bkt_id == 0:
                bucket_display = f"Bucket:{0}| "
            else:
                bucket_display = f"Bucket:[{0}, {bkt_id}]| "

            val_loss_display = f"Loss:{loss_cont_display:0.3f}| "
            val_acc_display = f"Acc:{acc_cont_display:0.3f}| "

            print(
                epoch_display,
                bucket_display,
                val_loss_display,
                val_acc_display,
            )

        val_loss_cont[bkt_id].append(loss.item())
        val_acc_cont[bkt_id].append(acc.item())

    out = {
        "val_loss_cont": val_loss_cont,
        "val_acc_cont": val_acc_cont,
    }

    return out
