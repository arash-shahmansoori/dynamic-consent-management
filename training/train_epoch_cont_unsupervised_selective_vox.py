from utils import custom_timer_with_return
from .train_selective_inductive_bias_unsupervised_vox import (
    train_selective_inductive_bias_unsup_vox,
)


@custom_timer_with_return
def train_per_epoch_contrastive_unsupervised_selective_vox(
    hparams,
    args,
    device,
    outputs,
    buckets,
    logger,
    epoch,
    train_loss,
    train_acc,
    create_buffer,
    early_stopping,
    **kwargs_training_val,
):
    # Create index collection according to a given strategy: (1) & (2)
    lf_collection, feats_init, labels_init = create_buffer.create_collect_indx(
        args, buckets
    )  # (1)
    # (
    #     lf_collect_progress,
    #     bkt_samples,
    #     bkt_labels,
    # ) = create_buffer.create_progressive_collect_indx(args, buckets) # (2)

    agent_method = {
        "train_dvec": kwargs_training_val["agent"].train_dvec,
        "train_dvec_adapted": kwargs_training_val["agent"].train_dvec_proposed,
        "train_dvec_proposed": kwargs_training_val["agent"].train_dvec_proposed,
        "train_dvec_latent": kwargs_training_val["agent"].train_dvec_latent,
        "train_dvec_latent_adapted": kwargs_training_val[
            "agent"
        ].train_dvec_latent_proposed,
        "train_dvec_latent_proposed": kwargs_training_val[
            "agent"
        ].train_dvec_latent_proposed,
    }

    early_stop_status = []
    early_stop_status_bkt = {bucket_id: [] for _, bucket_id in enumerate(buckets)}

    # feats_init, labels_init = [], []
    for indx, bucket_id in enumerate(buckets):
        # Generate selective inductive bias from the buffer stack (unsupervised)
        props = train_selective_inductive_bias_unsup_vox(
            args,
            outputs,
            device,
            agent_method,
            hparams.train_dvec_mode,
            early_stopping[bucket_id],
            early_stop_status,
            early_stop_status_bkt,
            bucket_id,
            epoch,
            logger,
            kwargs_training_val,
        )

        # Create contrastive latent feature per bucket
        xe = (
            kwargs_training_val["dvectors"][props["bucket_id_selected"]](props["x"])
            .view(-1, args.dim_emb)
            .detach()
        )

        # Custom strategy (1) for creating the buffer including the new registrations
        stacked_x, stacked_y = create_buffer.inter_bucket_sample(
            lf_collection[indx],
            xe,
            props["y"],
            feats_init,
            labels_init,
            permute_samples=False,
        )

        # Custom strategy (2) for creating the buffer including the new registrations
        # bkt_samples[str(indx)], bkt_labels[str(indx)] = xe, props["y"]
        # stacked_x, stacked_y = create_buffer.inter_bucket_sample_v2(
        #     indx,
        #     lf_collect_progress,
        #     bkt_samples,
        #     bkt_labels,
        #     device,
        #     permute_samples=False,
        # )

        input_buffer = {"feat": stacked_x, "label": stacked_y}

        # Selectively train the latent dvec net
        agent_method[hparams.train_dvec_latent_mode](
            kwargs_training_val["dvec_latent"],
            kwargs_training_val["dvec_latent_ma"],
            kwargs_training_val["opt_cls_type"],
            kwargs_training_val["contrastive_loss_latent"],
            input_buffer,
            epoch,
            kwargs_training_val["ma_n"],
            kwargs_training_val["filename"],
            kwargs_training_val["filename_dir"],
        )

        if hparams.ma_mode == "swa":
            xe_out = kwargs_training_val["dvec_latent_ma"](
                input_buffer["feat"].view(-1, args.dim_emb)
            ).detach()
        else:
            xe_out = kwargs_training_val["dvec_latent"](
                input_buffer["feat"].view(-1, args.dim_emb)
            ).detach()

        spks_per_buckets_sofar = len(input_buffer["label"].unique(dim=0))

        train_loss_cont = kwargs_training_val["contrastive_loss_latent"](
            xe_out.view(spks_per_buckets_sofar, -1, args.latent_dim)
        )

        cos_sim_matrix = kwargs_training_val[
            "contrastive_loss_latent"
        ].compute_similarity_matrix(
            xe_out.view(spks_per_buckets_sofar, -1, args.latent_dim)
        )

        train_acc_cont = kwargs_training_val["contrastive_loss_latent"].calc_acc(
            cos_sim_matrix,
            input_buffer["label"],
            spks_per_buckets_sofar,
        )

        train_loss[bucket_id].append(train_loss_cont.item())
        train_acc[bucket_id].append(train_acc_cont.item())

        if args.log_training:
            loss_display = train_loss_cont.item()
            acc_display = train_acc_cont.item()

            epoch_display = f"Train Epoch: {epoch}| "
            if bucket_id == 0:
                bucket_display = f"Bucket:{0}| "
            else:
                bucket_display = f"Bucket:[{0}, {bucket_id}]| "

            loss_display = f"Loss:{loss_display:0.3f}| "
            acc_display = f"Acc:{acc_display:0.3f}| "

            print(epoch_display, bucket_display, loss_display, acc_display)

    out = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "early_stops_status": props["early_stopping"],
    }

    return out
