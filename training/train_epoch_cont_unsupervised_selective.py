from utils import custom_timer_with_return


from .train_selective_inductive_bias_unsupervised import (
    train_selective_inductive_bias_unsup,
)


@custom_timer_with_return
def train_per_epoch_contrastive_unsupervised_selective(
    hparams,
    args,
    device,
    outputs,
    buckets,
    logger,
    epoch,
    create_buffer,
    early_stopping,
    **kwargs_training_val,
):
    spk_per_bkt_storage = hparams.num_of_buckets * [args.spk_per_bucket]
    spk_per_bkt_reg_storage = hparams.num_of_buckets * [0]

    utts_per_spk = create_buffer.num_per_spk_utts_progressive_mem(
        spk_per_bkt_storage,
        spk_per_bkt_reg_storage,
    )
    lf_collection = create_buffer.utt_index_per_bucket_collection(
        spk_per_bkt_storage,
        spk_per_bkt_reg_storage,
        utts_per_spk,
    )

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

    feats_init, labels_init = [], []
    for indx, bucket_id in enumerate(buckets):
        # Generate selective inductive bias from the buffer stack (unsupervised)
        props = train_selective_inductive_bias_unsup(
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

        # Selective classification from the trained buffer stack (unsupervised)
        # if (
        #     not torch.tensor(props["early_stopping_bkt"][bucket_id])
        #     .view(-1)
        #     .item()
        # ):

        # Create contrastive latent feature per bucket
        xe = (
            kwargs_training_val["dvectors"][props["bucket_id_selected"]](props["x"])
            .view(-1, args.dim_emb)
            .detach()
        )

        ## Create speaker buffers for contrastive embedding replay (old version)
        # stacked_x_stored, stacked_y_stored = create_buffer.update(
        #     xe,
        #     props["y"],
        #     props["bucket_id_selected"],
        #     feats_init,
        #     labels_init,
        #     permute_buffer_=False,
        # )

        # stacked_x = torch.stack(stacked_x_stored, dim=0).view(-1, args.dim_emb)
        # stacked_y = torch.stack(stacked_y_stored, dim=0).view(-1)

        # Create input buffer for supervised contrastive embedding replay

        # A) Custom strategy for creating the buffer including the new registrations
        stacked_x, stacked_y = create_buffer.inter_bucket_sample(
            lf_collection[indx],
            xe,
            props["y"],
            feats_init,
            labels_init,
            permute_samples=False,
        )

        # B) Sequential strategy for creating the buffer including the new registrations
        # stacked_x, stacked_y = xe, props["y"]

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

    return props["early_stopping"]
