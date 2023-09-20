from utils import custom_timer
from .train_reg_epoch_bkt import train_reg_per_round_per_epoch_per_bkt


@custom_timer
def train_reg_per_round_per_epoch(
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
    spk_per_bkt_storage,
    spk_per_bkt_reg_storage,
    new_reg_buffer,
    **kwargs_training,
):

    utts_per_spk = new_reg_buffer.num_per_spk_utts_progressive_mem(
        spk_per_bkt_storage,
        spk_per_bkt_reg_storage,
    )

    lf_collection = new_reg_buffer.utt_index_per_bucket_collection(
        spk_per_bkt_storage,
        spk_per_bkt_reg_storage,
        utts_per_spk,
    )

    early_stopping_status = []
    feats_init, labels_init = [], []

    for indx, bucket_id in enumerate(buckets):

        feat_props = train_reg_per_round_per_epoch_per_bkt(
            args,
            device,
            outputs,
            bucket_id,
            opt_unique_bkt_sofar,
            indx_opt_unique_bkt_sofar,
            opt_unique_bkt,
            indx_opt_unique_bkt,
            epoch,
            early_stopping_status,
            **kwargs_training,
        )

        # # Create input buffer for supervised contrastive embedding replay
        # n_utts_selected = new_reg_buffer.num_per_bucket_utts_per_epoch_mem(
        #     feat_props["num_spk_per_bkt"],
        #     feat_props["num_new_reg_bkt"],
        #     hparams.num_of_buckets,
        # )

        # lf = new_reg_buffer.utt_index_per_bucket(
        #     feat_props["num_spk_per_bkt"],
        #     feat_props["num_new_reg_bkt"],
        #     n_utts_selected,
        # )

        # # A) Custom strategy for creating the buffer including the new registrations
        # stacked_x, stacked_y = new_reg_buffer.inter_bucket_sample(
        #     lf,
        #     feat_props["feat_bkt"],
        #     feat_props["label_bkt"],
        #     feats_init,
        #     labels_init,
        # )

        # # B) Sequential strategy for creating the buffer including the new registrations
        # # stacked_x, stacked_y = feat_props["feat_bkt"], feat_props["label_bkt"]

        # A) Custom strategy for creating the buffer including the new registrations
        stacked_x, stacked_y = new_reg_buffer.inter_bucket_sample(
            lf_collection[indx],
            feat_props["feat_bkt"],
            feat_props["label_bkt"],
            feats_init,
            labels_init,
        )

        input_buffer = {"feat": stacked_x, "label": stacked_y}

        # Train the classifier using the contrastive embedding replay
        kwargs_training["agent"].train_cls(
            kwargs_training["classifier"],
            kwargs_training["classifier_ma"],
            kwargs_training["optimizer"],
            kwargs_training["ce_loss"],
            input_buffer,
            epoch,
            kwargs_training["ma_n"],
            kwargs_training["filename_dir_reg"],
        )
