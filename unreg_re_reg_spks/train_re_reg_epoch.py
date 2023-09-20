from utils import custom_timer
from .train_re_reg_epoch_bkt import train_re_reg_per_epoch_per_bkt


@custom_timer
def train_re_reg_per_epoch(
    hparams,
    args,
    device,
    outputs,
    outputs_updated,
    buckets,
    epoch,
    spk_per_bkt_storage,
    re_reg_buffer,
    **kwargs_training,
):

    utts_per_spk = re_reg_buffer.num_per_spk_utts_progressive_mem(
        spk_per_bkt_storage,
        hparams.num_of_buckets * [0],
    )
    lf_collection = re_reg_buffer.utt_index_per_bucket_collection(
        spk_per_bkt_storage,
        hparams.num_of_buckets * [0],
        utts_per_spk,
    )

    early_stopping_status = []
    feats_init, labels_init = [], []

    for indx, bucket_id in enumerate(buckets):

        feat_props = train_re_reg_per_epoch_per_bkt(
            args,
            device,
            outputs,
            outputs_updated,
            bucket_id,
            epoch,
            early_stopping_status,
            **kwargs_training,
        )

        if feat_props["feat_bkt"] != None:

            # Custom strategy for creating the buffer
            stacked_x, stacked_y = re_reg_buffer.inter_bucket_sample(
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
                kwargs_training["filename_dir_re_reg"],
            )
