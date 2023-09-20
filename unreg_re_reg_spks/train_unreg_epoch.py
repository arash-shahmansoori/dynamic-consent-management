from utils import custom_timer
from .train_unreg_epoch_bkt import train_unreg_per_epoch_per_bkt


@custom_timer
def train_unreg_per_epoch(
    hparams,
    args,
    device,
    updated_outputs,
    outputs,
    buckets,
    epoch,
    spk_per_bkt_storage,
    unreg_buffer,
    **kwargs_training,
):

    utts_per_spk = unreg_buffer.num_per_spk_utts_progressive_mem(
        spk_per_bkt_storage,
        hparams.num_of_buckets * [0],
    )
    lf_collection = unreg_buffer.utt_index_per_bucket_collection(
        spk_per_bkt_storage,
        hparams.num_of_buckets * [0],
        utts_per_spk,
    )

    early_stopping_status = []
    feats_init, labels_init = [], []

    for indx, bucket_id in enumerate(buckets):

        if spk_per_bkt_storage[indx] != 0:

            feat_props = train_unreg_per_epoch_per_bkt(
                args,
                device,
                updated_outputs,
                outputs,
                bucket_id,
                epoch,
                early_stopping_status,
                **kwargs_training,
            )

            if feat_props["feat_bkt"] != None:

                # Custom strategy for creating the buffer
                stacked_x, stacked_y = unreg_buffer.inter_bucket_sample(
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
                    kwargs_training["filename_dir_unreg"],
                )
