from utils import custom_timer
from .train_unreg_epoch_bkt_unsup import train_unreg_per_epoch_per_bkt_unsup


@custom_timer
def train_unreg_per_epoch_unsup(
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

    agent_method = {
        "train_dvec": kwargs_training["agent"].train_dvec,
        "train_dvec_adapted": kwargs_training["agent"].train_dvec_proposed,
        "train_dvec_proposed": kwargs_training["agent"].train_dvec_proposed,
        "train_dvec_latent": kwargs_training["agent"].train_dvec_latent,
        "train_dvec_latent_adapted": kwargs_training[
            "agent"
        ].train_dvec_latent_proposed,
        "train_dvec_latent_proposed": kwargs_training[
            "agent"
        ].train_dvec_latent_proposed,
    }

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

        feat_props = train_unreg_per_epoch_per_bkt_unsup(
            hparams,
            args,
            device,
            updated_outputs,
            outputs,
            bucket_id,
            epoch,
            early_stopping_status,
            agent_method,
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
                permute_samples=False,
            )

            input_buffer = {"feat": stacked_x, "label": stacked_y}

            # Train the classifier using the contrastive embedding replay
            agent_method[hparams.train_dvec_latent_mode](
                kwargs_training["dvec_latent"],
                kwargs_training["dvec_latent_ma"],
                kwargs_training["opt_cls_type"],
                kwargs_training["contrastive_loss_latent"],
                input_buffer,
                epoch,
                kwargs_training["ma_n"],
                kwargs_training["filename_unreg"],
                kwargs_training["filename_dir_unreg"],
            )
