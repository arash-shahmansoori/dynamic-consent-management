import torch

from utils import custom_timer, label_normalizer_progressive
from .train_reg_epoch_bkt_unsup import train_reg_per_round_per_epoch_per_bkt_unsup


@custom_timer
def train_reg_per_round_per_epoch_unsup(
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
    spk_init = []
    spk_normalized_prev = torch.tensor([])

    for indx, bucket_id in enumerate(buckets):

        feat_props = train_reg_per_round_per_epoch_per_bkt_unsup(
            hparams,
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
            agent_method,
            **kwargs_training,
        )

        spk_normalized_progressive_per_bkt = label_normalizer_progressive(
            feat_props["label_bkt"],
            spk_normalized_prev,
            spk_init,
        )
        spk_normalized_prev = spk_normalized_progressive_per_bkt

        # # Create input buffer for unsupervised contrastive embedding replay
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
        #     permute_samples=False,
        # )

        # # B) Sequential strategy for creating the buffer including the new registrations
        # # stacked_x, stacked_y = feat_props["feat_bkt"], feat_props["label_bkt"]

        # A) Custom strategy for creating the buffer including the new registrations
        stacked_x, stacked_y = new_reg_buffer.inter_bucket_sample(
            lf_collection[indx],
            feat_props["feat_bkt"],
            spk_normalized_progressive_per_bkt,
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
            kwargs_training["filename"],
            kwargs_training["filename_reg"],
            kwargs_training["filename_dir_reg"],
        )
