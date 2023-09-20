import torch

from utils import custom_timer_with_return
from .train_selective_inductive_bias_supervised import (
    train_selective_inductive_bias_sup,
)


@custom_timer_with_return
def train_per_epoch_contrastive_supervised_selective(
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
    **kwargs_training,
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

    early_stop_status = []
    early_stop_status_bkt = {bucket_id: [] for _, bucket_id in enumerate(buckets)}

    feats_init, labels_init = [], []
    for indx, bucket_id in enumerate(buckets):

        # Generate selective inductive bias from the buffer stack (supervised)
        props = train_selective_inductive_bias_sup(
            args,
            outputs,
            device,
            kwargs_training["early_stop"][bucket_id],
            early_stop_status,
            early_stop_status_bkt,
            bucket_id,
            epoch,
            logger,
            kwargs_training,
        )

        # Create contrastive latent feature
        xe = kwargs_training["dvectors"][bucket_id](props["x"]).detach()

        # # Create speaker buffers for contrastive embedding replay (old version)
        # stacked_x_stored, stacked_y_stored = create_buffer.update(
        #     xe,
        #     props["y"],
        #     props["bucket_id_selected"],
        #     feats_init,
        #     labels_init,
        #     permute_buffer_=True,
        # )

        # x_buffer = torch.stack(stacked_x_stored, dim=0).view(-1, args.dim_emb)
        # t_buffer = torch.stack(stacked_y_stored, dim=0).view(-1)

        # Create input buffer for supervised contrastive embedding replay

        # A) Custom strategy for creating the buffer including the new registrations
        stacked_x, stacked_y = create_buffer.inter_bucket_sample(
            lf_collection[indx],
            xe,
            props["y"],
            feats_init,
            labels_init,
        )

        # B) Sequential strategy for creating the buffer including the new registrations
        # stacked_x, stacked_y = xe, props["y"]

        input_buffer = {"feat": stacked_x, "label": stacked_y}

        # Train the classifier
        kwargs_training["agent"].train_cls(
            kwargs_training["classifier"],
            kwargs_training["classifier_ma"],
            kwargs_training["optimizer"],
            kwargs_training["ce_loss"],
            input_buffer,
            epoch,
            kwargs_training["ma_n"],
            kwargs_training["filename_dir"],
        )

        acc, loss = kwargs_training["agent"].accuracy_loss(
            kwargs_training["classifier"],
            kwargs_training["classifier_ma"],
            kwargs_training["ce_loss"],
            input_buffer["feat"],
            input_buffer["label"],
        )

        train_loss[bucket_id].append(loss.item())
        train_acc[bucket_id].append(acc.item())

    out = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "early_stops_status": props["early_stopping"],
    }

    return out
