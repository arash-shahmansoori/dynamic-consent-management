from utils import custom_timer_with_return
from .train_selective_inductive_bias_supervised_vox import (
    train_selective_inductive_bias_sup_vox,
)


@custom_timer_with_return
def train_per_epoch_contrastive_supervised_selective_vox(
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
    # Create index collection according to a given strategy: (1) & (2)
    lf_collection, feats_init, labels_init = create_buffer.create_collect_indx(
        args, buckets
    )  # (1)
    # (
    #     lf_collect_progress,
    #     bkt_samples,
    #     bkt_labels,
    # ) = create_buffer.create_progressive_collect_indx(
    #     args, buckets
    # )  # (2)

    early_stop_status = []
    early_stop_status_bkt = {bucket_id: [] for _, bucket_id in enumerate(buckets)}

    for indx, bucket_id in enumerate(buckets):
        # Generate selective inductive bias from the buffer stack (supervised)
        props = train_selective_inductive_bias_sup_vox(
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

        # Custom strategy (1) for creating the buffer including the new registrations
        stacked_x, stacked_y = create_buffer.inter_bucket_sample(
            lf_collection[indx],
            xe,
            props["y"],
            feats_init,
            labels_init,
        )

        # Custom strategy (2) for creating the buffer including the new registrations
        # bkt_samples[str(indx)], bkt_labels[str(indx)] = xe, props["y"]
        # stacked_x, stacked_y = create_buffer.inter_bucket_sample_v2(
        #     indx,
        #     lf_collect_progress,
        #     bkt_samples,
        #     bkt_labels,
        #     device,
        # )

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

        if args.log_training:
            loss_display = loss.item()
            acc_display = acc.item()

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
