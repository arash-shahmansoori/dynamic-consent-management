from torch.utils.data import DataLoader

from utils import (
    save_model_ckpt_scratch_cls,
    custom_timer_with_return,
    create_calibrated_length,
)


def train_dvec_e2e(
    args,
    hparams,
    model_dvec_latent,
    opt_cls_type,
    cont_loss,
    input_data,
    epoch,
    device,
    filename_dir,
):
    optimizer = opt_cls_type(
        [
            {
                "params": list(model_dvec_latent.parameters())
                + list(cont_loss.parameters()),
                "weight_decay": hparams.weight_decay,
            }
        ],
        lr=2e-1,
        momentum=hparams.momentum,
        nesterov=hparams.nesterov,
        dampening=hparams.dampening,
    )

    # Set up model for training
    model_dvec_latent.train()
    cont_loss.train()

    x_buffer_noncalibrated = input_data["x"]

    optimizer.zero_grad()

    out = model_dvec_latent(x_buffer_noncalibrated)

    # loss = cont_loss(
    #     out.view(
    #         args.n_speakers,
    #         -1,
    #         args.latent_dim,
    #     )
    # )
    loss = cont_loss(
        out.view(
            args.n_speakers,
            -1,
            args.dim_emb,
        )
    )

    loss.backward()

    optimizer.step()

    cos_sim_matrix = cont_loss.compute_similarity_matrix(
        out.view(args.n_speakers, -1, args.dim_emb)
    )

    acc = cont_loss.calc_acc(cos_sim_matrix)

    # Save the checkpoint for "model"
    model_dvec_latent.to("cpu")

    save_model_ckpt_scratch_cls(
        epoch,
        model_dvec_latent,
        optimizer,
        cont_loss,
        loss,
        filename_dir,
    )

    model_dvec_latent.to(device)

    return loss, acc


@custom_timer_with_return
def train_scratch_per_epoch_unsup(
    args,
    hparams,
    device,
    epoch,
    model_dvec_latent,
    opt_cls_type,
    cont_loss,
    train_acc_storage,
    train_loss_storage,
    train_dvec_e2e,
    early_stopping,
    **kwargs_training,
):
    train_loader = DataLoader(
        kwargs_training["dataset"],
        batch_size=args.n_speakers,
        shuffle=False,
        collate_fn=kwargs_training["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch = next(iter(train_loader))

    x, _, spk = mel_db_batch
    x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk = spk.to(device)

    input_data = {"x": x, "y": spk}

    loss, acc = train_dvec_e2e(
        args,
        hparams,
        model_dvec_latent,
        opt_cls_type,
        cont_loss,
        input_data,
        epoch,
        device,
        kwargs_training["filename_dir"],
    )

    if args.log_training:
        loss_display = loss.item()
        acc_display = acc.item()

        epoch_display = f"Train Epoch: {epoch}| "

        loss_display = f"Loss:{loss_display:0.3f}| "
        acc_display = f"Acc:{acc_display:0.3f}| "

        print(epoch_display, loss_display, acc_display)

    train_acc_storage.append(acc.tolist())
    train_loss_storage.append(loss.item())

    out = {
        "train_loss": train_loss_storage,
        "train_acc": train_acc_storage,
        "early_stopping": early_stopping.early_stop,
    }

    return out
