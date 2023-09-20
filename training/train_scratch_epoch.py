import torch

from torch.utils.data import DataLoader
from utils import save_model_ckpt_scratch_cls, custom_timer_with_return


def train_cls_scratch(args, model, inputs, opt, criterion):
    # Set up model for training
    model.train()

    opt.zero_grad()

    out = model(inputs["x"])

    loss = criterion(out.view(args.n_speakers, -1, args.dim_emb))

    # Backward
    loss.backward()

    opt.step()

    return loss


@custom_timer_with_return
def train_scratch_per_epoch(
    args,
    device,
    epoch,
    classifier,
    optimizer,
    criterion,
    train_acc_storage,
    train_loss_storage,
    train_cls_scratch,
    test_cls_scratch,
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

    if args.delta and args.delta_delta:
        feat_dim_processed = args.feature_dim * 3
    elif args.delta:
        feat_dim_processed = args.feature_dim * 2
    else:
        feat_dim_processed = args.feature_dim

    x = x.reshape(-1, args.seg_len, feat_dim_processed).to(device)
    spk = spk.to(device)

    input_data = {"x": x, "y": spk}

    loss = train_cls_scratch(args, classifier, input_data, optimizer, criterion)
    # train_acc, _ = test_cls_scratch(
    #     args,
    #     classifier,
    #     input_data,
    #     criterion,
    #     mode="train",
    # )

    # if args.log_training:
    #     epoch_display = f"Train Epoch: {epoch}| "
    #     loss_display = f"Loss:{loss:0.3f}| "
    #     acc_display = f"Acc:{train_acc:0.3f}| "

    #     print(
    #         epoch_display,
    #         loss_display,
    #         acc_display,
    #     )

    # train_acc_storage.append(train_acc.tolist())
    # train_loss_storage.append(loss.item())

    out = {
        # "train_acc": train_acc_storage,
        # "train_loss": train_loss_storage,
        "early_stopping": early_stopping.early_stop,
    }

    # Save the checkpoint for "model"
    if epoch % args.save_every == 0:
        classifier.to("cpu")

        save_model_ckpt_scratch_cls(
            epoch,
            classifier,
            optimizer,
            criterion,
            loss,
            kwargs_training["filename_dir"],
        )

        classifier.to(device)

    return out


@custom_timer_with_return
def train_scratch(
    args,
    device,
    classifier,
    optimizer,
    ce_loss,
    train_cls_scratch,
    **kwargs_training,
):
    for epoch in range(args.epoch):
        train_scratch_per_epoch(
            args,
            device,
            epoch,
            classifier,
            optimizer,
            ce_loss,
            train_cls_scratch,
            **kwargs_training,
        )
