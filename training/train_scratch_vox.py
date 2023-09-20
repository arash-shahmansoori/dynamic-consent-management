import torch
import torch.nn as nn


from torch.optim import Adam
from pathlib import Path

from utils import (
    SophiaG,
    HyperParams,
    SpeakerClassifierE2ESupervisedV2,
    StableSupContLoss,
    save_as_json,
    create_filenames_scratch_vox,
    dataset_spk_kwargs,
    filename_kwargs_scratch,
    create_filenames_results_scratch_vox,
)

from .train_scratch_epoch_vox import train_cls_scratch_vox, train_scratch_per_epoch_vox
from evaluation import eval_metrics_cont_loss, eval_scratch_per_epoch_vox
from scheduler_early_stop import EarlyStoppingCustomLoss
from preprocess_data import (
    ClassificationDatasetSpkr,
    ClassificationDatasetSpkrV2,
    collateSpkr,
    create_dataset_speaker_arguments,
    SubDatasetSpk,
)


def train_from_scratch_vox(
    args,
    hparams: HyperParams,
    buckets,
    device,
    ckpt_scratch=None,
):
    # Dictionaries of filenames for the checkpoints of classifier
    filename, filename_dir = create_filenames_scratch_vox(args)

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_results_scratch_vox(
        args,
        hparams.ma_mode,
        args.spk_per_bucket,
        hparams.train_dvec_mode,
        args.agnt_num,
    )

    # Create training/validation datasets
    data_dir, speaker_infos = create_dataset_speaker_arguments(
        args,
        args.data_dir_vox_train,
    )
    validation_data_dir, speaker_infos_validation = create_dataset_speaker_arguments(
        args,
        args.data_dir_vox_test,
    )

    dataset = ClassificationDatasetSpkr(
        data_dir,
        speaker_infos,
        args.n_train_vox_utts,
        args.seg_len,
    )

    dataset_validation = ClassificationDatasetSpkrV2(
        validation_data_dir,
        speaker_infos_validation,
        args.seg_len,
    )

    # loss
    criterion = StableSupContLoss(args).to(device)

    # E2E from scratch training
    cls_scratch = SpeakerClassifierE2ESupervisedV2(args).to(device)

    # optimizer_scratch = Adam(
    #     list(cls_scratch.parameters()) + list(criterion.parameters()),
    #     lr=3e-4,
    #     amsgrad=True,
    # )
    optimizer_scratch = SophiaG(
        list(cls_scratch.parameters()) + list(criterion.parameters()),
        lr=3e-4,
        betas=(0.9, 0.95),
        rho=0.03,
    )

    # Load available checkpoints for the speaker recognition
    if ckpt_scratch is not None:
        ckpt_scratch = torch.load(ckpt_scratch)
        cls_scratch.load_state_dict(ckpt_scratch[hparams.model_str])
        optimizer_scratch.load_state_dict(ckpt_scratch[hparams.opt_str])

    # Initializing early stoppings for the buckets
    if args.early_stopping:
        early_stopping = EarlyStoppingCustomLoss(args)

    # Create kwargs for the training/validation function
    kwargs_dataset = dataset_spk_kwargs(
        SubDatasetSpk,
        collateSpkr,
        dataset,
        dataset_validation,
    )

    kwargs_filename_cls = filename_kwargs_scratch(filename, filename_dir)

    # Combine training kwargs
    kwargs_training = kwargs_dataset | kwargs_filename_cls

    # Combine validation kwargs
    kwargs_validation = kwargs_dataset

    # Initialize training/validation accuracy and loss to be saved during training/validation
    train_acc_storage, train_loss_storage = [], []
    val_acc_storage, val_loss_storage = [], []

    td_per_epoch = []

    for epoch in range(args.epoch):
        td, train_out = train_scratch_per_epoch_vox(
            args,
            device,
            epoch,
            cls_scratch,
            optimizer_scratch,
            criterion,
            train_acc_storage,
            train_loss_storage,
            train_cls_scratch_vox,
            eval_metrics_cont_loss,
            early_stopping,
            **kwargs_training,
        )

        # Store the elapsed time per epoch in a list
        td_per_epoch.append(td)

        # val_out = eval_scratch_per_epoch_vox(
        #     args,
        #     device,
        #     epoch,
        #     cls_scratch,
        #     criterion,
        #     val_acc_storage,
        #     val_loss_storage,
        #     eval_metrics_cont_loss,
        #     **kwargs_validation,
        # )

        # Update early stopping parameters for the buckets
        if args.early_stopping:
            for _, bkt_id in enumerate(buckets):
                # early_stopping(
                #     torch.tensor(train_out["train_acc"]).view(-1)[-1],
                #     epoch,
                #     bkt_id,
                # )
                # early_stopping(
                #     torch.tensor(val_out["val_acc"]).view(-1)[-1],
                #     epoch,
                #     bkt_id,
                # )
                # early_stopping(
                #     torch.tensor(val_out["val_acc"]).view(-1)[-1],
                #     torch.tensor(val_out["val_loss"]).view(-1)[-1],
                #     epoch,
                #     bkt_id,
                # )
                early_stopping(
                    torch.tensor(train_out["train_loss"]).view(-1)[-1],
                    epoch,
                    bkt_id,
                )

        # # Save the required metrics as JSON files
        # save_as_json(
        #     paths_filenames["dir_acc_val"],
        #     paths_filenames["filename_acc_val"],
        #     val_out["val_acc"],
        # )

        # Break training if the early stopping status is ``True''
        if train_out["early_stopping"]:
            break

        save_as_json(
            paths_filenames["dir_loss_train"],
            paths_filenames["filename_loss_train"],
            train_out["train_loss"],
        )
        save_as_json(
            paths_filenames["dir_acc_train"],
            paths_filenames["filename_acc_train"],
            train_out["train_acc"],
        )

    # Save the overal elapsed time as a JSON file
    save_as_json(
        paths_filenames["dir_td"],
        paths_filenames["filename_time_delay"],
        td_per_epoch,
    )
