import torch


from torch.optim import SGD
from utils import (
    HyperParams,
    SpeakerClassifierE2EUnsupervisedV2,
    GE2ELossStable,
    get_logger,
    dataset_kwargs,
    filename_kwargs_scratch,
    save_as_json,
    create_filenames_scratch_unsupervised_v2,
    create_filenames_unsupervised_results_v2,
    create_moving_average_collection,
)


from .train_scratch_epoch_unsup import train_scratch_per_epoch_unsup, train_dvec_e2e
from evaluation import eval_scratch_per_epoch_unsup, eval_metrics_cont_loss
from scheduler_early_stop import (
    EarlyStoppingCustomLossAcc,
    swa_scheduling_unsup,
    no_ma_scheduling,
)
from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    SubDatasetGdrSpk,
    collateGdrSpkr,
    create_dataset_arguments,
)


def train_from_scratch_unsup(
    args,
    hparams: HyperParams,
    buckets,
    device,
    ckpt_dvec_latent=None,
):
    # Dictionaries of filenames for the checkpoints of dvectors and latent features
    filename, filename_dir = create_filenames_scratch_unsupervised_v2(args)

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_unsupervised_results_v2(args, args.agnt_num)

    # Create training/validation datasets
    data_dir, speaker_infos = create_dataset_arguments(args, args.data_dir)
    validation_data_dir, speaker_infos_validation = create_dataset_arguments(
        args,
        args.validation_data_dir,
    )

    dataset = ClassificationDatasetGdrSpkr(
        data_dir,
        speaker_infos,
        args.n_utterances_unlabeled,
        args.seg_len,
    )
    dataset_validation = ClassificationDatasetGdrSpkr(
        validation_data_dir,
        speaker_infos_validation,
        args.nv_utterances_unlabeled,
        args.seg_len,
    )

    # Build the models for d-vectors and load the available checkpoints for the buckets
    dvec_model = SpeakerClassifierE2EUnsupervisedV2(args).to(device)

    # Unsupervised contrastive loss for the latent space
    contrastive_loss_latent = GE2ELossStable(args).to(device)

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_dvec_latent is not None:
        ckpt_dvec_latent = torch.load(ckpt_dvec_latent)
        dvec_model.load_state_dict(ckpt_dvec_latent[hparams.model_str])
        contrastive_loss_latent.load_state_dict(ckpt_dvec_latent[hparams.contloss_str])

    # Initializing early stoppings for the buckets
    if args.early_stopping:
        early_stopping = EarlyStoppingCustomLossAcc(args)

    # Create kwargs for the training/validation function
    kwargs_dataset = dataset_kwargs(
        SubDatasetGdrSpk,
        collateGdrSpkr,
        dataset,
        dataset_validation,
    )
    kwargs_filename = filename_kwargs_scratch(filename, filename_dir)

    # Combine training and validation kwargs
    kwargs_training_val = kwargs_dataset | kwargs_filename

    # Logging
    logger = get_logger()

    # Initialize validation accuracy and loss to be saved
    train_acc_storage, train_loss_storage = [], []
    val_acc_storage, val_loss_storage = [], []

    # Initialize the elapsed time per epoch
    td_per_epoch = []

    for epoch in range(args.epoch):
        # Train the d-vectors per epoch and evaluate the performance
        td, train_out = train_scratch_per_epoch_unsup(
            args,
            hparams,
            device,
            epoch,
            dvec_model,
            SGD,
            contrastive_loss_latent,
            train_acc_storage,
            train_loss_storage,
            train_dvec_e2e,
            early_stopping,
            **kwargs_training_val,
        )

        # Store the elapsed time per epoch in a list
        td_per_epoch.append(td)

        # Scheduling
        # Non-metric based
        # kwargs_training["lr_scheduler"].step()

        # Evaluate the performance per epoch
        val_out = eval_scratch_per_epoch_unsup(
            args,
            device,
            dvec_model,
            contrastive_loss_latent,
            val_loss_storage,
            val_acc_storage,
            eval_metrics_cont_loss,
            epoch,
            **kwargs_training_val,
        )

        # Scheduling
        # Metric-based
        # scheduler_metric(val_out["val_acc_cont"])

        # Update early stopping parameters for the buckets
        if args.early_stopping:
            for _, bkt_id in enumerate(buckets):
                early_stopping(
                    torch.tensor(val_out["val_acc"]).view(-1)[-1],
                    torch.tensor(val_out["val_loss"]).view(-1)[-1],
                    epoch,
                    bkt_id,
                )

        # Save the required validation metrics as JSON files
        save_as_json(
            paths_filenames["dir_loss_cont_val"],
            paths_filenames["filename_loss_cont_val"],
            val_out["val_loss"],
        )
        save_as_json(
            paths_filenames["dir_acc_cont_val"],
            paths_filenames["filename_acc_cont_val"],
            val_out["val_acc"],
        )

        # Break training if the early stopping status is ``True''
        if train_out["early_stopping"]:
            break

    # Save the overal elapsed time as a JSON file
    save_as_json(
        paths_filenames["dir_td"],
        paths_filenames["filename_time_delay"],
        td_per_epoch,
    )
