import torch


from torch.optim import SGD


from utils import (
    HyperParams,
    cor_seq_counter_list,
    unreg_spks_per_bkts,
    compute_spks_per_bkts_storage,
    DvecModelDynamicUnRegUnsupervised,
    DvecOptimizerUnRegUnsupervised,
    DvecGeneralDynamicRegUnsupervised,
    AttentivePooledLSTMDvector,
    UnsupClsLatent,
    GE2ELoss,
    GE2ELossLatent,
    get_logger,
    dataset_kwargs,
    model_kwargs_unsupervised,
    opt_kwargs,
    loss_kwargs_unsupervised,
    save_as_json,
    create_filenames_dvec_unsupervised,
    create_filenames_dvec_unsupervised_latent,
    create_filenames_unreg_unsup_results,
    moving_average,
    create_moving_average_collection,
)


from evaluation import (
    eval_per_epoch_per_bucket_contrastive_unsupervised_unreg,
)
from scheduler_early_stop import (
    EarlyStoppingCustomUnreg,
    swa_scheduling,
    no_ma_scheduling,
)

from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    SubDatasetGdrSpk,
    collateGdrSpkr,
    create_dataset_arguments,
)

from create_buffer import CreateMultiStridedSamples
from agent import AgentUnSupervisedUnregV2
from .train_unreg_epoch_unsup_v2 import train_unreg_per_epoch_unsup_v2


def unreg_unsup_v2(
    args,
    hparams: HyperParams,
    buckets,
    device,
    unreg_spks,
    status_dvec_latent,
    ckpt_dvec_latent=None,
):

    # Dictionaries of filenames for the checkpoints of dvectors and classifier
    filenames_dvec_and_dirs = create_filenames_dvec_unsupervised(
        buckets,
        args,
        hparams,
        unreg_spks,
    )
    filenames_and_dirs = create_filenames_dvec_unsupervised_latent(
        args,
        hparams,
        unreg_spks,
    )

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_unreg_unsup_results(
        args,
        hparams,
        args.spk_per_bucket,
        unreg_spks,
    )

    # Create moving average collection of functions
    moving_average_collection = create_moving_average_collection(
        swa_scheduling,
        no_ma_scheduling,
    )

    # Create the index list of speakers
    labels = [i for i in range(args.n_speakers)]

    outputs = cor_seq_counter_list(
        len(labels),
        args.spk_per_bucket,
        args.spk_per_bucket,
    )

    updated_outputs, _ = unreg_spks_per_bkts(outputs, unreg_spks)

    # Create list of number of speakers per buckets
    spk_per_bkt_storage_old = compute_spks_per_bkts_storage(outputs)
    spk_per_bkt_storage = compute_spks_per_bkts_storage(updated_outputs)

    unreg_bkts_storage = [
        i
        for i in range(len(spk_per_bkt_storage_old))
        if spk_per_bkt_storage[i] != spk_per_bkt_storage_old[i]
    ]

    # Create training/validation datasets
    data_dir, speaker_infos = create_dataset_arguments(args, args.data_dir)
    data_dir_other, speaker_infos_other = create_dataset_arguments(
        args,
        args.data_dir_other,
    )

    validation_data_dir, speaker_infos_validation = create_dataset_arguments(
        args, args.validation_data_dir
    )

    dataset = ClassificationDatasetGdrSpkr(
        data_dir,
        speaker_infos,
        args.n_utterances_unlabeled,
        args.seg_len,
    )
    dataset_other = ClassificationDatasetGdrSpkr(
        data_dir_other,
        speaker_infos_other,
        args.n_utterances_labeled_reg,
        args.seg_len,
    )

    dataset_validation = ClassificationDatasetGdrSpkr(
        validation_data_dir,
        speaker_infos_validation,
        args.nv_utterances_unlabeled,
        args.seg_len,
    )

    # Build the models for d-vectors and load the available checkpoints for the buckets
    dvec_model_obj = DvecModelDynamicUnRegUnsupervised(
        device,
        buckets,
        unreg_bkts_storage,
        args,
    )
    dvec_opt_obj = DvecOptimizerUnRegUnsupervised(
        device,
        buckets,
        args,
        unreg_bkts_storage,
        hparams,
    )

    model_dvec = DvecGeneralDynamicRegUnsupervised(
        dvec_model_obj,
        dvec_opt_obj,
        SGD,
        device,
        buckets,
        args,
    )

    dvectors, cont_losses, opt_dvecs, _ = model_dvec.load_model_opt(
        hparams,
        AttentivePooledLSTMDvector,
        GE2ELoss,
        filenames_dvec_and_dirs["filename_dvec"],
        filenames_dvec_and_dirs["filename_dvec_unreg"],
    )

    # d-vec in the latent space
    dvec_latent = UnsupClsLatent(args).to(device)

    # Create the moving average model
    dvec_latent_ma = UnsupClsLatent(args).to(device)
    ma_n = 0

    # Unsupervised contrastive loss for the latent space
    contrastive_loss_latent = GE2ELossLatent(args).to(device)

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_dvec_latent is not None:
        ckpt_dvec_latent = torch.load(ckpt_dvec_latent)
        dvec_latent.load_state_dict(ckpt_dvec_latent[hparams.model_str])
        contrastive_loss_latent.load_state_dict(ckpt_dvec_latent[hparams.contloss_str])

        start_epoch_available = ckpt_dvec_latent.get(hparams.start_epoch)

        if start_epoch_available and status_dvec_latent == "re_reg_dvec_latent":
            start_epoch = start_epoch_available + 1
        else:
            start_epoch = 0

        dvec_latent_ma.load_state_dict(ckpt_dvec_latent[hparams.model_ma_str])
        if hparams.ma_mode == "swa":
            ma_n = ckpt_dvec_latent[hparams.ma_n_str]
    else:
        start_epoch = 0

    # Initializing early stoppings for the buckets
    early_stopping = {bucket_id: [] for bucket_id in range(hparams.num_of_buckets)}
    for bucket_id in range(hparams.num_of_buckets):
        if args.early_stopping:
            early_stopping[bucket_id] = EarlyStoppingCustomUnreg(args)

    # Initialize the buffer class (if required)
    unreg_buffer = CreateMultiStridedSamples(args)

    # Instantiate the Agent class
    agent = AgentUnSupervisedUnregV2(args, device, hparams)

    # Create kwargs for the training/validation function
    kwargs_dataset = dataset_kwargs(
        SubDatasetGdrSpk,
        collateGdrSpkr,
        dataset,
        dataset_validation,
        dataset_other,
    )

    kwargs_model = model_kwargs_unsupervised(agent, dvectors, dvec_latent)
    kwargs_filename_dvec = filenames_dvec_and_dirs
    kwargs_filename_cls = filenames_and_dirs

    kwargs_opt = opt_kwargs(
        SGD,
        opt_dvecs,
        SGD,
        None,
        early_stopping,
    )

    kwargs_loss = loss_kwargs_unsupervised(cont_losses, contrastive_loss_latent)

    # Combine training and validation kwargs
    kwargs_training_val = (
        kwargs_dataset
        | kwargs_model
        | {"dvec_latent_ma": dvec_latent_ma, "ma_n": ma_n}
        | kwargs_opt
        | kwargs_loss
        | kwargs_filename_dvec
        | kwargs_filename_cls
    )

    # Logging
    logger = get_logger()

    # Initialze early stopping status per buckets
    early_stopping_status = {bkt: False for bkt in buckets}
    early_stopping_status_unreg = {bkt: False for bkt in unreg_bkts_storage}

    # Initialize training/validation accuracy and loss to be saved during training/validation
    val_loss = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    val_acc = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}

    # Initialize the elapsed time per epoch
    td_per_epoch = []

    for epoch in range(start_epoch, start_epoch + args.epoch):

        # Train the d-vectors per epoch and evaluate the performance
        td = train_unreg_per_epoch_unsup_v2(
            hparams,
            args,
            device,
            updated_outputs,
            outputs,
            buckets,
            epoch,
            spk_per_bkt_storage,
            unreg_buffer,
            **kwargs_training_val,
        )

        # Store the elapsed time per epoch in a list
        td_per_epoch.append(td)

        # Moving average strategy
        moving_average_collection[hparams.ma_mode](
            swa_start=args.swa_start,
            swa_lr=args.swa_lr,
            lr_cls=hparams.lr_cls,
            epochs=args.epoch,
            moving_average=moving_average,
            **kwargs_training_val,
        )

        # Evaluate the performance per epoch
        val_out = eval_per_epoch_per_bucket_contrastive_unsupervised_unreg(
            hparams,
            args,
            device,
            updated_outputs,
            outputs,
            buckets,
            val_loss,
            val_acc,
            epoch,
            **kwargs_training_val,
        )

        # Scheduling
        # Metric-based
        # scheduler_metric(val_out["val_loss"])

        # Update early stopping parameters for the buckets
        if args.early_stopping:
            for _, bkt_id in enumerate(buckets):

                early_stopping[bkt_id](
                    torch.tensor(val_out["val_acc"][bkt_id]).view(-1)[-1],
                    spk_per_bkt_storage[bkt_id],
                    epoch,
                    bkt_id,
                )

                if kwargs_training_val["early_stop"][bkt_id].early_stop:
                    early_stopping_status[bkt_id] = True

                    if bkt_id not in unreg_bkts_storage:
                        early_stopping_status[bkt_id] = True
                    elif bkt_id in unreg_bkts_storage:
                        early_stopping_status_unreg[bkt_id] = True

                    logger.info(f"Training of the bucket:{bkt_id} completed.")

        # Save the required validation metrics as JSON files
        save_as_json(
            paths_filenames["dir_acc_val"],
            paths_filenames["filename_acc_val"],
            val_out["val_acc"],
        )

        # Break the training loop if all the buckets have ``True'' early stopping status
        if all(early_stopping_status_unreg.values()):
            break

    # Save the overal elapsed time as a JSON file
    save_as_json(
        paths_filenames["dir_td"],
        paths_filenames["filename_time_delay"],
        td_per_epoch,
    )
