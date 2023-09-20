import torch
import torch.nn as nn


from torch.optim import Adam, SGD


from utils import (
    HyperParams,
    cor_seq_counter_list,
    unreg_spks_per_bkts,
    compute_spks_per_bkts_storage,
    DvecModelDynamicReReg,
    DvecOptimizer,
    DvecGeneralDynamicReReg,
    AttentivePooledLSTMDvector,
    SpeakerClassifierRec_v2,
    SupConLoss,
    get_logger,
    dataset_kwargs,
    model_kwargs,
    opt_kwargs,
    loss_kwargs,
    save_as_json,
    create_filenames_dvec,
    create_filenames_cls,
    create_filenames_re_reg_results,
    moving_average,
    create_moving_average_collection,
)


from evaluation import eval_per_epoch_progressive_contrastive_supervised_unreg_rereg
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
from agent import AgentSupervised
from .train_re_reg_epoch import train_re_reg_per_epoch


def re_reg_sup(
    args,
    hparams: HyperParams,
    buckets,
    device,
    unreg_spks,
    status_cls,
    ckpt_cls=None,
):

    # Dictionaries of filenames for the checkpoints of dvectors and classifier
    filenames_dvecs_and_dirs = create_filenames_dvec(buckets, args, hparams, unreg_spks)
    filenames_and_dirs = create_filenames_cls(args, hparams, unreg_spks)

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_re_reg_results(
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

    # Speaker indices per buckets before unregistering
    outputs = cor_seq_counter_list(
        len(labels),
        args.spk_per_bucket,
        args.spk_per_bucket,
    )
    # Speaker indices per buckets after unregistering
    outputs_updated, _ = unreg_spks_per_bkts(outputs, unreg_spks)

    # Create list of number of speakers per buckets
    spk_per_bkt_storage_old = compute_spks_per_bkts_storage(outputs_updated)
    spk_per_bkt_storage = compute_spks_per_bkts_storage(outputs)

    unreg_bkts_storage = [
        i
        for i in range(len(spk_per_bkt_storage_old))
        if spk_per_bkt_storage[i] != spk_per_bkt_storage_old[i]
    ]

    # Create training/validation datasets
    data_dir, speaker_infos = create_dataset_arguments(args, args.data_dir)
    validation_data_dir, speaker_infos_validation = create_dataset_arguments(
        args, args.validation_data_dir
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
    dvec_model_obj = DvecModelDynamicReReg(
        device,
        buckets,
        spk_per_bkt_storage_old,
        args,
    )
    dvec_opt_obj = DvecOptimizer(device, buckets, args, hparams)

    model_dvec = DvecGeneralDynamicReReg(
        dvec_model_obj,
        dvec_opt_obj,
        SGD,
        device,
        buckets,
        args,
    )

    dvectors, opt_dvecs, _ = model_dvec.load_model_opt(
        hparams,
        AttentivePooledLSTMDvector,
        SupConLoss,
        filenames_dvecs_and_dirs["filename_dvec"],
        filenames_dvecs_and_dirs["filename_dvec_unreg"],
        filenames_dvecs_and_dirs["filename_dvec_re_reg"],
    )

    # Classifier to be trained on the contrastive embedding replay
    classifier = SpeakerClassifierRec_v2(args).to(device)
    optimizer = Adam(classifier.parameters(), lr=hparams.lr_cls, amsgrad=True)

    classifier_ma = SpeakerClassifierRec_v2(args).to(device)
    ma_n = 0

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_cls is not None:
        ckpt_cls = torch.load(ckpt_cls)
        classifier.load_state_dict(ckpt_cls[hparams.model_str])
        optimizer.load_state_dict(ckpt_cls[hparams.opt_str])

        start_epoch_available = ckpt_cls.get(hparams.start_epoch)

        if start_epoch_available and status_cls == "re_reg_cls":
            start_epoch = start_epoch_available + 1
        else:
            start_epoch = 0

        classifier_ma.load_state_dict(ckpt_cls[hparams.model_ma_str])
        if hparams.ma_mode == "swa":
            ma_n = ckpt_cls[hparams.ma_n_str]
    else:
        start_epoch = 0

    # Initializing early stoppings for the buckets
    early_stopping = {bucket_id: [] for bucket_id in range(hparams.num_of_buckets)}
    for bucket_id in range(hparams.num_of_buckets):
        if args.early_stopping:
            early_stopping[bucket_id] = EarlyStoppingCustomUnreg(args)

    # The losses
    contrastive_loss = SupConLoss(args).to(device)  # Unsupervised contrastive loss
    ce_loss = nn.CrossEntropyLoss().to(device)  # CrossEntropy loss for the classifier

    # Initialize the buffer class (if required)
    re_reg_buffer = CreateMultiStridedSamples(args)

    # Instantiate the Agent class
    agent = AgentSupervised(args, device, hparams)

    # Create kwargs for the training/validation function
    kwargs_dataset = dataset_kwargs(
        SubDatasetGdrSpk,
        collateGdrSpkr,
        dataset,
        dataset_validation,
    )

    kwargs_model = model_kwargs(agent, dvectors, classifier)
    kwargs_filename_dvec = filenames_dvecs_and_dirs
    kwargs_filename_cls = filenames_and_dirs

    kwargs_opt = opt_kwargs(
        SGD,
        opt_dvecs,
        Adam,
        optimizer,
        early_stopping,
    )

    kwargs_loss = loss_kwargs(contrastive_loss, ce_loss)

    # Combine training and validation kwargs
    kwargs_training_val = (
        kwargs_dataset
        | kwargs_model
        | {"classifier_ma": classifier_ma, "ma_n": ma_n}
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

    # Initialize validation accuracy and loss to be saved during training
    val_loss = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    val_acc = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    pred_indx = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    gtruth_indx = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}

    # Initialize the elapsed time per epoch
    td_per_epoch = []

    for epoch in range(start_epoch, start_epoch + args.epoch):

        # Train the d-vectors per epoch and evaluate the performance
        td = train_re_reg_per_epoch(
            hparams,
            args,
            device,
            outputs,
            outputs_updated,
            buckets,
            epoch,
            spk_per_bkt_storage,
            re_reg_buffer,
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
        val_out = eval_per_epoch_progressive_contrastive_supervised_unreg_rereg(
            args,
            device,
            outputs,
            unreg_bkts_storage,
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

        # Break the training loop if all the buckets with previously unregistered speakers
        # have ``True'' early stopping status
        if early_stopping_status[bkt_id] and all(early_stopping_status_unreg.values()):
            break

    # Save the overal elapsed time as a JSON file
    save_as_json(
        paths_filenames["dir_td"],
        paths_filenames["filename_time_delay"],
        td_per_epoch,
    )
