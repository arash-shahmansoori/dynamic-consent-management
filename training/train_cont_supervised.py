import torch
import torch.nn as nn


from torch.optim import Adam, SGD


from utils import (
    HyperParams,
    cor_seq_counter_list,
    DvecModel,
    DvecOptimizer,
    DvecGeneral,
    AttentivePooledLSTMDvector,
    # AttentivePooledLSTMDvectorLiterature,
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
    create_filenames_results,
    moving_average,
    create_moving_average_collection,
)


from .train_epoch_cont_supervised_selective import (
    train_per_epoch_contrastive_supervised_selective,
)
from evaluation import eval_per_epoch_progressive_contrastive_supervised
from scheduler_early_stop import (
    EarlyStoppingCustom,
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


def train_contrastive_supervised(
    args,
    hparams: HyperParams,
    buckets,
    device,
    ckpt_cls=None,
):

    # Dictionaries of filenames for the checkpoints of dvectors and classifier
    filenames_dvecs_and_dirs = create_filenames_dvec(buckets, args, hparams)
    filenames_and_dirs = create_filenames_cls(args, hparams)

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_results(
        args,
        hparams.ma_mode,
        args.max_mem,
        args.spk_per_bucket,
        hparams.train_dvec_mode,
        args.agnt_num,
    )

    # Create moving average collection of functions
    moving_average_collection = create_moving_average_collection(
        swa_scheduling,
        no_ma_scheduling,
    )

    # Create the index list of speakers
    labels = [i for i in range(args.n_speakers)]

    outputs = cor_seq_counter_list(
        len(labels), args.spk_per_bucket, args.spk_per_bucket
    )

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
    dvec_model_obj = DvecModel(device, buckets, args)
    dvec_opt_obj = DvecOptimizer(device, buckets, args, hparams)

    model_dvec = DvecGeneral(dvec_model_obj, dvec_opt_obj, SGD, device, buckets, args)
    dvectors, opt_dvecs, _ = model_dvec.load_model_opt(
        hparams,
        AttentivePooledLSTMDvector,
        # AttentivePooledLSTMDvectorLiterature,
        SupConLoss,
        filenames_dvecs_and_dirs["filename_dvec"],
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

        if start_epoch_available:
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
            early_stopping[bucket_id] = EarlyStoppingCustom(args)

    # The losses
    contrastive_loss = SupConLoss(args).to(device)  # Unsupervised contrastive loss
    ce_loss = nn.CrossEntropyLoss().to(device)  # CrossEntropy loss for the classifier

    # Initialize the buffer class
    create_buffer = CreateMultiStridedSamples(args)

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

    # Initialize training/validation accuracy and loss to be saved during training/validation
    train_loss = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    val_loss = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}

    train_acc = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    val_acc = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}

    # Initialize the elapsed time per epoch
    td_per_epoch = []

    for epoch in range(start_epoch, start_epoch + args.epoch):

        # Train the d-vectors per epoch and evaluate the performance
        td, train_out = train_per_epoch_contrastive_supervised_selective(
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
        val_out = eval_per_epoch_progressive_contrastive_supervised(
            args,
            device,
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
                    epoch,
                    bkt_id,
                )

        # Save the required validation metrics as JSON files
        save_as_json(
            paths_filenames["dir_loss_val"],
            paths_filenames["filename_loss_val"],
            val_out["val_loss"],
        )

        save_as_json(
            paths_filenames["dir_acc_val"],
            paths_filenames["filename_acc_val"],
            val_out["val_acc"],
        )

        # Break training if the early stopping status is ``True'' after completion of progressive registrations
        if train_out["early_stops_status"][hparams.num_of_buckets - 1]:
            break

        # Save the required training metrics as JSON files
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
