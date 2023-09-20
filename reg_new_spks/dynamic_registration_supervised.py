import torch
import torch.nn as nn


from torch.optim import Adam, SGD


from utils import (
    HyperParams,
    cor_seq_counter_list,
    DvecModelDynamicReg,
    DvecOptimizer,
    DvecGeneralDynamicReg,
    AttentivePooledLSTMDvector,
    SpeakerClassifierRec_DR_v2,
    SpeakerClassifierRec_v2,
    SupConLoss,
    per_round_spks_per_bkts_storage,
    create_filenames_dvec,
    create_filenames_cls,
    get_logger,
    dataset_kwargs,
    model_kwargs,
    opt_kwargs,
    loss_kwargs,
    filename_kwargs_dvec,
    filename_kwargs_cls,
    create_filenames_reg_supervised_results,
    strategy_per_bkt_indx,
    save_as_json,
    moving_average,
    create_moving_average_collection,
)
from compute_optimal_buckets import (
    create_unique_opt_bkts_spks,
    create_unique_opt_bkt_spks_existing,
    create_unique_opt_bkt_spks_sofar,
    compute_opt_bkt_final,
    unique_opt_seq_final,
)
from scheduler_early_stop import (
    EarlyStoppingCustom,
    # swa_schedule,
    # adjust_learning_rate,
    swa_scheduling,
    no_ma_scheduling,
    early_stop_strategy_bkt,
)

from .train_reg_epoch import train_reg_per_round_per_epoch
from evaluation_reg import eval_reg_progressive_per_round_per_epoch_sup
from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    SubDatasetGdrSpk,
    collateGdrSpkr,
    create_dataset_arguments,
)

from agent import AgentSupervised

from create_buffer import CreateMultiStridedSamples


def dyn_reg_sup(
    args,
    hparams: HyperParams,
    data_dir_old,
    buckets,
    device,
    status_cls,
    ckpt_cls=None,
):

    # Initialize starting epoch per round
    start_epoch = {}

    # Dictionaries of filenames for the checkpoints of dvectors and classifier
    filenames_dvecs_and_dirs = create_filenames_dvec(buckets, args, hparams)
    filenames_and_dirs = create_filenames_cls(args, hparams)

    # Loading json files
    data_dir, speaker_infos = create_dataset_arguments(args, data_dir_old)
    data_dir_other, speaker_infos_other = create_dataset_arguments(
        args,
        args.data_dir_other,
    )

    validation_data_dir, speaker_infos_validation = create_dataset_arguments(
        args,
        args.validation_data_dir,
    )
    (
        validation_data_dir_other,
        speaker_infos_validation_other,
    ) = create_dataset_arguments(args, args.validation_data_dir_other)

    labels = [i for i in range(args.n_speakers)]

    outputs = cor_seq_counter_list(
        len(labels),
        args.spk_per_bucket,
        args.spk_per_bucket,
    )

    # Datasets for training and validation; including main and new registrations(i.e., other)
    dataset = ClassificationDatasetGdrSpkr(
        data_dir,
        speaker_infos,
        args.n_utterances_labeled_old,
        args.seg_len,
    )
    dataset_validation = ClassificationDatasetGdrSpkr(
        validation_data_dir,
        speaker_infos_validation,
        args.nt_utterances_labeled,
        args.seg_len,
    )

    dataset_other = ClassificationDatasetGdrSpkr(
        data_dir_other,
        speaker_infos_other,
        args.n_utterances_labeled_reg,
        args.seg_len,
    )
    dataset_validation_other = ClassificationDatasetGdrSpkr(
        validation_data_dir_other,
        speaker_infos_validation_other,
        args.nt_utterances_labeled,
        args.seg_len,
    )

    # Build the models for d-vectors and load the available checkpoints for the buckets
    dvec_model_obj = DvecModelDynamicReg(device, buckets, args)
    dvec_opt_obj = DvecOptimizer(device, buckets, args, hparams)

    model_dvec = DvecGeneralDynamicReg(
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
        filenames_dvecs_and_dirs["filename_dvec_reg"],
    )

    # Classifier to be trained on the contrastive embedding replay
    if hparams.round_num == 0 and status_cls == "cls":
        classifier = SpeakerClassifierRec_v2(args).to(device)

        # Create the moving average model
        classifier_ma = SpeakerClassifierRec_v2(args).to(device)
        ma_n = 0

    else:
        classifier = SpeakerClassifierRec_DR_v2(args).to(device)

        classifier_ma = SpeakerClassifierRec_DR_v2(args).to(device)
        ma_n = 0

    optimizer = Adam(classifier.parameters(), lr=hparams.lr_cls, amsgrad=True)

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_cls is not None:
        ckpt_cls = torch.load(ckpt_cls)
        classifier.load_state_dict(ckpt_cls[hparams.model_str])
        optimizer.load_state_dict(ckpt_cls[hparams.opt_str])

        start_epoch_round_available = ckpt_cls.get(
            f"start_epoch_round_{hparams.round_num}"
        )

        if start_epoch_round_available and start_epoch_round_available >= 0:
            start_epoch[hparams.round_num] = start_epoch_round_available + 1
        else:
            start_epoch[hparams.round_num] = 0

        classifier_ma.load_state_dict(ckpt_cls[hparams.model_ma_str])
        ma_n = ckpt_cls[hparams.ma_n_str]
    else:
        start_epoch[hparams.round_num] = 0

    # Modify the last layer of the classifier to match the updated number of speakers
    if hparams.round_num == 0 and status_cls == "cls":
        classifier.out = nn.Linear(
            args.latent_dim, args.n_speakers + args.n_speakers_other
        ).to(device)
        optimizer = Adam(classifier.parameters(), lr=hparams.lr_cls, amsgrad=True)

        classifier_ma.out = nn.Linear(
            args.latent_dim, args.n_speakers + args.n_speakers_other
        ).to(device)

    # Initialize learning rate scheduler
    # scheduler = MultiStepLR(optimizer, milestones=[20, 30], gamma=0.5)
    # scheduler_metric = LRScheduler(optimizer, args)

    # Initializing early stoppings for the buckets
    early_stopping = {bucket_id: [] for _, bucket_id in enumerate(buckets)}
    for _, bucket_id in enumerate(buckets):
        early_stopping[bucket_id] = EarlyStoppingCustom(args)

    # The losses
    contrastive_loss = SupConLoss(args).to(device)  # Supervised contrastive loss
    ce_loss = nn.CrossEntropyLoss().to(device)  # CrossEntropy loss for the classifier

    # Initialize the buffer class (if required)
    new_reg_buffer = CreateMultiStridedSamples(args)

    # Instantiate the Agent class
    agent = AgentSupervised(args, device, hparams)

    # Create kwargs for the training/validation function
    kwargs_dataset = dataset_kwargs(
        SubDatasetGdrSpk,
        collateGdrSpkr,
        dataset,
        dataset_validation,
        dataset_other,
        dataset_validation_other,
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

    # Combine training kwargs
    kwargs_training = (
        kwargs_dataset
        | kwargs_model
        | {"classifier_ma": classifier_ma, "ma_n": ma_n}
        | kwargs_opt
        | kwargs_loss
        | kwargs_filename_dvec
        | kwargs_filename_cls
    )

    # Combine validation kwargs
    kwargs_validation = (
        kwargs_dataset
        | kwargs_model
        | {"classifier_ma": classifier_ma, "ma_n": ma_n}
        | kwargs_loss
    )

    # Logging
    logger = get_logger()

    ####################################################################
    ########      For the existing set of optimal buckets      #########
    ####################################################################
    opt_unique_bkt, indx_opt_unique_bkt = create_unique_opt_bkt_spks_existing(
        args,
        hparams.round_num,
    )

    opt_unique_bkt_sofar, indx_opt_unique_bkt_sofar = create_unique_opt_bkt_spks_sofar(
        args,
        hparams.round_num,
    )

    if len(opt_unique_bkt) != 0:
        print(
            f"The existing optimal bucket(s) for DP current round:{hparams.round_num}"
        )

    ####################################################################
    ###     For the optimal set of initial buckets (if required)     ###
    ####################################################################
    if len(opt_unique_bkt) == 0:
        opt_unique_bkt, indx_opt_unique_bkt = create_unique_opt_bkts_spks(
            dvectors,
            args,
            hparams,
            dataset_validation,
            dataset_validation_other,
            dataset_validation_other,
            device,
            compute_opt_bkt_final,
            unique_opt_seq_final,
        )

    # Create per round list of speakers per buckets so far
    # and the number of new registrations list per buckets
    spk_per_bkt_storage, spk_per_bkt_reg_storage = per_round_spks_per_bkts_storage(
        args,
        buckets,
        opt_unique_bkt_sofar,
        opt_unique_bkt,
    )

    # Create path file names for saving the metrics per round
    paths_filenames = create_filenames_reg_supervised_results(
        args,
        hparams.ma_mode,
        args.max_mem,
        args.epochs_per_dvector,
        args.epochs_per_cls,
        hparams.round_num,
        hparams.pcnt_old,
        args.agnt_num,
    )

    # Create moving average collection of functions
    moving_average_collection = create_moving_average_collection(
        swa_scheduling,
        no_ma_scheduling,
    )

    # Initialze early stopping status per buckets
    early_stopping_status = {bkt: False for bkt in buckets}

    # Initialize the elapsed time per epoch per round
    td_per_epoch_per_round = []

    # Initialize validation accuracies to be saved during registrations
    val_acc_opt_bkt = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}

    # Main registration loop
    for epoch in range(
        start_epoch[hparams.round_num], start_epoch[hparams.round_num] + args.epoch
    ):

        if len(opt_unique_bkt) != 0:

            # Register new speakers per round per epoch
            td_train = train_reg_per_round_per_epoch(
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
            )

            td_per_epoch_per_round.append(td_train)

            # Moving average strategy
            moving_average_collection[hparams.ma_mode](
                swa_start=args.swa_start,
                swa_lr=args.swa_lr,
                lr_cls=hparams.lr_cls,
                epochs=args.epoch,
                moving_average=moving_average,
                **kwargs_training,
            )

            # Evaluate the performance after registration of new speakers per round per epoch
            val_acc_round = eval_reg_progressive_per_round_per_epoch_sup(
                args,
                device,
                outputs,
                buckets,
                opt_unique_bkt_sofar,
                indx_opt_unique_bkt_sofar,
                opt_unique_bkt,
                indx_opt_unique_bkt,
                epoch,
                val_acc_opt_bkt,
                **kwargs_validation,
            )

            # Scheduling
            # Metric-based
            # scheduler_metric(val_out_round["val_loss"])

            # Update early stopping parameters for the optimal buckets per round
            if args.early_stopping:

                for _, bkt_id in enumerate(buckets):
                    early_stopping[bkt_id](
                        torch.tensor(val_acc_round[bkt_id]).view(-1)[-1],
                        epoch,
                        bkt_id,
                    )

                    if kwargs_training["early_stop"][bkt_id].early_stop:
                        early_stopping_status[bkt_id] = True
                        logger.info(f"Training of the bucket:{bkt_id} completed.")

            # Save the required validation metrics as JSON files
            save_as_json(
                paths_filenames["dir_acc_val"],
                paths_filenames["filename_acc_val"],
                val_acc_round,
            )

            # Break training if the early stopping status is ``True'' after completion of progressive registrations
            if early_stopping_status[hparams.num_of_buckets - 1]:
                break

        else:
            print("Registrations completed.")
            break

    # Save the overal elapsed time as a JSON file
    save_as_json(
        paths_filenames["dir_td"],
        paths_filenames["filename_time_delay"],
        td_per_epoch_per_round,
    )
