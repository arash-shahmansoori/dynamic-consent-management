import torch


from torch.optim import Adam

from utils import (
    SophiaG,
    HyperParams,
    cor_seq_counter_list,
    DvecModelUnsupervised,
    DvecOptimizerUnsupervised,
    DvecGeneralUnsupervised,
    AttentivePooledLSTMDvector,
    UnsupClsLatent,
    get_logger,
    dataset_spk_kwargs,
    model_kwargs_unsupervised,
    opt_kwargs,
    loss_kwargs_unsupervised,
    # filename_kwargs_dvec,
    # filename_kwargs_cls,
    save_as_json,
    create_filenames_dvec_unsupervised_vox,
    create_filenames_dvec_unsupervised_latent_vox,
    create_filenames_unsupervised_results_vox,
    # moving_average,
    # create_moving_average_collection,
)


from .train_epoch_cont_unsupervised_selective_vox import (
    train_per_epoch_contrastive_unsupervised_selective_vox,
)
from evaluation import eval_per_epoch_progressive_contrastive_unsupervised_vox
from scheduler_early_stop import (
    EarlyStoppingCustomLoss,
    # swa_scheduling_unsup,
    # no_ma_scheduling,
)
from preprocess_data import (
    ClassificationDatasetSpkr,
    ClassificationDatasetSpkrV2,
    SubDatasetSpk,
    collateSpkr,
    create_dataset_speaker_arguments,
)
from create_buffer import CreateMultiStridedSamplesV2
from agent import AgentUnSupervised


def train_contrastive_unsupervised_vox(
    args,
    hparams: HyperParams,
    buckets,
    cont_loss_feat,
    cont_loss_latent,
    device,
    ckpt_dvec_latent=None,
):
    # Dictionaries of filenames for the checkpoints of dvectors and latent features
    filenames_dvec_and_dirs = create_filenames_dvec_unsupervised_vox(
        buckets,
        args,
        hparams,
        cont_loss_feat,
    )
    filenames_and_dirs = create_filenames_dvec_unsupervised_latent_vox(
        args,
        hparams,
        cont_loss_feat,
    )

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_unsupervised_results_vox(
        args,
        hparams.ma_mode,
        args.max_mem_unsup,
        args.spk_per_bucket,
        hparams.train_dvec_mode,
        args.agnt_num,
        cont_loss_feat,
    )

    # # Create moving average collection of functions
    # moving_average_collection = create_moving_average_collection(
    #     swa_scheduling_unsup,
    #     no_ma_scheduling,
    # )

    # Create the index list of speakers
    labels = [i for i in range(args.n_speakers)]

    outputs = cor_seq_counter_list(
        len(labels),
        args.spk_per_bucket,
        args.spk_per_bucket,
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
    # dataset_validation = ClassificationDatasetSpkr(
    #     validation_data_dir,
    #     speaker_infos_validation,
    #     args.n_test_vox_utts,
    #     args.seg_len,
    # )

    # Build the models for d-vectors and load the available checkpoints for the buckets
    dvec_model_obj = DvecModelUnsupervised(device, buckets, args)
    dvec_opt_obj = DvecOptimizerUnsupervised(device, buckets, args, hparams)

    model_dvec = DvecGeneralUnsupervised(
        dvec_model_obj,
        dvec_opt_obj,
        SophiaG,
        device,
        buckets,
        args,
    )
    dvectors, cont_losses, opt_dvecs, _ = model_dvec.load_model_opt(
        hparams,
        AttentivePooledLSTMDvector,
        cont_loss_feat,
        filenames_dvec_and_dirs["filename_dvec"],
    )

    # d-vec in the latent space to be trained on the contrastive embedding replay
    dvec_latent = UnsupClsLatent(args).to(device)

    # Create the moving average model
    dvec_latent_ma = UnsupClsLatent(args).to(device)
    ma_n = 0

    # Unsupervised contrastive loss for the latent space
    contrastive_loss_latent = cont_loss_latent(args).to(device)

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_dvec_latent is not None:
        ckpt_dvec_latent = torch.load(ckpt_dvec_latent)
        dvec_latent.load_state_dict(ckpt_dvec_latent[hparams.model_str])
        contrastive_loss_latent.load_state_dict(ckpt_dvec_latent[hparams.contloss_str])

        start_epoch_available = ckpt_dvec_latent.get(hparams.start_epoch)

        if start_epoch_available:
            start_epoch = start_epoch_available + 1
        else:
            start_epoch = 0

        dvec_latent_ma.load_state_dict(ckpt_dvec_latent[hparams.model_ma_str])

        if hparams.ma_mode == "swa":
            ma_n = ckpt_dvec_latent[hparams.ma_n_str]
    else:
        start_epoch = 0

    # Initializing early stoppings for the buckets
    early_stopping = {bucket_id: [] for _, bucket_id in enumerate(buckets)}
    for _, bucket_id in enumerate(buckets):
        if args.early_stopping:
            early_stopping[bucket_id] = EarlyStoppingCustomLoss(args)

    # Instantiate the Agent class
    agent = AgentUnSupervised(args, device, hparams)

    # Initialize the buffer class
    create_buffer = CreateMultiStridedSamplesV2(args)

    # Create kwargs for the training/validation function
    kwargs_dataset = dataset_spk_kwargs(
        SubDatasetSpk,
        collateSpkr,
        dataset,
        dataset_validation,
    )

    kwargs_model = model_kwargs_unsupervised(agent, dvectors, dvec_latent)
    kwargs_filename_dvec = filenames_dvec_and_dirs
    kwargs_filename_cls = filenames_and_dirs
    kwargs_opt = opt_kwargs(SophiaG, opt_dvecs, SophiaG, None, early_stopping)
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

    # Initialize validation accuracy and loss to be saved
    val_loss_cont = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    val_acc_cont = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}

    train_loss = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    train_acc = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}

    # Initialize the elapsed time per epoch
    td_per_epoch = []

    for epoch in range(start_epoch, start_epoch + args.epoch):
        # Train the d-vectors per epoch and evaluate the performance
        td, train_out = train_per_epoch_contrastive_unsupervised_selective_vox(
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
            early_stopping,
            **kwargs_training_val,
        )

        # Store the elapsed time per epoch in a list
        td_per_epoch.append(td)

        #     # Scheduling
        #     # Non-metric based
        #     # kwargs_training["lr_scheduler"].step()

        #     # Moving average strategy
        #     moving_average_collection[hparams.ma_mode](
        #         swa_start=args.swa_start,
        #         swa_lr=args.swa_lr,
        #         lr_cls=hparams.lr_cls,
        #         epochs=args.epoch,
        #         moving_average=moving_average,
        #         **kwargs_training_val,
        #     )

        # # Evaluate the performance per epoch
        # val_out = eval_per_epoch_progressive_contrastive_unsupervised_vox(
        #     hparams,
        #     args,
        #     device,
        #     outputs,
        #     buckets,
        #     val_loss_cont,
        #     val_acc_cont,
        #     epoch,
        #     **kwargs_training_val,
        # )

        #     # Scheduling
        #     # Metric-based
        #     # scheduler_metric(val_out["val_acc_cont"])

        # Update early stopping parameters for the buckets
        if args.early_stopping:
            for _, bkt_id in enumerate(buckets):
                # if bkt_id == buckets[-1]:
                # early_stopping[bkt_id](
                #     torch.tensor(val_out["val_acc"][bkt_id]).view(-1)[-1],
                #     epoch,
                #     bkt_id,
                # )
                # early_stopping[bkt_id](
                #     torch.tensor(val_out["val_acc_cont"][bkt_id]).view(-1)[-1],
                #     torch.tensor(val_out["val_loss_cont"][bkt_id]).view(-1)[-1],
                #     epoch,
                #     bkt_id,
                # )
                early_stopping[bkt_id](
                    torch.tensor(train_out["train_loss"][bkt_id]).view(-1)[-1],
                    epoch,
                    bkt_id,
                )

            # # Save the required validation metrics as JSON files
            # save_as_json(
            #     paths_filenames["dir_loss_cont_val"],
            #     paths_filenames["filename_loss_cont_val"],
            #     val_out["val_loss_cont"],
            # )
            # save_as_json(
            #     paths_filenames["dir_acc_cont_val"],
            #     paths_filenames["filename_acc_cont_val"],
            #     val_out["val_acc_cont"],
            # )

        # Break training if the early stopping status is ``True'' after completion of progressive registrations
        if train_out["early_stops_status"][len(buckets) - 1]:
            break

        # Save the required training metrics as JSON files
        save_as_json(
            paths_filenames["dir_loss_cont_train"],
            paths_filenames["filename_loss_cont_train"],
            train_out["train_loss"],
        )

        save_as_json(
            paths_filenames["dir_acc_cont_train"],
            paths_filenames["filename_acc_cont_train"],
            train_out["train_acc"],
        )

    # Save the overal elapsed time as a JSON file
    save_as_json(
        paths_filenames["dir_td"],
        paths_filenames["filename_time_delay"],
        td_per_epoch,
    )
