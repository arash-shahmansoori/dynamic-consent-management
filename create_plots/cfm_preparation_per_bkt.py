import torch

from torch.optim import SGD
from utils import (
    create_filenames_dvec,
    cor_seq_counter_list,
    unreg_spks_per_bkts,
    compute_spks_per_bkts_storage,
    DvecModelDynamicUnReg,
    DvecOptimizer,
    DvecGeneralDynamicReg,
    AttentivePooledLSTMDvector,
    SpeakerClassifierRec_v2,
    SupConLoss,
    dataset_kwargs,
    model_kwargs,
)
from agent import AgentSupervised
from evaluation import eval_per_epoch_per_bucket_contrastive_supervised
from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    SubDatasetGdrSpk,
    collateGdrSpkr,
    create_dataset_arguments,
)


def prepare_cfm_per_bkt(
    args,
    hparams,
    buckets,
    device,
    unreg_spks,
    ckpt_cls=None,
):

    # Dictionaries of filenames for the checkpoints of dvectors and classifier
    filenames_dvecs_and_dirs = create_filenames_dvec(buckets, args, hparams, unreg_spks)

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

    # Create validation dataset
    validation_data_dir, speaker_infos_validation = create_dataset_arguments(
        args,
        args.validation_data_dir,
    )

    dataset_validation = ClassificationDatasetGdrSpkr(
        validation_data_dir,
        speaker_infos_validation,
        args.nv_utterances_unlabeled,
        args.seg_len,
    )

    # Build the models for d-vectors and load the available checkpoints for the buckets
    dvec_model_obj = DvecModelDynamicUnReg(device, buckets, unreg_bkts_storage, args)
    dvec_opt_obj = DvecOptimizer(device, buckets, args, hparams)

    model_dvec = DvecGeneralDynamicReg(
        dvec_model_obj,
        dvec_opt_obj,
        SGD,
        device,
        buckets,
        args,
    )

    dvectors, _, _ = model_dvec.load_model_opt(
        hparams,
        AttentivePooledLSTMDvector,
        SupConLoss,
        filenames_dvecs_and_dirs["filename_dvec"],
        filenames_dvecs_and_dirs["filename_dvec_unreg"],
    )

    # Classifier to be evaluated
    classifier = SpeakerClassifierRec_v2(args).to(device)

    # Instantiate the Agent class
    agent = AgentSupervised(args, device, hparams)

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_cls is not None:
        ckpt_cls = torch.load(ckpt_cls)
        classifier.load_state_dict(ckpt_cls[hparams.model_str])

    # Create kwargs for the validation function
    kwargs_dataset = dataset_kwargs(
        SubDatasetGdrSpk,
        collateGdrSpkr,
        None,
        dataset_validation,
    )

    kwargs_model = model_kwargs(agent, dvectors, classifier)

    kwargs_val = kwargs_dataset | kwargs_model | {"classifier_ma": None, "ma_n": None}

    # Initialize validation accuracy and loss to be saved during training
    pred_indx = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}
    gtruth_indx = {bkt_ids: [] for _, bkt_ids in enumerate(buckets)}

    for _ in range(args.epoch_test):

        # Evaluate the performance per epoch per bucket
        val_out_per_bucket = eval_per_epoch_per_bucket_contrastive_supervised(
            args,
            device,
            outputs,
            buckets,
            pred_indx,
            gtruth_indx,
            **kwargs_val,
        )

    return val_out_per_bucket
