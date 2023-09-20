import torch


from torch.optim import SGD
from torch.utils.data import DataLoader
from utils import (
    HyperParams,
    AttentivePooledLSTMDvector,
    UnsupClsLatentDR,
    DvecModelDynamicRegUnsupervised,
    DvecOptimizerUnsupervised,
    DvecGeneralDynamicRegUnsupervised,
    create_filenames_dvec_unsupervised,
    GE2ELoss,
)
from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    SubDatasetGdrSpk,
    collateGdrSpkr,
    create_dataset_arguments,
)
from compute_optimal_buckets import create_unique_opt_bkt_spks_existing


def prepare_data_tsne_new_spks_unsup(
    args,
    hparams: HyperParams,
    buckets,
    round_num,
    device,
    ckpt_cls=None,
):

    # Dictionaries of filenames for the checkpoints of dvectors and classifier
    filenames_dvec_and_dirs = create_filenames_dvec_unsupervised(
        buckets,
        args,
        hparams,
    )

    opt_unique_bkt, indx_opt_unique_bkt = create_unique_opt_bkt_spks_existing(
        args,
        round_num,
    )

    # Create validation dataset
    (
        validation_data_dir_other,
        speaker_infos_validation_other,
    ) = create_dataset_arguments(args, args.validation_data_dir_other)

    dataset_validation_other = ClassificationDatasetGdrSpkr(
        validation_data_dir_other,
        speaker_infos_validation_other,
        args.nt_utterances_labeled,
        args.seg_len,
    )

    # Build the models for d-vectors and load the available checkpoints for the buckets
    dvec_model_obj = DvecModelDynamicRegUnsupervised(device, buckets, args)
    dvec_opt_obj = DvecOptimizerUnsupervised(device, buckets, args, hparams)

    model_dvec_obj = DvecGeneralDynamicRegUnsupervised(
        dvec_model_obj,
        dvec_opt_obj,
        SGD,
        device,
        buckets,
        args,
    )

    dvectors, _, _, _ = model_dvec_obj.load_model_opt(
        hparams,
        AttentivePooledLSTMDvector,
        GE2ELoss,
        filenames_dvec_and_dirs["filename_dvec"],
        filenames_dvec_and_dirs["filename_dvec_reg"],
    )

    for _, bkt_id in enumerate(buckets):
        dvectors[bkt_id].eval()

    # d-vec in the latent space to be trained on the contrastive embedding replay
    dvec_latent = UnsupClsLatentDR(args).to(device)

    # Create the moving average model
    dvec_latent_ma = UnsupClsLatentDR(args).to(device)

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_cls is not None:
        ckpt_cls = torch.load(ckpt_cls)
        dvec_latent.load_state_dict(ckpt_cls[hparams.model_str])
        dvec_latent_ma.load_state_dict(ckpt_cls[hparams.model_ma_str])

    # Visualize feature maps
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    dvec_latent.hidden.register_forward_hook(get_activation("hidden"))

    act_storage, xe_val_storage = [], []
    for _ in range(args.epoch_test):

        # for indx_selected_id, bucket_id in enumerate(opt_unique_bkt):
        for _, bucket_id in enumerate(buckets):
            if bucket_id in opt_unique_bkt:
                spk_selected_strategy = opt_unique_bkt.index(bucket_id)
                sub_lbs_current_validation = [
                    indx_opt_unique_bkt[spk_selected_strategy]
                ]

                sub_dataset_current_validation = SubDatasetGdrSpk(
                    dataset_validation_other,
                    sub_lbs_current_validation,
                )
                validation_loader_current = DataLoader(
                    sub_dataset_current_validation,
                    batch_size=len(sub_lbs_current_validation),
                    collate_fn=collateGdrSpkr,
                    drop_last=True,
                )

                mel_db_batch_validation = next(iter(validation_loader_current))

                x_val, _, spk_val = mel_db_batch_validation
                x_val = x_val.reshape(-1, args.seg_len, args.feature_dim)
                x_val, spk_val = x_val.to(device), spk_val.to(device)

                input_val_data = {"x_val": x_val, "y_val": spk_val}

                xe_val = dvectors[bucket_id](input_val_data["x_val"]).detach()

                # Store contrastive embeddings for validation
                xe_val_storage.append(xe_val)

    x_val_buffer = torch.cat(xe_val_storage, dim=0).view(-1, args.dim_emb)

    _ = dvec_latent(x_val_buffer)

    act = activation["hidden"]

    act_storage.append(act)

    embs = torch.cat(act_storage, dim=0).view(-1, args.latent_dim).tolist()

    return embs
