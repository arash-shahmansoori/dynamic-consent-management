import numpy as np
import torch

from torch.optim import SGD
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from pathlib import Path

from utils import (
    HyperParams,
    AttentivePooledLSTMDvector,
    SpeakerClassifierRec_v2,
    DvecModel,
    cor_seq_counter_list,
    create_filenames_dvec,
    DvecGeneral,
    DvecOptimizer,
    SupConLoss,
    save_as_json,
)

from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    SubDatasetGdrSpk,
    collateGdrSpkr,
    create_dataset_arguments,
)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve

from .lib_performance import (
    compute_minDcf,
    fast_actDCF,
    effective_prior,
)
from .compute_cllr import cllr_v2, min_cllr_v2
from .lib_math import optimal_llr
from .pav_rocch import PAV


def compute_true_negative(tp):

    tn = []
    for i in range(len(tp)):
        tpx = torch.tensor([x for ix, x in enumerate(tp) if ix != i]).view(-1)
        tn.append(torch.sum(tpx))

    return torch.tensor(tn).view(-1)


def verification_performance_sup(
    args,
    hparams: HyperParams,
    buckets,
    device,
    ckpt_cls=None,
):
    """EER to verify the speaker registration in a certain bucket.
    This function computes EER, and DCF based metrics.
    """

    # Dictionaries of filenames for the checkpoints of dvectors and classifier
    filenames_dvecs_and_dirs = create_filenames_dvec(buckets, args, hparams)

    # Create the dataset for testing
    test_data_dir, speaker_infos = create_dataset_arguments(
        args,
        args.validation_data_dir,
    )
    dataset = ClassificationDatasetGdrSpkr(
        test_data_dir,
        speaker_infos,
        args.nt_utterances_labeled,
        args.seg_len,
    )

    # Create directories for saving the metrics
    result_dir_eer = args.result_dir_eer
    result_dir_eer_path = Path(result_dir_eer)
    result_dir_eer_path.mkdir(parents=True, exist_ok=True)

    labels = [i for i in range(args.n_speakers)]

    outputs = cor_seq_counter_list(
        len(labels),
        args.spk_per_bucket,
        args.spk_per_bucket,
    )

    # Build the models for d-vectors and load the available checkpoints for the buckets
    dvec_model_obj = DvecModel(device, buckets, args)
    dvec_opt_obj = DvecOptimizer(device, buckets, args, hparams)

    model_dvec = DvecGeneral(dvec_model_obj, dvec_opt_obj, SGD, device, buckets, args)
    dvectors, _, _ = model_dvec.load_model_opt(
        hparams,
        AttentivePooledLSTMDvector,
        SupConLoss,
        filenames_dvecs_and_dirs["filename_dvec"],
    )

    for _, bkt_id in enumerate(buckets):
        dvectors[bkt_id].eval()

    # Build the model for the classifier and load the available checkpoints
    classifier = SpeakerClassifierRec_v2(args).to(device)

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_cls is not None:
        ckpt_cls = torch.load(ckpt_cls)
        classifier.load_state_dict(ckpt_cls[hparams.model_str])
        classifier.eval()

    fpr, fnr, tpr, thresholds = dict(), dict(), dict(), dict()
    tar_llrs, nontar_llrs, tars, nontars = dict(), dict(), dict(), dict()
    _y_spk, _scores = dict(), dict()
    pav = dict()

    Cmin_llr_list = []

    eer_list_micro, eer_list_macro = [], []
    min_dcf_list_micro, min_dcf_list_macro = [], []

    emb_spk_storage, spk_storage = [], []
    for _ in range(args.epoch_test):

        # Main loop for evaluating the performance of contrastive CL
        for _, bucket_id in enumerate(buckets):
            sub_labels = outputs[bucket_id]

            sub_dataset = SubDatasetGdrSpk(dataset, sub_labels)
            sub_loader = DataLoader(
                sub_dataset,
                batch_size=len(sub_labels),
                collate_fn=collateGdrSpkr,
                drop_last=True,
                pin_memory=True,
            )

            mel_db_batch = next(iter(sub_loader))

            x, _, spk = mel_db_batch
            x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)

            emb_spk = dvectors[bucket_id](x)
            emb_spk_storage.append(emb_spk.detach())

            spk_storage.append(spk.to(device))

    emb_spk_buffer = torch.stack(emb_spk_storage, dim=0).view(-1, args.dim_emb)
    t_buffer = torch.stack(spk_storage, dim=0)

    y_spk = label_binarize(
        t_buffer.view(-1).cpu(),
        classes=[i_ for i_ in range(args.n_speakers)],
    )

    scores, logits = classifier(emb_spk_buffer)

    # Compute fpr, tpr, threshold, tar_llrs, nontar_llrs after progressive bucket registrations
    for i in range(args.n_speakers):
        (
            fpr[i],
            tpr[i],
            thresholds[i],
        ) = roc_curve(y_spk[:, i], scores[:, i].cpu().detach(), pos_label=1)
        fnr[i] = 1 - tpr[i]

        # tar_llrs[i], nontar_llrs[i] = optimal_llr(
        #     logits[(logits[:, i] > 0), i].cpu().detach(),
        #     logits[(logits[:, i] <= 0), i].cpu().detach(),
        #     laplace=True,
        #     monotonicity_epsilon=1e-6,
        # )

        tars[i], nontars[i] = (
            logits[(logits[:, i] > 0), i].cpu().detach(),
            logits[(logits[:, i] <= 0), i].cpu().detach(),
        )

        _y_spk[i], _scores[i] = y_spk[:, i], scores[:, i].cpu().detach()

    # Aggregate all tar_llrs and nontar_llrs
    # all_tar_llrs = np.concatenate([tar_llrs[i] for i in range(args.n_speakers)])
    # all_nontar_llrs = np.concatenate([nontar_llrs[i] for i in range(args.n_speakers)])

    all_tars = np.concatenate([tars[i] for i in range(args.n_speakers)])
    all_nontars = np.concatenate([nontars[i] for i in range(args.n_speakers)])

    all_scores = np.concatenate([_scores[i] for i in range(args.n_speakers)])
    all_y_spk = np.concatenate([_y_spk[i] for i in range(args.n_speakers)])

    # Compute C_llr
    # C_llr_v2 = cllr_v2(all_tar_llrs, all_nontar_llrs, deriv=True)
    # C_llr = C_llr_v2[0]

    pav = PAV(all_scores, all_y_spk)
    Cmin_llr_v2 = min_cllr_v2(pav)
    Cmin_llr_list.append(Cmin_llr_v2)

    # Compute Actual DCF, and minDCF using another formulation
    plo = effective_prior(1e-2, 1, 1)
    # _actDCF = fast_actDCF(all_tars, all_nontars, plo, normalize=True)
    # actDCF = torch.tensor(_actDCF)

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.n_speakers)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(args.n_speakers):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= args.n_speakers

    fnr_macro = 1 - mean_tpr

    _, eer_threshold_macro = torch.min(
        abs(torch.tensor(fnr_macro) - torch.tensor(all_fpr)),
        dim=0,
    )

    eer_macro = 100 * all_fpr[eer_threshold_macro.item()]
    eer_list_macro.append(eer_macro)

    min_dcf_macro, _ = compute_minDcf(
        fnr_macro,
        all_fpr,
        thresholds,
        plo,
        c_miss=1,
        c_fa=1,
    )
    min_dcf_list_macro.append(min_dcf_macro)

    # (
    #     fpr_micro,
    #     tpr_micro,
    #     thresholds_micro,
    # ) = roc_curve(y_spk.ravel(), scores.cpu().detach().ravel())

    # fnr_micro = 1 - tpr_micro
    # _, eer_threshold_micro = torch.min(
    #     abs(torch.tensor(fnr_micro) - torch.tensor(fpr_micro)),
    #     dim=0,
    # )

    # eer_micro = 100 * fpr_micro[eer_threshold_micro.item()]
    # eer_list_micro.append(eer_micro)

    # min_dcf_micro, _ = compute_minDcf(
    #     fnr_micro,
    #     fpr_micro,
    #     thresholds_micro,
    #     plo,
    #     c_miss=1,
    #     c_fa=1,
    # )
    # min_dcf_list_micro.append(min_dcf_micro)

    print(
        f"EER_macro:{eer_macro:.4f}, "
        f"minDCF:{min_dcf_macro:.4f}, Cmin_llr:{Cmin_llr_v2:.4f}, "
    )

    # file_name_eer_sup_macro = f"EER_Macro_sup_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.json"
    # file_name_dcf_sup_macro = f"EER_Macro_sup_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.json"
    # file_name_cllr_sup_macro = f"EER_Macro_sup_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.json"

    # save_as_json(result_dir_eer_path, file_name_eer_unsup_macro, eer_list_macro)
    # save_as_json(result_dir_eer_path, file_name_dcf_unsup_macro, min_dcf_list_macro)
    # save_as_json(result_dir_eer_path, file_name_cllr_unsup_macro, Cmin_llr_list)
