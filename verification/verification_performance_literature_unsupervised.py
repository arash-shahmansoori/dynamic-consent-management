import numpy as np
import torch


from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from pathlib import Path

from utils import (
    HyperParams,
    SpeakerClassifierE2EUnsupervisedV2,
    GE2ELossLatent,
    save_as_json,
)

from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    collateGdrSpkr,
    create_dataset_arguments,
)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
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


def customize_stack(num_speakers, seg_len, feat_dim, feat, label):

    feat_custom_storage, label_custom_storage = [], []
    for i in range(num_speakers):
        feat_custom_storage.append(feat[:, i, :].reshape(-1))
        label_custom_storage.append(label[:, i, :].reshape(-1))

    return (
        torch.cat(feat_custom_storage, dim=0).reshape(-1, seg_len, feat_dim),
        torch.cat(label_custom_storage, dim=0).reshape(-1),
    )


def verification_performance_literature_unsup(
    args,
    hparams: HyperParams,
    device,
    ckpt_dvec_latent=None,
):
    """EER to verify the speaker registration in a certain bucket.
    This function computes EER, DCF, and LLR based metrics.
    """

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

    # Build the models for d-vectors and load the available checkpoints for the buckets
    dvec_model = SpeakerClassifierE2EUnsupervisedV2(args).to(device)

    # Unsupervised contrastive loss for the latent space
    contrastive_loss_latent = GE2ELossLatent(args).to(device)

    # Load available checkpoints for the speaker recognition in latent space
    if ckpt_dvec_latent is not None:
        ckpt_dvec_latent = torch.load(ckpt_dvec_latent)
        dvec_model.load_state_dict(ckpt_dvec_latent[hparams.model_str])
        contrastive_loss_latent.load_state_dict(ckpt_dvec_latent[hparams.contloss_str])

    fpr, fnr, tpr, thresholds = dict(), dict(), dict(), dict()
    tar_llrs, nontar_llrs, tars, nontars = dict(), dict(), dict(), dict()
    _y_spk, _scores = dict(), dict()

    Cmin_llr_list = []

    eer_list_macro = []
    min_dcf_list_macro = []

    feat_spk_storage, spk_storage = [], []
    for _ in range(args.epoch_test):

        sub_loader = DataLoader(
            dataset,
            batch_size=args.n_speakers,
            collate_fn=collateGdrSpkr,
            drop_last=True,
            pin_memory=True,
        )

        mel_db_batch = next(iter(sub_loader))

        x, _, spk = mel_db_batch
        x = x.reshape(
            -1,
            args.seg_len * args.feature_dim,
            args.n_speakers,
            args.nv_utterances_unlabeled,
        ).to(device)

        feat_spk_storage.append(x)
        spk_storage.append(spk.reshape(-1).to(device))

    t_buffer = torch.cat(spk_storage, dim=0).reshape(
        args.epoch_test,
        args.n_speakers,
        args.nv_utterances_unlabeled,
    )

    feat_spk_buffer = torch.stack(feat_spk_storage, dim=0).reshape(
        args.epoch_test,
        args.n_speakers,
        args.nv_utterances_unlabeled * args.seg_len * args.feature_dim,
    )

    new_feat_spk_buffer, new_label_buffer = customize_stack(
        args.n_speakers,
        args.seg_len,
        args.feature_dim,
        feat_spk_buffer,
        t_buffer,
    )

    logits = dvec_model(new_feat_spk_buffer.reshape(-1, args.seg_len, args.feature_dim))

    cos_sim_matrix = contrastive_loss_latent.compute_similarity_matrix(
        logits.view(
            args.n_speakers,
            args.nv_utterances_unlabeled * args.epoch_test,
            args.dim_emb,
        )
    )

    y_spk = label_binarize(
        new_label_buffer.reshape(-1).cpu(),
        classes=[i_ for i_ in range(args.n_speakers)],
    )

    total_logits, total_probs = contrastive_loss_latent.logits_probs(cos_sim_matrix)

    # Compute fpr, tpr, threshold, tar_llrs, nontar_llrs after progressive bucket registrations
    for i in range(args.n_speakers):
        (
            fpr[i],
            tpr[i],
            thresholds[i],
        ) = roc_curve(y_spk[:, i], total_probs[:, i].cpu().detach(), pos_label=1)
        fnr[i] = 1 - tpr[i]

        tars[i], nontars[i] = (
            total_logits[(total_logits[:, i] > 0), i].cpu().detach(),
            total_logits[(total_logits[:, i] <= 0), i].cpu().detach(),
        )

        _y_spk[i], _scores[i] = y_spk[:, i], total_probs[:, i].cpu().detach()

    # Aggregate
    # all_tars = np.concatenate([tars[i] for i in range(args.n_speakers)])
    # all_nontars = np.concatenate([nontars[i] for i in range(args.n_speakers)])

    all_scores = np.concatenate([_scores[i] for i in range(args.n_speakers)])
    all_y_spk = np.concatenate([_y_spk[i] for i in range(args.n_speakers)])

    # # Compute C_llr
    # C_llr_v2 = cllr_v2(all_tar_llrs, all_nontar_llrs, deriv=True)
    # C_llr = C_llr_v2[0]

    # Compute Actual DCF, and minDCF using another formulation
    plo = effective_prior(1e-2, 1, 1)
    # _actDCF = fast_actDCF(all_tars, all_nontars, plo, normalize=True)
    # actDCF = torch.tensor(_actDCF).item()

    pav = PAV(all_scores, all_y_spk)
    Cmin_llr_v2 = min_cllr_v2(pav)
    Cmin_llr_list.append(Cmin_llr_v2)

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

    print(
        f"EER_macro:{eer_macro:.4f}, "
        f"minDCF:{min_dcf_macro:.4f}, Cmin_llr:{Cmin_llr_v2:.4f}, "
    )

    # file_name_eer_unsup_macro = f"EER_Macro_unsup_literature_agnt_{args.agnt_num}.json"
    # file_name_dcf_unsup_macro = f"EER_Macro_unsup_literature_agnt_{args.agnt_num}.json"
    # file_name_cllr_unsup_macro = f"EER_Macro_unsup_literature_agnt_{args.agnt_num}.json"

    # save_as_json(result_dir_eer_path, file_name_eer_unsup_macro, eer_list_macro)
    # save_as_json(result_dir_eer_path, file_name_dcf_unsup_macro, min_dcf_list_macro)
    # save_as_json(result_dir_eer_path, file_name_cllr_unsup_macro, Cmin_llr_list)
