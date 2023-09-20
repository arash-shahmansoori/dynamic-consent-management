from .utils_args import parse_args
from .utils_hyper_params import HyperParams

from .utils_metric import num_correct
from .utils_logger import get_logger

from .utils_functions import (
    cor_seq_counter_list,
    cor_seq_counter,
    unreg_spks_per_bkts,
    create_calibrated_length,
    normalize_per_bkt_labels,
    label_normalizer_per_bucket,
    label_normalizer_progressive,
    progressive_indx_normalization,
    Progressive_normalized_label,
    customized_labels_unreg_unsup,
    customized_labels_unreg_unsup_bkt,
)
from .utils_euclidean_distance import pairwise_distances, compute_prototypes

from .utils_models import (
    AttentivePooledLSTMDvector,
    AttentivePooledLSTMDvectorLiterature,
    SpeakerClassifierRec,
    SpeakerClassifierRec_DR,
    SpeakerClassifierRec_v2,
    SpeakerClassifierRec_DR_v2,
    SpeakerClassifierE2E,
    SpeakerClassifierE2ESupervisedV2,
    SpeakerClassifierE2EUnsupervised,
    SpeakerClassifierE2EUnsupervisedV2,
    UnsupClsLatent,
    UnsupClsLatentDR,
    moving_average,
)

from .utils_build_load import (
    DvecModel,
    DvecOptimizer,
    DvecModelUnsupervised,
    DvecOptimizerUnsupervised,
    DvecOptimizerUnRegUnsupervised,
    DvecGeneral,
    DvecGeneralUnsupervised,
    DvecModelDynamicReg,
    DvecModelDynamicReReg,
    DvecModelDynamicRegUnsupervised,
    DvecModelDynamicUnRegUnsupervised,
    DvecModelDynamicReRegUnsupervised,
    DvecGeneralDynamicRegUnsupervised,
    DvecGeneralDynamicReRegUnsupervised,
    DvecGeneralDynamicReg,
    DvecModelDynamicUnReg,
    DvecGeneralDynamicReReg,
    model_loader_dvec,
    model_loader_dvec_latent,
    model_loader_dvec_latent_dynamic_reg,
    model_loader_dvec_per_bkt,
    model_loader_dvec_dynamic_reg_per_bkt,
    cont_loss_loader_dvec_latent,
    cont_loss_loader_dvec_latent_dynamic_reg,
    contloss_loader_per_bkt,
    cont_loss_loader_dvec_dynamic_reg_per_bkt,
    opt_loader_dvec_per_bkt,
    dvec_model_loader_dynamic_unreg,
)

from .utils_losses import (
    GE2ELoss,
    GE2ELossLatent,
    GE2ELossStable,
    GE2ELossB,
    GE2ELossSup,
    AngleProtoLoss,
    AngProtoLossStable,
    SupConLoss,
    StableSupContLoss,
)
from .utils_sophia import SophiaG


from .utils_per_round_spks_per_bkts import per_round_spks_per_bkts_storage
from .utils_compute_spks_per_bkts import compute_spks_per_bkts_storage
from .utils_filenames import (
    create_filenames_dvec,
    create_filenames_dvec_unreg_spks_bkt,
    create_filenames_dvec_dynamic_scratch,
    create_filenames_dvec_unsup_dynamic_scratch,
    create_filenames_dvec_unsupervised,
    create_filenames_dvec_unsupervised_latent,
    create_filenames_dvec_unsupervised_vox,
    create_filenames_dvec_unsupervised_latent_vox,
    create_filenames_dvec_vox,
    create_filenames_dvec_vox_v2,
    create_filenames_cls,
    create_filenames_cls_vox,
    create_filenames_cls_vox_v2,
    create_filenames_cls_dynamic_scratch,
    create_filenames_dvec_latent_dynamic_scratch,
    create_filenames_scratch_vox,
    create_filenames_scratch_unsupervised_vox,
    create_filenames_scratch_unsupervised_proto_vox,
    create_cls_checkpoint_dir,
    create_cls_scratch_checkpoint_dir,
    create_cls_checkpoint_dir_reg,
    create_cls_checkpoint_dir_dynamic_reg_scratch,
    create_dvec_latent_checkpoint_dir_dynamic_reg_scratch,
    create_cls_checkpoint_dir_unreg,
    create_cls_checkpoint_dir_re_reg,
    create_dvec_checkpoint_dir_unsup,
    create_cls_checkpoint_dir_reg_unsup,
    create_dvec_latent_scratch_checkpoint_dir,
    create_dvec_latent_checkpoint_dir,
    create_dvec_latent_checkpoint_dir_unsup_unreg,
    create_dvec_latent_checkpoint_dir_unsup_re_reg,
    create_filenames_modular_cls,
    create_filenames_scratch,
    create_filenames_scratch_unsupervised,
    create_filenames_scratch_unsupervised_v2,
    create_filenames_bkts_json,
    create_filenames_results,
    create_filenames_results_sup_scratch,
    create_filenames_dynamic_reg_results,
    create_filenames_unreg_results,
    create_filenames_re_reg_results,
    create_filenames_results_vox,
    create_filenames_results_scratch_vox,
    create_filenames_unsupervised_results_vox,
    create_filenames_unsupervised_results_scratch_vox,
    create_filenames_dynamic_reg_unsup_results,
    create_filename_dynamic_reg_td_results,
    create_filenames_tsne_results,
    create_filenames_tsne_unsup_results,
    create_filenames_unreg_tsne_results,
    create_filenames_unsupervised_results,
    create_filenames_unsupervised_results_v2,
    create_filenames_reg_supervised_results,
    create_filenames_reg_supervised_scratch_results,
    create_filenames_reg_unsupervised_scratch_results,
    create_filenames_reg_unsupervised_results,
    create_filenames_unreg_unsup_results,
    create_filenames_re_reg_unsup_results,
    create_filenames_modular_results,
    create_moving_average_collection,
)
from .utils_strategy_filenames import (
    create_strategy_filenames,
    create_filenames_dynamic_reg_causal_results,
    create_filenames_dynamic_reg_sup_causal_results,
    create_filenames_dynamic_reg_unsup_causal_results,
    strategy_per_bkt_indx,
)
from .utils_folder_file_copy import create_spks_per_agnt_dataset
from .utils_tsne import tsne
from .utils_save_ckpts_metrics import (
    save_model_ckpt_dvec,
    save_model_ckpt_cls,
    save_model_ckpt_scratch_cls,
    save_as_json,
)


from .utils_time_decorator import custom_timer, custom_timer_with_return

from .utils_kwargs import (
    dataset_kwargs,
    dataset_spk_kwargs,
    model_kwargs,
    model_kwargs_unsupervised,
    model_kwargs_unsupervised_unreg,
    model_unsupervised_kwargs,
    opt_kwargs,
    loss_kwargs,
    loss_kwargs_unsupervised,
    filename_kwargs_dvec,
    filename_kwargs_cls,
    filename_kwargs_scratch,
)
from .utils_custom_confusion_matrix import (
    custom_confusion_matrix,
    normalize_custom_confusion_matrix,
)
