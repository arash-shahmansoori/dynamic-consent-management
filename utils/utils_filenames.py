import os
from pathlib import Path


def create_filenames_dvec(buckets, args, hparams, unreg_spks=[]):
    """Create the dictionaries of the d-vectors checkpoints' filenames in the supervised setting."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_dvec, filename_dvec_dir = dict(), dict()
    filename_dvec_reg, filename_dvec_reg_dir = dict(), dict()
    filename_dvec_unreg, filename_dvec_unreg_dir = dict(), dict()
    filename_dvec_re_reg, filename_dvec_re_reg_dir = dict(), dict()

    for bkt_id in buckets:
        filename_dvec[
            bkt_id
        ] = f"ckpt_dvec_{bkt_id}_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_reg[
            bkt_id
        ] = f"ckpt_dvec_dp_opt_reg_in_{bkt_id}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_unreg[
            bkt_id
        ] = f"ckpt_dvec_unreg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_re_reg[
            bkt_id
        ] = f"ckpt_dvec_re_reg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_dvec_{bkt_id}_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_reg_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_dvec_dp_opt_reg_in_{bkt_id}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_unreg_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_dvec_unreg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_re_reg_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_dvec_re_reg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    file_names_and_dirs = {
        "filename_dvec": filename_dvec,
        "filename_dvec_reg": filename_dvec_reg,
        "filename_dvec_unreg": filename_dvec_unreg,
        "filename_dvec_re_reg": filename_dvec_re_reg,
        "filename_dvec_dir": filename_dvec_dir,
        "filename_dvec_dir_reg": filename_dvec_reg_dir,
        "filename_dvec_dir_unreg": filename_dvec_unreg_dir,
        "filename_dvec_dir_re_reg": filename_dvec_re_reg_dir,
    }

    return file_names_and_dirs


def create_filenames_dvec_vox(buckets, args, hparams):
    """Create the dictionaries of the d-vectors checkpoints' filenames in the supervised setting."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_dvec, filename_dvec_dir = dict(), dict()

    for bkt_id in buckets:
        filename_dvec[
            bkt_id
        ] = f"ckpt_vox_dvec_{bkt_id}_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_vox_dvec_{bkt_id}_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    file_names_and_dirs = {
        "filename_dvec": filename_dvec,
        "filename_dvec_dir": filename_dvec_dir,
    }

    return file_names_and_dirs


def create_filenames_dvec_vox_v2(buckets, args, hparams):
    """Create the dictionaries of the d-vectors checkpoints' filenames in the supervised setting."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_dvec, filename_dvec_dir = dict(), dict()

    for bkt_id in buckets:
        filename_dvec[
            bkt_id
        ] = f"ckpt_vox_dvec_{bkt_id}_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}_v2.pt"

        filename_dvec_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_vox_dvec_{bkt_id}_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}_v2.pt"

    file_names_and_dirs = {
        "filename_dvec": filename_dvec,
        "filename_dvec_dir": filename_dvec_dir,
    }

    return file_names_and_dirs


def create_filenames_dvec_unreg_spks_bkt(buckets, args, hparams, unreg_spks=[]):
    """Create the dictionaries of the d-vectors checkpoints' filenames in the supervised setting."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_dvec_unreg, filename_dvec_unreg_dir = dict(), dict()

    for bkt_id in buckets:
        filename_dvec_unreg[
            bkt_id
        ] = f"ckpt_dvec_unreg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_unreg_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_dvec_unreg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    file_names_and_dirs = {
        "filename_dvec_unreg": filename_dvec_unreg,
        "filename_dvec_dir_unreg": filename_dvec_unreg_dir,
    }

    return file_names_and_dirs


def create_filenames_cls(args, hparams, unreg_spks=[]):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_cls_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_reg = f"ckpt_cls_dp_opt_reg_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_unreg = (
        f"ckpt_cls_unreg_{unreg_spks}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    )
    filename_re_reg = (
        f"ckpt_cls_re_reg_{unreg_spks}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    )

    filename_dir = f"{checkpoint_dir_path}/ckpt_cls_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_reg_dir = f"{checkpoint_dir_path}/ckpt_cls_dp_opt_reg_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_unreg_dir = f"{checkpoint_dir_path}/ckpt_cls_unreg_{unreg_spks}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_re_reg_dir = f"{checkpoint_dir_path}/ckpt_cls_re_reg_{unreg_spks}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename": filename,
        "filename_reg": filename_reg,
        "filename_unreg": filename_unreg,
        "filename_re_reg": filename_re_reg,
        "filename_dir": filename_dir,
        "filename_dir_reg": filename_reg_dir,
        "filename_dir_unreg": filename_unreg_dir,
        "filename_dir_re_reg": filename_re_reg_dir,
    }

    return filenames_and_dirs


def create_filenames_cls_vox(args, hparams):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_vox_cls_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_dir = f"{checkpoint_dir_path}/ckpt_vox_cls_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename": filename,
        "filename_dir": filename_dir,
    }

    return filenames_and_dirs


def create_filenames_cls_vox_v2(args, hparams):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_vox_cls_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}_v2.pt"
    filename_dir = f"{checkpoint_dir_path}/ckpt_vox_cls_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}_v2.pt"

    filenames_and_dirs = {
        "filename": filename,
        "filename_dir": filename_dir,
    }

    return filenames_and_dirs


def create_filenames_modular_cls(args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir_modular
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_modular_cls_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.pt"

    filename_dir = f"{checkpoint_dir_path}/ckpt_modular_cls_spkperbkt_{args.spk_per_bucket}_stride_{args.stride}_epdvec_{args.epochs_per_dvector}_epcls_{args.epochs_per_cls}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.pt"

    return filename, filename_dir


def create_filenames_dvec_dynamic_scratch(buckets, args, hparams, round_num):
    """Create the dictionaries of the d-vectors checkpoints' filenames in the supervised setting."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_dvec_reg, filename_dvec_reg_dir = dict(), dict()

    for bkt_id in buckets:
        filename_dvec_reg[
            bkt_id
        ] = f"ckpt_dvec_scratch_dp_opt_reg_round_{round_num}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_reg_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_dvec_scratch_dp_opt_reg_round_{round_num}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    file_names_and_dirs = {
        "filename_dvec_reg": filename_dvec_reg,
        "filename_dvec_dir_reg": filename_dvec_reg_dir,
    }

    return file_names_and_dirs


def create_filenames_cls_dynamic_scratch(args, hparams, round_num):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename_reg = f"ckpt_cls_scratch_dp_opt_reg_round_{round_num}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filename_reg_dir = f"{checkpoint_dir_path}/ckpt_cls_scratch_dp_opt_reg_round_{round_num}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename_reg": filename_reg,
        "filename_dir_reg": filename_reg_dir,
    }

    return filenames_and_dirs


def create_filenames_dvec_unsup_dynamic_scratch(buckets, args, hparams, round_num):
    """Create the dictionaries of the d-vectors checkpoints' filenames in the unsupervised setting."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_dvec_reg, filename_dvec_reg_dir = dict(), dict()

    for bkt_id in buckets:
        filename_dvec_reg[
            bkt_id
        ] = f"ckpt_dvec_unsup_scratch_dp_opt_reg_round_{round_num}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_reg_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_dvec_unsup_scratch_dp_opt_reg_round_{round_num}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    file_names_and_dirs = {
        "filename_dvec_reg": filename_dvec_reg,
        "filename_dvec_dir_reg": filename_dvec_reg_dir,
    }

    return file_names_and_dirs


def create_filenames_dvec_latent_dynamic_scratch(args, hparams, round_num):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_reg = f"ckpt_dvec_latent_scratch_dp_opt_reg_round_{round_num}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filename_reg_dir = f"{checkpoint_dir_dvector_path}/ckpt_dvec_latent_scratch_dp_opt_reg_round_{round_num}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename_reg": filename_reg,
        "filename_dir_reg": filename_reg_dir,
    }

    return filenames_and_dirs


def create_filenames_scratch(args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_sup_cont_scratch_agnt_{args.agnt_num}.pt"
    filename_dir = (
        f"{checkpoint_dir_path}/ckpt_sup_cont_scratch_agnt_{args.agnt_num}.pt"
    )

    return filename, filename_dir


def create_filenames_scratch_vox(args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_cls_scratch_vox_agnt_{args.agnt_num}.pt"
    filename_dir = f"{checkpoint_dir_path}/ckpt_cls_scratch_vox_agnt_{args.agnt_num}.pt"

    return filename, filename_dir


def create_filenames_bkts_json(args, _bunch_reg_steps):
    """Create the dictionaries of the JSON filenames."""

    file_name_indx_pop_dict = {
        file_counter: [] for file_counter in range(_bunch_reg_steps)
    }
    file_name_indx_bktopt_dict = {
        file_counter: [] for file_counter in range(_bunch_reg_steps)
    }

    #######################################################################################################
    ################### For a different optimal set of initial buckets (if required) ######################
    #######################################################################################################

    for count in range(_bunch_reg_steps):
        # Use these files containing optimal buckets in each round
        file_name_indx_pop_dict[count].append(
            f"indices_removed_list_of_new_registered_speakers_agnt_{args.agnt_num}_round_{count}.json"
        )
        file_name_indx_bktopt_dict[count].append(
            f"indices_bktopt_agnt_{args.agnt_num}_round_{count}.json"
        )

        # Use the following files to reproduce the results in the paper
        # file_name_indx_pop_dict[count].append(
        #     f"indices_removed_list_of_new_registered_agnt_{args.agnt_num}_speakers{count}_v6_f.json"
        # )
        # file_name_indx_bktopt_dict[count].append(
        #     f"indices_bktopt_agnt_{args.agnt_num}_list_{count}_v6_f.json"
        # )

    return file_name_indx_pop_dict, file_name_indx_bktopt_dict


def create_filenames_dvec_unsupervised(buckets, args, hparams, unreg_spks=[]):
    """Create the dictionaries of the d-vectors checkpoints' filenames
    in the unsupervised setting."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_dvec, filename_dvec_dir = dict(), dict()
    filename_dvec_ablation, filename_dvec_dir_ablation = dict(), dict()
    filename_dvec_reg, filename_dvec_dir_reg = dict(), dict()
    filename_dvec_unreg, filename_dvec_dir_unreg = dict(), dict()
    filename_dvec_re_reg, filename_dvec_dir_re_reg = dict(), dict()

    for bkt_id in buckets:
        filename_dvec[
            bkt_id
        ] = f"ckpt_unsup_mode_{hparams.train_dvec_mode}_epdvec{args.epochs_per_dvector}_bkt_{bkt_id}_spkperbkt_{args.spk_per_bucket}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_ablation[
            bkt_id
        ] = f"ckpt_unsup_ablation_mode_{hparams.train_dvec_mode}_epdvec{args.epochs_per_dvector}_bkt_{bkt_id}_spkperbkt_{args.spk_per_bucket}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_reg[
            bkt_id
        ] = f"ckpt_unsup_dvec_dp_opt_reg_in_{bkt_id}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_unreg[
            bkt_id
        ] = f"ckpt_unsup_dvec_unreg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_re_reg[
            bkt_id
        ] = f"ckpt_unsup_dvec_re_reg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_unsup_mode_{hparams.train_dvec_mode}_epdvec{args.epochs_per_dvector}_bkt_{bkt_id}_spkperbkt_{args.spk_per_bucket}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_dir_ablation[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_unsup_ablation_mode_{hparams.train_dvec_mode}_epdvec{args.epochs_per_dvector}_bkt_{bkt_id}_spkperbkt_{args.spk_per_bucket}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_dir_reg[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_unsup_dvec_dp_opt_reg_in_{bkt_id}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_dir_unreg[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_unsup_dvec_unreg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
        filename_dvec_dir_re_reg[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_unsup_dvec_re_reg_{unreg_spks}_in_{bkt_id}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filenames_dvec_and_dirs = {
        "filename_dvec": filename_dvec,
        "filename_dvec_ablation": filename_dvec_ablation,
        "filename_dvec_reg": filename_dvec_reg,
        "filename_dvec_unreg": filename_dvec_unreg,
        "filename_dvec_re_reg": filename_dvec_re_reg,
        "filename_dvec_dir": filename_dvec_dir,
        "filename_dvec_dir_ablation": filename_dvec_dir_ablation,
        "filename_dvec_dir_reg": filename_dvec_dir_reg,
        "filename_dvec_dir_unreg": filename_dvec_dir_unreg,
        "filename_dvec_dir_re_reg": filename_dvec_dir_re_reg,
    }

    return filenames_dvec_and_dirs


def create_filenames_dvec_unsupervised_vox(buckets, args, hparams, cont_loss_feat):
    """Create the dictionaries of the d-vectors checkpoints' filenames
    in the unsupervised setting."""

    method_name = cont_loss_feat.__name__

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename_dvec, filename_dvec_dir = dict(), dict()

    for bkt_id in buckets:
        filename_dvec[
            bkt_id
        ] = f"ckpt_unsup_{method_name}_vox_mode_{hparams.train_dvec_mode}_epdvec{args.epochs_per_dvector}_bkt_{bkt_id}_spkperbkt_{args.spk_per_bucket}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

        filename_dvec_dir[
            bkt_id
        ] = f"{checkpoint_dir_dvector_path}/ckpt_unsup_{method_name}_vox_mode_{hparams.train_dvec_mode}_epdvec{args.epochs_per_dvector}_bkt_{bkt_id}_spkperbkt_{args.spk_per_bucket}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filenames_dvec_and_dirs = {
        "filename_dvec": filename_dvec,
        "filename_dvec_dir": filename_dvec_dir,
    }

    return filenames_dvec_and_dirs


def create_filenames_scratch_unsupervised(args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_cls_scratch_unsup_agnt_{args.agnt_num}.pt"
    filename_dir = (
        f"{checkpoint_dir_dvector_path}/ckpt_cls_scratch_unsup_agnt_{args.agnt_num}.pt"
    )

    return filename, filename_dir


def create_filenames_scratch_unsupervised_v2(args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_unsup_cont_scratch_agnt_{args.agnt_num}_v2.pt"
    filename_dir = f"{checkpoint_dir_dvector_path}/ckpt_unsup_cont_scratch_agnt_{args.agnt_num}_v2.pt"

    return filename, filename_dir


def create_filenames_scratch_unsupervised_vox(args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_cls_scratch_unsup_vox_agnt_{args.agnt_num}.pt"
    filename_dir = f"{checkpoint_dir_dvector_path}/ckpt_cls_scratch_unsup_vox_agnt_{args.agnt_num}.pt"

    return filename, filename_dir


def create_filenames_scratch_unsupervised_proto_vox(args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_cls_scratch_unsup_proto_vox_agnt_{args.agnt_num}.pt"
    filename_dir = f"{checkpoint_dir_dvector_path}/ckpt_cls_scratch_unsup_proto_vox_agnt_{args.agnt_num}.pt"

    return filename, filename_dir


def create_cls_checkpoint_dir_reg(
    args,
    filename,
    filename_reg,
    filename_dir,
    filename_dir_reg,
):
    """Create the available checkpoint for classifier during dynamic registrations."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            if file == filename_reg:
                status_storage.append("reg_cls")
                file_dir_storage.append(filename_dir_reg)
            elif file != filename_reg and file == filename:
                status_storage.append("cls")
                file_dir_storage.append(filename_dir)
            else:
                status_storage.append(None)
                file_dir_storage.append("")

    if filename_dir_reg in file_dir_storage:
        print(f"New registration checkpoint found")
        file_dir = filename_dir_reg
        status = "reg_cls"
    elif filename_dir in file_dir_storage:
        print(f"Base checkpoint found")
        file_dir = filename_dir
        status = "cls"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_cls_checkpoint_dir_unreg(
    args,
    filename,
    filename_unreg,
    filename_dir,
    filename_dir_unreg,
):
    """Create the available checkpoint for classifier during removing speakers."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            if file == filename_unreg:
                status_storage.append("unreg_cls")
                file_dir_storage.append(filename_dir_unreg)
            elif file != filename_unreg and file == filename:
                status_storage.append("cls")
                file_dir_storage.append(filename_dir)
            else:
                status_storage.append(None)
                file_dir_storage.append("")

    if filename_dir_unreg in file_dir_storage:
        print(f"New unregistration checkpoint found")
        file_dir = filename_dir_unreg
        status = "unreg_cls"
    elif filename_dir in file_dir_storage:
        print(f"Base checkpoint found")
        file_dir = filename_dir
        status = "cls"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_cls_checkpoint_dir_re_reg(
    args,
    filename_unreg,
    filename_re_reg,
    filename_dir_unreg,
    filename_dir_re_reg,
):
    """Create the available checkpoint for classifier during removing speakers."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            if file == filename_re_reg:
                status_storage.append("re_reg_cls")
                file_dir_storage.append(filename_dir_re_reg)
            elif file != filename_re_reg and file == filename_unreg:
                status_storage.append("unreg_cls")
                file_dir_storage.append(filename_dir_unreg)
            else:
                status_storage.append(None)
                file_dir_storage.append("")

    if filename_dir_re_reg in file_dir_storage:
        print(f"New re-registration checkpoint found")
        file_dir = filename_dir_re_reg
        status = "re_reg_cls"
    elif filename_dir_unreg in file_dir_storage:
        print(f"unregistration checkpoint found")
        file_dir = filename_dir_unreg
        status = "unreg_cls"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_dvec_latent_checkpoint_dir_unsup_re_reg(
    args,
    filename_unreg,
    filename_re_reg,
    filename_dir_unreg,
    filename_dir_re_reg,
):
    """Create the available checkpoint for classifier during removing speakers."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename_re_reg:
                status_storage.append("re_reg_dvec_latent")
                file_dir_storage.append(filename_dir_re_reg)
            elif file != filename_re_reg and file == filename_unreg:
                status_storage.append("unreg_dvec_latent")
                file_dir_storage.append(filename_dir_unreg)
            else:
                status_storage.append(None)
                file_dir_storage.append("")

    if filename_dir_re_reg in file_dir_storage:
        print(f"New re-registration checkpoint found")
        file_dir = filename_dir_re_reg
        status = "re_reg_dvec_latent"
    elif filename_dir_unreg in file_dir_storage:
        print(f"unregistration checkpoint found")
        file_dir = filename_dir_unreg
        status = "unreg_dvec_latent"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_dvec_checkpoint_dir_unsup(
    args,
    filename,
    filename_dir,
):
    """Create the available checkpoint for classifier during dynamic registrations."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename:
                status_storage.append("dvec")
                file_dir_storage.append(filename_dir)
            else:
                status_storage.append(None)
                file_dir_storage.append("")

    if filename_dir in file_dir_storage:
        print(f"Base dvec checkpoint found")
        file_dir = filename_dir
        status = "dvec"
    else:
        print(f"No dvec checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_cls_checkpoint_dir_reg_unsup(
    args,
    filename,
    filename_reg,
    filename_dir,
    filename_dir_reg,
):
    """Create the available checkpoint for classifier during dynamic registrations."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename_reg:
                status_storage.append("reg_cls")
                file_dir_storage.append(filename_dir_reg)
            elif file != filename_reg and file == filename:
                status_storage.append("cls")
                file_dir_storage.append(filename_dir)
            else:
                status_storage.append(None)
                file_dir_storage.append("")

    if filename_dir_reg in file_dir_storage:
        print(f"New registration checkpoint found")
        file_dir = filename_dir_reg
        status = "reg_cls"
    elif filename_dir in file_dir_storage:
        print(f"Base checkpoint found")
        file_dir = filename_dir
        status = "cls"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_dvec_latent_checkpoint_dir_unsup_unreg(
    args,
    filename,
    filename_unreg,
    filename_dir,
    filename_dir_unreg,
):
    """Create the available checkpoint for classifier during dynamic registrations."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename_unreg:
                status_storage.append("unreg_dvec_latent")
                file_dir_storage.append(filename_dir_unreg)
            elif file != filename_unreg and file == filename:
                status_storage.append("dvec_latent")
                file_dir_storage.append(filename_dir)
            else:
                status_storage.append(None)
                file_dir_storage.append("")

    if filename_dir_unreg in file_dir_storage:
        print(f"New unregistration checkpoint found")
        file_dir = filename_dir_unreg
        status = "unreg_dvec_latent"
    elif filename_dir in file_dir_storage:
        print(f"Base checkpoint found")
        file_dir = filename_dir
        status = "dvec_latent"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_cls_checkpoint_dir(args, filename, filename_dir):
    """Create the available checkpoint for classifier during training."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            if file == filename:
                file_dir_storage.append(filename_dir)
                status_storage.append("cls")
            else:
                file_dir_storage.append(None)
                status_storage.append("")

    if filename_dir in file_dir_storage:
        print(f"Base checkpoint found")
        file_dir = filename_dir
        status = "cls"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_cls_checkpoint_dir_dynamic_reg_scratch(args, filename, filename_dir):
    """Create the available checkpoint for classifier during training."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            if file == filename:
                file_dir_storage.append(filename_dir)
                status_storage.append("cls")
            else:
                file_dir_storage.append(None)
                status_storage.append("")

    if filename_dir in file_dir_storage:
        print(f"Dynamic reg scratch checkpoint found")
        file_dir = filename_dir
        status = "cls"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_dvec_latent_checkpoint_dir_dynamic_reg_scratch(args, filename, filename_dir):
    """Create the available checkpoint for classifier during training."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename:
                file_dir_storage.append(filename_dir)
                status_storage.append("cls")
            else:
                file_dir_storage.append(None)
                status_storage.append("")

    if filename_dir in file_dir_storage:
        print(f"Dynamic reg scratch checkpoint found")
        file_dir = filename_dir
        status = "cls"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_cls_scratch_checkpoint_dir(args, filename, filename_dir):
    """Create the available checkpoint for classifier during training."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            if file == filename:
                file_dir_storage.append(filename_dir)
                status_storage.append("cls")
            else:
                file_dir_storage.append(None)
                status_storage.append("")

    if filename_dir in file_dir_storage:
        print(f"Base checkpoint found")
        file_dir = filename_dir
        status = "cls"
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_dvec_latent_scratch_checkpoint_dir(args, filename, filename_dir):
    """Create the available checkpoint for dvec_latent during training."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename:
                file_dir_storage.append(filename_dir)
                status_storage.append("dvec_latent")
            else:
                file_dir_storage.append(None)
                status_storage.append("")

    if filename_dir in file_dir_storage:
        print(f"Base dvec latent checkpoint found")
        file_dir = filename_dir
        status = "dvec_latent"
    else:
        print(f"No dvec latent checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_dvec_latent_checkpoint_dir(args, filename, filename_dir):
    """Create the available checkpoint for dvec_latent during training."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename:
                file_dir_storage.append(filename_dir)
                status_storage.append("dvec_latent")
            else:
                file_dir_storage.append(None)
                status_storage.append("")

    if filename_dir in file_dir_storage:
        print(f"Base dvec latent checkpoint found")
        file_dir = filename_dir
        status = "dvec_latent"
    else:
        print(f"No dvec latent checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_filenames_dvec_unsupervised_latent(args, hparams, unreg_spks=[]):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_unsup_spkperbkt_{args.spk_per_bucket}_epdvec{args.epochs_per_dvector}_mode_{hparams.train_dvec_latent_mode}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_ablation = f"ckpt_unsup_ablation_spkperbkt_{args.spk_per_bucket}_epdvec{args.epochs_per_dvector}_mode_{hparams.train_dvec_latent_mode}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filename_reg = f"ckpt_unsup_cls_dp_opt_reg_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_unreg = f"ckpt_unsup_cls_unreg_{unreg_spks}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_re_reg = f"ckpt_unsup_cls_re_reg_{unreg_spks}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filename_dir = f"{checkpoint_dir_dvector_path}/ckpt_unsup_spkperbkt_{args.spk_per_bucket}_epdvec{args.epochs_per_dvector}_mode_{hparams.train_dvec_latent_mode}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_dir_ablation = f"{checkpoint_dir_dvector_path}/ckpt_unsup_ablation_spkperbkt_{args.spk_per_bucket}_epdvec{args.epochs_per_dvector}_mode_{hparams.train_dvec_latent_mode}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filename_dir_reg = f"{checkpoint_dir_dvector_path}/ckpt_unsup_cls_dp_opt_reg_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_dir_unreg = f"{checkpoint_dir_dvector_path}/ckpt_unsup_cls_unreg_{unreg_spks}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"
    filename_dir_re_reg = f"{checkpoint_dir_dvector_path}/ckpt_unsup_cls_re_reg_{unreg_spks}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename": filename,
        "filename_ablation": filename_ablation,
        "filename_reg": filename_reg,
        "filename_unreg": filename_unreg,
        "filename_re_reg": filename_re_reg,
        "filename_dir": filename_dir,
        "filename_dir_ablation": filename_dir_ablation,
        "filename_dir_reg": filename_dir_reg,
        "filename_dir_unreg": filename_dir_unreg,
        "filename_dir_re_reg": filename_dir_re_reg,
    }

    return filenames_and_dirs


def create_filenames_dvec_unsupervised_latent_vox(args, hparams, cont_loss_latent):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    method_name = cont_loss_latent.__name__

    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_unsup_{method_name}_vox_spkperbkt_{args.spk_per_bucket}_epdvec{args.epochs_per_dvector}_mode_{hparams.train_dvec_latent_mode}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filename_dir = f"{checkpoint_dir_dvector_path}/ckpt_unsup_{method_name}_vox_spkperbkt_{args.spk_per_bucket}_epdvec{args.epochs_per_dvector}_mode_{hparams.train_dvec_latent_mode}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename": filename,
        "filename_dir": filename_dir,
    }

    return filenames_and_dirs


def create_moving_average_collection(swa_scheduling, no_ma_scheduling):
    """Create collection of moving average functions."""

    moving_average_collection = {"swa": swa_scheduling, "no_ma": no_ma_scheduling}

    return moving_average_collection


def create_filenames_results(
    args,
    ma_mode,
    max_mem,
    spk_per_bkt,
    train_dvec_mode,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_train = args.result_dir_acc_train
    result_dir_acc_train_path = Path(result_dir_acc_train)
    result_dir_acc_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_train = args.result_dir_loss_train
    result_dir_loss_train_path = Path(result_dir_loss_train)
    result_dir_loss_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"

    filename_loss_val = f"loss_sup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"
    filename_acc_val = f"acc_sup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"

    filename_loss_train = f"loss_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"
    filename_acc_train = f"acc_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_train": result_dir_acc_train_path,
        "dir_acc_val": result_dir_acc_val_path,
        "dir_loss_train": result_dir_loss_train_path,
        "dir_loss_val": result_dir_loss_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_train": filename_acc_train,
        "filename_acc_val": filename_acc_val,
        "filename_loss_train": filename_loss_train,
        "filename_loss_val": filename_loss_val,
    }


def create_filenames_results_sup_scratch(args, agnt_num):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_train = args.result_dir_acc_train
    result_dir_acc_train_path = Path(result_dir_acc_train)
    result_dir_acc_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_train = args.result_dir_loss_train
    result_dir_loss_train_path = Path(result_dir_loss_train)
    result_dir_loss_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_libri_sup_train_scratch_agnt_{agnt_num}.json"

    filename_loss_val = f"loss_libri_sup_val_scratch_agnt_{agnt_num}.json"
    filename_acc_val = f"acc_libri_sup_val_scratch_agnt_{agnt_num}.json"

    filename_loss_train = f"loss_libri_sup_train_scratch_agnt_{agnt_num}.json"
    filename_acc_train = f"acc_libri_sup_train_scratch_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_train": result_dir_acc_train_path,
        "dir_acc_val": result_dir_acc_val_path,
        "dir_loss_train": result_dir_loss_train_path,
        "dir_loss_val": result_dir_loss_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_train": filename_acc_train,
        "filename_acc_val": filename_acc_val,
        "filename_loss_train": filename_loss_train,
        "filename_loss_val": filename_loss_val,
    }


def create_filenames_results_vox(
    args,
    ma_mode,
    max_mem,
    spk_per_bkt,
    train_dvec_mode,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_train = args.result_dir_acc_train
    result_dir_acc_train_path = Path(result_dir_acc_train)
    result_dir_acc_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_train = args.result_dir_loss_train
    result_dir_loss_train_path = Path(result_dir_loss_train)
    result_dir_loss_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_vox_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"

    filename_loss_val = f"loss_vox_sup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"
    filename_acc_val = f"acc_vox_sup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"

    filename_loss_train = f"loss_vox_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"
    filename_acc_train = f"acc_vox_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_train": result_dir_acc_train_path,
        "dir_acc_val": result_dir_acc_val_path,
        "dir_loss_train": result_dir_loss_train_path,
        "dir_loss_val": result_dir_loss_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_train": filename_acc_train,
        "filename_acc_val": filename_acc_val,
        "filename_loss_train": filename_loss_train,
        "filename_loss_val": filename_loss_val,
    }


def create_filenames_results_scratch_vox(
    args,
    ma_mode,
    spk_per_bkt,
    train_dvec_mode,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_train = args.result_dir_acc_train
    result_dir_acc_train_path = Path(result_dir_acc_train)
    result_dir_acc_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_train = args.result_dir_loss_train
    result_dir_loss_train_path = Path(result_dir_loss_train)
    result_dir_loss_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_vox_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"

    filename_loss_val = f"loss_vox_sup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"
    filename_acc_val = f"acc_vox_sup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"

    filename_loss_train = f"loss_vox_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"
    filename_acc_train = f"acc_vox_sup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_train": result_dir_acc_train_path,
        "dir_acc_val": result_dir_acc_val_path,
        "dir_loss_train": result_dir_loss_train_path,
        "dir_loss_val": result_dir_loss_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_train": filename_acc_train,
        "filename_acc_val": filename_acc_val,
        "filename_loss_train": filename_loss_train,
        "filename_loss_val": filename_loss_val,
    }


def create_filenames_dynamic_reg_results(
    args,
    hparams,
    new_reg_spk_indx,
    opt_bkt_indx,
    _spk_indx,
    buckets_old,
    buckets,
    round_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_dp_reg_round_{round_num}_pcnt_{hparams.pcnt_old}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_time_delay_storage = {f"{bkt_indx}": "" for bkt_indx in buckets}
    filename_acc_val_storage = {
        f"{spk_indx}_{bkt_indx}": ""
        for spk_indx, bkt_indx in zip(new_reg_spk_indx, opt_bkt_indx)
    }
    filename_acc_val_old_storage = {
        f"{spk_indx}_{bkt_indx}": ""
        for spk_indx, bkt_indx in zip(new_reg_spk_indx, opt_bkt_indx)
    }
    filename_acc_val_new_storage = {
        f"{spk_indx}_{bkt_indx}": ""
        for spk_indx, bkt_indx in zip(new_reg_spk_indx, opt_bkt_indx)
    }

    for _, bkt_indx_old in enumerate(buckets_old):
        filename_acc_val_old = f"acc_old_dp_val_reg_{_spk_indx}_in_{bkt_indx_old}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"
        filename_acc_val_old_storage[
            f"{_spk_indx}_{bkt_indx_old}"
        ] = filename_acc_val_old

    for spk_indx, bkt_indx in zip(new_reg_spk_indx, opt_bkt_indx):
        filename_acc_val = f"acc_dp_val_reg_{spk_indx}_in_{bkt_indx}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"
        filename_acc_val_new = f"acc_new_dp_val_reg_{spk_indx}_in_{bkt_indx}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"

        filename_acc_val_storage[f"{spk_indx}_{bkt_indx}"] = filename_acc_val
        filename_acc_val_new_storage[f"{spk_indx}_{bkt_indx}"] = filename_acc_val_new

    for bucket_id in buckets:
        filename_time_delay_bkt = f"td_dp_reg_in_{bucket_id}_round_{round_num}_pcnt_{hparams.pcnt_old}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"
        filename_time_delay_storage[f"{bucket_id}"] = filename_time_delay_bkt
    return {
        "dir_td": result_dir_td_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_bkt": filename_time_delay_storage,
        "filename_acc_val": filename_acc_val_storage,
        "filename_acc_val_old": filename_acc_val_old_storage,
        "filename_acc_val_new": filename_acc_val_new_storage,
    }


def create_filenames_dynamic_reg_unsup_results(
    args,
    hparams,
    new_reg_spk_indx,
    opt_bkt_indx,
    _spk_indx,
    buckets_old,
    buckets,
    round_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths for unsupervised registrations."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_cont_val = args.result_dir_acc_cont_val
    result_dir_acc_cont_val_path = Path(result_dir_acc_cont_val)
    result_dir_acc_cont_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_dp_reg_round_{round_num}_pcnt_{hparams.pcnt_old}_max_mem{args.max_mem}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.json"

    filename_time_delay_storage = {f"{bkt_indx}": "" for bkt_indx in buckets}
    filename_acc_val_storage = {
        f"{spk_indx}_{bkt_indx}": ""
        for spk_indx, bkt_indx in zip(new_reg_spk_indx, opt_bkt_indx)
    }
    filename_acc_val_old_storage = {
        f"{spk_indx}_{bkt_indx}": ""
        for spk_indx, bkt_indx in zip(new_reg_spk_indx, opt_bkt_indx)
    }
    filename_acc_val_new_storage = {
        f"{spk_indx}_{bkt_indx}": ""
        for spk_indx, bkt_indx in zip(new_reg_spk_indx, opt_bkt_indx)
    }

    for _, bkt_indx_old in enumerate(buckets_old):
        filename_acc_val_old = f"acc_unsup_old_dp_val_reg_{_spk_indx}_in_{bkt_indx_old}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"
        filename_acc_val_old_storage[
            f"{_spk_indx}_{bkt_indx_old}"
        ] = filename_acc_val_old

    for spk_indx, bkt_indx in zip(new_reg_spk_indx, opt_bkt_indx):
        filename_acc_val = f"acc_unsup_dp_val_reg_{spk_indx}_in_{bkt_indx}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"
        filename_acc_val_new = f"acc_unsup_new_dp_val_reg_{spk_indx}_in_{bkt_indx}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"

        filename_acc_val_storage[f"{spk_indx}_{bkt_indx}"] = filename_acc_val
        filename_acc_val_new_storage[f"{spk_indx}_{bkt_indx}"] = filename_acc_val_new

    for bucket_id in buckets:
        filename_time_delay_bkt = f"td_unsup_dp_reg_in_{bucket_id}_round_{round_num}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"
        filename_time_delay_storage[f"{bucket_id}"] = filename_time_delay_bkt
    return {
        "dir_td": result_dir_td_path,
        "dir_acc_val": result_dir_acc_cont_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_bkt": filename_time_delay_storage,
        "filename_acc_val": filename_acc_val_storage,
        "filename_acc_val_old": filename_acc_val_old_storage,
        "filename_acc_val_new": filename_acc_val_new_storage,
    }


def create_filename_dynamic_reg_td_results(args, hparams, round_num):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_dp_reg_round_{round_num}_pcnt_{hparams.pcnt_old}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"
    filename_time_delay_val = f"td_val_dp_reg_round_{round_num}_pcnt_{hparams.pcnt_old}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_val": filename_time_delay_val,
    }


def create_filenames_unreg_results(args, hparams, spk_per_bkt, unreg_spks):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_unreg_{unreg_spks}_spkperbkt_{spk_per_bkt}_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"
    filename_acc_val = f"acc_unreg_{unreg_spks}_spkperbkt_{spk_per_bkt}_val_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"
    filename_pred_indx_val = f"pred_indx_unreg_{unreg_spks}_spkperbkt_{spk_per_bkt}_val_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"
    filename_gtruth_indx_val = f"gtruth_indx_unreg_{unreg_spks}_spkperbkt_{spk_per_bkt}_val_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_val": filename_acc_val,
        "filename_pred_indx_val": filename_pred_indx_val,
        "filename_gtruth_indx_val": filename_gtruth_indx_val,
    }


def create_filenames_re_reg_results(args, hparams, spk_per_bkt, unreg_spks):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_re_reg_{unreg_spks}_spkperbkt_{spk_per_bkt}_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_acc_val = f"acc_re_reg_{unreg_spks}_spkperbkt_{spk_per_bkt}_val_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_unreg_unsup_results(args, hparams, spk_per_bkt, unreg_spks):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_unsup_unreg_{unreg_spks}_spkperbkt_{spk_per_bkt}_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_acc_val = f"acc_unsup_unreg_{unreg_spks}_spkperbkt_{spk_per_bkt}_val_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_re_reg_unsup_results(args, hparams, spk_per_bkt, unreg_spks):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_unsup_re_reg_{unreg_spks}_spkperbkt_{spk_per_bkt}_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_acc_val = f"acc_unsup_re_reg_{unreg_spks}_spkperbkt_{spk_per_bkt}_val_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_ma_{hparams.ma_mode}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_tsne_results(args, hparams):
    # Create paths for saving the activation during validation for TSNE
    result_dir_tsne = args.result_dir_tsne
    result_dir_tsne_path = Path(result_dir_tsne)
    result_dir_tsne_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the activations to be saved as .json files
    filename_tsne_pred_labels = (
        f"tsne_pred_labels_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.json"
    )
    filename_tsne_pred_feats = (
        f"tsne_pred_feats_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.json"
    )

    return {
        "dir_tsne": result_dir_tsne_path,
        "tsne_pred_labels": filename_tsne_pred_labels,
        "tsne_pred_feats": filename_tsne_pred_feats,
    }


def create_filenames_unreg_tsne_results(args, hparams, bucket):
    # Create paths for saving the activation during validation for TSNE
    result_dir_tsne = args.result_dir_tsne
    result_dir_tsne_path = Path(result_dir_tsne)
    result_dir_tsne_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the activations to be saved as .json files
    filename_tsne_pred_labels = f"tsne_pred_reduced_dim_labels_unreg_bkt_{bucket}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.json"
    filename_tsne_pred_feats = f"tsne_pred_reduced_dim_feats_unreg_bkt_{bucket}_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.json"

    return {
        "dir_tsne": result_dir_tsne_path,
        "tsne_pred_labels": filename_tsne_pred_labels,
        "tsne_pred_feats": filename_tsne_pred_feats,
    }


def create_filenames_tsne_unsup_results(args, hparams):
    # Create paths for saving the activation during validation for TSNE
    result_dir_tsne = args.result_dir_tsne
    result_dir_tsne_path = Path(result_dir_tsne)
    result_dir_tsne_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the activations to be saved as .json files
    filename_tsne_pred_labels = (
        f"tsne_unsup_pred_labels_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.json"
    )
    filename_tsne_pred_feats = (
        f"tsne_unsup_pred_feats_ma_{hparams.ma_mode}_agnt_{args.agnt_num}.json"
    )

    return {
        "dir_tsne": result_dir_tsne_path,
        "tsne_pred_labels": filename_tsne_pred_labels,
        "tsne_pred_feats": filename_tsne_pred_feats,
    }


def create_filenames_unsupervised_results(
    args,
    ma_mode,
    max_mem_unsup,
    spk_per_bkt,
    train_dvec_mode,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_cont_val = args.result_dir_loss_val
    result_dir_loss_cont_val_path = Path(result_dir_loss_cont_val)
    result_dir_loss_cont_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_cont_val = args.result_dir_acc_val
    result_dir_acc_cont_val_path = Path(result_dir_acc_cont_val)
    result_dir_acc_cont_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_unsup_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"
    filename_time_delay_val = f"td_unsup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"

    filename_loss_cont_val = f"loss_unsup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"
    filename_acc_cont_val = f"acc_unsup_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_loss_cont_val": result_dir_loss_cont_val_path,
        "dir_acc_cont_val": result_dir_acc_cont_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_val": filename_time_delay_val,
        "filename_loss_cont_val": filename_loss_cont_val,
        "filename_acc_cont_val": filename_acc_cont_val,
    }


def create_filenames_unsupervised_results_v2(args, agnt_num):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_cont_val = args.result_dir_loss_val
    result_dir_loss_cont_val_path = Path(result_dir_loss_cont_val)
    result_dir_loss_cont_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_cont_val = args.result_dir_acc_val
    result_dir_acc_cont_val_path = Path(result_dir_acc_cont_val)
    result_dir_acc_cont_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_libri_unsup_train_scratch_agnt_{agnt_num}_v2.json"

    filename_loss_cont_val = f"loss_libri_unsup_val_scratch_agnt_{agnt_num}_v2.json"
    filename_acc_cont_val = f"acc_libri_unsup_val_scratch_agnt_{agnt_num}_v2.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_loss_cont_val": result_dir_loss_cont_val_path,
        "dir_acc_cont_val": result_dir_acc_cont_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_loss_cont_val": filename_loss_cont_val,
        "filename_acc_cont_val": filename_acc_cont_val,
    }


def create_filenames_unsupervised_results_vox(
    args,
    ma_mode,
    max_mem_unsup,
    spk_per_bkt,
    train_dvec_mode,
    agnt_num,
    cont_loss_feat,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    method_name = cont_loss_feat.__name__

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_cont_val = args.result_dir_loss_val
    result_dir_loss_cont_val_path = Path(result_dir_loss_cont_val)
    result_dir_loss_cont_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_cont_val = args.result_dir_acc_val
    result_dir_acc_cont_val_path = Path(result_dir_acc_cont_val)
    result_dir_acc_cont_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_cont_train = args.result_dir_acc_train
    result_dir_acc_cont_train_path = Path(result_dir_acc_cont_train)
    result_dir_acc_cont_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_cont_train = args.result_dir_loss_train
    result_dir_loss_cont_train_path = Path(result_dir_loss_cont_train)
    result_dir_loss_cont_train_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_unsup_{method_name}_vox_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"
    filename_time_delay_val = f"td_unsup_{method_name}_vox_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"

    filename_loss_cont_val = f"loss_unsup_{method_name}_vox_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"
    filename_acc_cont_val = f"acc_unsup_{method_name}_vox_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"

    filename_loss_cont_train = f"loss_unsup_{method_name}_vox_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"
    filename_acc_cont_train = f"acc_unsup_{method_name}_vox_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_max_mem_{max_mem_unsup}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_loss_cont_val": result_dir_loss_cont_val_path,
        "dir_acc_cont_val": result_dir_acc_cont_val_path,
        "dir_loss_cont_train": result_dir_loss_cont_train_path,
        "dir_acc_cont_train": result_dir_acc_cont_train_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_val": filename_time_delay_val,
        "filename_loss_cont_val": filename_loss_cont_val,
        "filename_acc_cont_val": filename_acc_cont_val,
        "filename_loss_cont_train": filename_loss_cont_train,
        "filename_acc_cont_train": filename_acc_cont_train,
    }


def create_filenames_unsupervised_results_scratch_vox(
    args,
    ma_mode,
    spk_per_bkt,
    train_dvec_mode,
    agnt_num,
    cont_loss_feat,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    method_name = cont_loss_feat.__name__

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_cont_val = args.result_dir_loss_val
    result_dir_loss_cont_val_path = Path(result_dir_loss_cont_val)
    result_dir_loss_cont_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_cont_val = args.result_dir_acc_val
    result_dir_acc_cont_val_path = Path(result_dir_acc_cont_val)
    result_dir_acc_cont_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_cont_train = args.result_dir_acc_train
    result_dir_acc_cont_train_path = Path(result_dir_acc_cont_train)
    result_dir_acc_cont_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_cont_train = args.result_dir_loss_train
    result_dir_loss_cont_train_path = Path(result_dir_loss_cont_train)
    result_dir_loss_cont_train_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_unsup_{method_name}_vox_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"
    filename_time_delay_val = f"td_unsup_{method_name}_vox_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"

    filename_loss_cont_val = f"loss_unsup_{method_name}_vox_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"
    filename_acc_cont_val = f"acc_unsup_{method_name}_vox_val_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"

    filename_loss_cont_train = f"loss_unsup_{method_name}_vox_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"
    filename_acc_cont_train = f"acc_unsup_{method_name}_vox_train_spkperbkt_{spk_per_bkt}_mode_{train_dvec_mode}_ma_{ma_mode}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_loss_cont_val": result_dir_loss_cont_val_path,
        "dir_acc_cont_val": result_dir_acc_cont_val_path,
        "dir_loss_cont_train": result_dir_loss_cont_train_path,
        "dir_acc_cont_train": result_dir_acc_cont_train_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_val": filename_time_delay_val,
        "filename_loss_cont_val": filename_loss_cont_val,
        "filename_acc_cont_val": filename_acc_cont_val,
        "filename_loss_cont_train": filename_loss_cont_train,
        "filename_acc_cont_train": filename_acc_cont_train,
    }


def create_filenames_reg_supervised_results(
    args,
    ma_mode,
    max_mem,
    epochs_per_dvector,
    epochs_per_cls,
    round_num,
    pcnt_old,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_reg_pcntold_{pcnt_old}_round_{round_num}_sup_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem}_agnt_{agnt_num}.json"
    filename_time_delay_val = f"td_val_reg_pcntold_{pcnt_old}_round_{round_num}_sup_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem}_agnt_{agnt_num}.json"

    filename_loss_val = f"loss_val_reg_pcntold_{pcnt_old}_round_{round_num}_sup_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem}_agnt_{agnt_num}.json"
    filename_acc_val = f"acc_val_reg_pcntold_{pcnt_old}_round_{round_num}_sup_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_loss_val": result_dir_loss_val_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_val": filename_time_delay_val,
        "filename_loss_val": filename_loss_val,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_reg_supervised_scratch_results(
    args,
    ma_mode,
    max_mem,
    epochs_per_dvector,
    epochs_per_cls,
    round_num,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_reg_scratch_round_{round_num}_sup_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem}_agnt_{agnt_num}.json"
    filename_time_delay_val = f"td_val_reg_scratch_round_{round_num}_sup_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem}_agnt_{agnt_num}.json"

    filename_loss_val = f"loss_val_reg_scratch_round_{round_num}_sup_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem}_agnt_{agnt_num}.json"
    filename_acc_val = f"acc_val_reg_scratch_round_{round_num}_sup_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_loss_val": result_dir_loss_val_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_val": filename_time_delay_val,
        "filename_loss_val": filename_loss_val,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_reg_unsupervised_scratch_results(
    args,
    ma_mode,
    max_mem_unsup,
    epochs_per_dvector,
    epochs_per_cls,
    round_num,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_reg_unsup_scratch_round_{round_num}_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem_unsup}_agnt_{agnt_num}.json"
    filename_time_delay_val = f"td_val_reg_unsup_scratch_round_{round_num}_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem_unsup}_agnt_{agnt_num}.json"

    filename_loss_val = f"loss_val_reg_unsup_scratch_round_{round_num}_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem_unsup}_agnt_{agnt_num}.json"
    filename_acc_val = f"acc_val_reg_unsup_scratch_round_{round_num}_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem_unsup}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_loss_val": result_dir_loss_val_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_time_delay_val": filename_time_delay_val,
        "filename_loss_val": filename_loss_val,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_reg_unsupervised_results(
    args,
    ma_mode,
    max_mem_unsup,
    epochs_per_dvector,
    epochs_per_cls,
    round_num,
    pcnt_old,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_unsup_reg_pcntold_{pcnt_old}_round_{round_num}_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem_unsup}_agnt_{agnt_num}.json"

    filename_loss_cont_val = f"loss_unsup_val_reg_pcntold_{pcnt_old}_round_{round_num}_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem_unsup}_agnt_{agnt_num}.json"
    filename_acc_cont_val = f"acc_unsup_val_reg_pcntold_{pcnt_old}_round_{round_num}_epdvec{epochs_per_dvector}_epcls{epochs_per_cls}_ma_{ma_mode}_max_mem{max_mem_unsup}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_loss_cont_val": result_dir_loss_val_path,
        "dir_acc_cont_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_loss_cont_val": filename_loss_cont_val,
        "filename_acc_cont_val": filename_acc_cont_val,
    }


def create_filenames_modular_results(args):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_modular_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_train = args.result_dir_acc_train_modular
    result_dir_acc_train_path = Path(result_dir_acc_train)
    result_dir_acc_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val_modular
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_train = args.result_dir_loss_train_modular
    result_dir_loss_train_path = Path(result_dir_loss_train)
    result_dir_loss_train_path.mkdir(parents=True, exist_ok=True)

    result_dir_loss_val = args.result_dir_loss_val_modular
    result_dir_loss_val_path = Path(result_dir_loss_val)
    result_dir_loss_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_modular_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_acc_train = f"acc_modular_train_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"
    filename_acc_val = f"acc_modular_val_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_loss_train = f"loss_modular_train_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"
    filename_loss_val = f"loss_modular_val_epdvec{args.epochs_per_dvector}_epcls{args.epochs_per_cls}_max_mem{args.max_mem}_agnt_{args.agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_train": result_dir_acc_train_path,
        "dir_acc_val": result_dir_acc_val_path,
        "dir_loss_train": result_dir_loss_train_path,
        "dir_loss_val": result_dir_loss_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_train": filename_acc_train,
        "filename_acc_val": filename_acc_val,
        "filename_loss_train": filename_loss_train,
        "filename_loss_val": filename_loss_val,
    }
