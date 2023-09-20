import numpy as np

from pathlib import Path


def strategy_per_bkt_indx(
    bucket_id,
    opt_unique_bkt_sofar,
    indx_opt_unique_bkt_sofar,
    opt_unique_bkt,
    indx_opt_unique_bkt,
):

    strategy_condition = {
        "strategy_1": (bucket_id in opt_unique_bkt)
        and (bucket_id not in opt_unique_bkt_sofar),
        "strategy_2": (bucket_id not in opt_unique_bkt)
        and (bucket_id in opt_unique_bkt_sofar),
        "strategy_3": (bucket_id in opt_unique_bkt)
        and (bucket_id in opt_unique_bkt_sofar),
        "strategy_4": (bucket_id not in opt_unique_bkt)
        and (bucket_id not in opt_unique_bkt_sofar),
    }

    if strategy_condition["strategy_1"]:
        spk_selected_strategy = opt_unique_bkt.index(bucket_id)

        return {
            "indx_selected_new_spks_overall": indx_opt_unique_bkt[
                spk_selected_strategy
            ],
            "indx_selected": bucket_id,
            "indx_strategy": 1,
        }

    elif strategy_condition["strategy_2"]:
        spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        sub_lbs_other_sofar = np.array(indx_opt_unique_bkt_sofar)[
            spk_selected_strategy_sofar
        ].tolist()

        return {
            "indx_selected_new_spks_overall": sub_lbs_other_sofar,
            "indx_selected": bucket_id,
            "indx_strategy": 2,
        }

    elif strategy_condition["strategy_3"]:
        spk_selected_strategy = opt_unique_bkt.index(bucket_id)
        spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        sub_lbs_other_sofar = np.array(indx_opt_unique_bkt_sofar)[
            spk_selected_strategy_sofar
        ].tolist()
        sub_lbs_other = [indx_opt_unique_bkt[spk_selected_strategy]]

        return {
            "indx_selected_new_spks_overall": sub_lbs_other_sofar + sub_lbs_other,
            "indx_selected": bucket_id,
            "indx_strategy": 3,
        }

    elif strategy_condition["strategy_4"]:
        spk_selected_strategy = -1  # Represent the already registered old speakers

        return {
            "indx_selected_new_spks_overall": -1,
            "indx_selected": bucket_id,
            "indx_strategy": 4,
        }

    else:
        raise ValueError


def create_filenames_dynamic_reg_causal_results(
    args,
    hparams,
    indx_selected_new_spks_overall,
    indx_selected,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the validation metrics
    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_acc_val_storage = {f"{indx_selected_new_spks_overall}_{indx_selected}": []}
    filename_acc_val = f"acc_dp_val_causal_reg_{indx_selected_new_spks_overall}_{indx_selected}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_acc_val_storage[
        f"{indx_selected_new_spks_overall}_{indx_selected}"
    ].append(filename_acc_val)

    return {
        "dir_acc_val": result_dir_acc_val_path,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_dynamic_reg_sup_causal_results(
    args,
    hparams,
    indx_selected_new_spks_overall,
    indx_selected,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the validation metrics
    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_acc_val_storage = {f"{indx_selected_new_spks_overall}_{indx_selected}": []}
    filename_acc_val = f"acc_sup_dp_val_causal_reg_{indx_selected_new_spks_overall}_{indx_selected}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_acc_val_storage[
        f"{indx_selected_new_spks_overall}_{indx_selected}"
    ].append(filename_acc_val)

    return {
        "dir_acc_val": result_dir_acc_val_path,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_dynamic_reg_unsup_causal_results(
    args,
    hparams,
    indx_selected_new_spks_overall,
    indx_selected,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the validation metrics
    result_dir_acc_cont_val = args.result_dir_acc_cont_val
    result_dir_acc_cont_val_path = Path(result_dir_acc_cont_val)
    result_dir_acc_cont_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_acc_val_storage = {f"{indx_selected_new_spks_overall}_{indx_selected}": []}
    filename_acc_val = f"acc_unsup_dp_val_causal_reg_{indx_selected_new_spks_overall}_{indx_selected}_pcnt_{hparams.pcnt_old}_ma_{hparams.ma_mode}_max_mem_{args.max_mem}_agnt_{args.agnt_num}.json"

    filename_acc_val_storage[
        f"{indx_selected_new_spks_overall}_{indx_selected}"
    ].append(filename_acc_val)

    return {
        "dir_acc_val": result_dir_acc_cont_val_path,
        "filename_acc_val": filename_acc_val,
    }


def create_strategy_filenames(
    args,
    hparams,
    buckets,
    opt_unique_bkt_sofar,
    indx_opt_unique_bkt_sofar,
    opt_unique_bkt,
    indx_opt_unique_bkt,
    create_filenames_results_causal,
):
    filename_acc_val_storage = {}
    indx_selected_new_spks_overall_storage, indx_selected_storage = [], []
    for _, bucket_id in enumerate(buckets):

        selcted_strategy_indx = strategy_per_bkt_indx(
            bucket_id,
            opt_unique_bkt_sofar,
            indx_opt_unique_bkt_sofar,
            opt_unique_bkt,
            indx_opt_unique_bkt,
        )

        indx_selected_new_spks_overall = selcted_strategy_indx[
            "indx_selected_new_spks_overall"
        ]
        indx_selected = selcted_strategy_indx["indx_selected"]

        indx_selected_new_spks_overall_storage.append(indx_selected_new_spks_overall)
        indx_selected_storage.append(indx_selected)

        # filename_acc_val_storage = {
        #     f"{indx_selected_new_spks_overall}_{indx_selected}": []
        #     for (indx_selected_new_spks_overall, indx_selected) in zip(
        #         indx_selected_new_spks_overall_storage, indx_selected_storage
        #     )
        # }

        filename_acc_val_storage[
            f"{indx_selected_new_spks_overall}_{indx_selected}"
        ] = []

        strategy_filnames = create_filenames_results_causal(
            args,
            hparams,
            indx_selected_new_spks_overall,
            indx_selected,
        )

        filename_acc_val_storage[
            f"{indx_selected_new_spks_overall}_{indx_selected}"
        ].append(strategy_filnames["filename_acc_val"])

    return strategy_filnames["dir_acc_val"], filename_acc_val_storage
