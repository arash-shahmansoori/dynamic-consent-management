import json
import numpy as np

from pathlib import Path

from compute_optimal_buckets import (
    create_unique_opt_bkt_spks_existing,
    create_unique_opt_bkt_spks_sofar,
)
from utils import (
    create_filenames_dynamic_reg_results,
    create_strategy_filenames,
    strategy_per_bkt_indx,
)


def load_new_reg_sup(args, hparams, round_num):
    """To load the metrics from the JSON files in the results directory (dynamic registrations)."""

    # For the existing set of optimal buckets
    opt_unique_bkt, indx_opt_unique_bkt = create_unique_opt_bkt_spks_existing(
        args,
        round_num,
    )

    opt_unique_bkt_sofar, indx_opt_unique_bkt_sofar = create_unique_opt_bkt_spks_sofar(
        args,
        round_num,
    )

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # Paths and file names for the metrics per round
    _spk_indx = -1  # To represent the old speakers in the buckets
    buckets_old = [
        b_old
        for b_old in buckets
        if (b_old not in opt_unique_bkt) and (b_old not in opt_unique_bkt_sofar)
    ]  # To represent the old buckets, i.e., excluding the new reg ans reg_sofar

    paths_filenames = create_filenames_dynamic_reg_results(
        args,
        hparams,
        indx_opt_unique_bkt,
        opt_unique_bkt,
        _spk_indx,
        buckets_old,
        buckets,
        round_num,
    )

    dir_acc_val_causal, paths_filenames_causal = create_strategy_filenames(
        args,
        hparams,
        buckets,
        opt_unique_bkt_sofar,
        indx_opt_unique_bkt_sofar,
        opt_unique_bkt,
        indx_opt_unique_bkt,
    )

    with open(
        Path(
            paths_filenames["dir_td"],
            paths_filenames["filename_time_delay"],
        ),
        "r",
    ) as elapsed_time:
        elapsed_time_round = json.load(elapsed_time)

    val_acc_round = {}
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
        indx_strategy = selcted_strategy_indx["indx_strategy"]

        if indx_selected_new_spks_overall == -1:

            with open(
                Path(
                    paths_filenames["dir_acc_val"],
                    paths_filenames["filename_acc_val_old"][
                        f"{_spk_indx}_{indx_selected}"
                    ],
                ),
                "r",
            ) as val_acc:
                val_acc_round[(f"{indx_strategy}", indx_selected)] = json.load(val_acc)

        else:

            with open(
                Path(
                    dir_acc_val_causal,
                    paths_filenames_causal[
                        f"{indx_selected_new_spks_overall}_{indx_selected}"
                    ][0],
                ),
                "r",
            ) as val_acc:
                val_acc_round[(f"{indx_strategy}", indx_selected)] = json.load(val_acc)

    return val_acc_round, elapsed_time_round
