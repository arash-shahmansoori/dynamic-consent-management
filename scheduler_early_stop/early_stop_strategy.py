import torch
import numpy as np

from functools import partial


def early_stop_strategy_bkt(
    early_stopping,
    val_out_round,
    epoch,
    bucket_id,
    opt_unique_bkt,
    indx_opt_unique_bkt,
    opt_unique_bkt_sofar,
    indx_opt_unique_bkt_sofar,
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
        # spk_selected_strategy = indx_opt_unique_bkt[opt_unique_bkt.index(bucket_id)]

        indx_spk_selected_strategy = opt_unique_bkt.index(bucket_id)
        spk_selected_strategy = indx_opt_unique_bkt[indx_spk_selected_strategy]

        val_acc_status = "val_acc"
    elif strategy_condition["strategy_2"]:
        # spk_selected_strategy = indx_opt_unique_bkt_sofar[
        #     opt_unique_bkt_sofar.index(bucket_id)
        # ]
        indx_spk_selected_strategy = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]
        spk_selected_strategy = np.array(indx_opt_unique_bkt_sofar)[
            indx_spk_selected_strategy
        ].tolist()

        val_acc_status = "val_acc"
    elif strategy_condition["strategy_3"]:
        # spk_selected_strategy = indx_opt_unique_bkt[opt_unique_bkt.index(bucket_id)]

        indx_spk_selected_strategy = opt_unique_bkt.index(bucket_id)
        indx_spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        spk_selected_strategy_sofar = np.array(indx_opt_unique_bkt_sofar)[
            indx_spk_selected_strategy_sofar
        ].tolist()
        spk_selected_strategy_current = [
            indx_opt_unique_bkt[indx_spk_selected_strategy]
        ]

        spk_selected_strategy = (
            spk_selected_strategy_sofar + spk_selected_strategy_current
        )

        val_acc_status = "val_acc"
    elif strategy_condition["strategy_4"]:
        spk_selected_strategy = -1  # Represent the already registered old speakers

        val_acc_status = "val_acc_old"
    else:
        raise ValueError

    new_early_stop = partial(
        early_stopping[bucket_id],
        torch.tensor(
            val_out_round[val_acc_status][f"{spk_selected_strategy}_{bucket_id}"]
        ).view(-1)[-1],
    )

    new_early_stop(epoch, bucket_id)
