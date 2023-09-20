import numpy as np

from functools import partial

from .eval_new_reg_epoch_bkt import eval_new_reg_per_round_per_epoch_per_bkt
from .eval_new_reg_causal_epoch_bkt import (
    eval_new_reg_causal_per_round_per_epoch_per_bkt,
)
from .eval_sofar_reg_causal_epoch_bkt import (
    eval_sofar_reg_causal_per_round_per_epoch_per_bkt,
)
from .eval_old_epoch_bkt import eval_old_per_round_per_epoch_per_bkt


def eval_reg_overall_per_round_per_epoch_per_bkt(
    args,
    device,
    outputs,
    bucket_id,
    opt_unique_bkt_sofar,
    indx_opt_unique_bkt_sofar,
    opt_unique_bkt,
    indx_opt_unique_bkt,
    **kwargs_validation,
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

        eval_reg_per_round_per_epoch_per_bkt = partial(
            eval_new_reg_per_round_per_epoch_per_bkt,
            indx_opt_unique_bkt,
            spk_selected_strategy,
        )

    elif strategy_condition["strategy_2"]:
        spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        eval_reg_per_round_per_epoch_per_bkt = partial(
            eval_sofar_reg_causal_per_round_per_epoch_per_bkt,
            indx_opt_unique_bkt_sofar,
            spk_selected_strategy_sofar,
        )

    elif strategy_condition["strategy_3"]:
        spk_selected_strategy = opt_unique_bkt.index(bucket_id)
        spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        eval_reg_per_round_per_epoch_per_bkt = partial(
            eval_new_reg_causal_per_round_per_epoch_per_bkt,
            indx_opt_unique_bkt_sofar,
            spk_selected_strategy_sofar,
            indx_opt_unique_bkt,
            spk_selected_strategy,
        )

    elif strategy_condition["strategy_4"]:
        spk_selected_strategy = -1  # Represent the already registered old speakers

        eval_reg_per_round_per_epoch_per_bkt = eval_old_per_round_per_epoch_per_bkt

    else:
        raise ValueError

    eval_out = eval_reg_per_round_per_epoch_per_bkt(
        args,
        outputs,
        bucket_id,
        device,
        **kwargs_validation,
    )

    return eval_out