from functools import partial
from .train_selective_sofar_reg_inductive_bias_supervised_causal import (
    train_selective_sofar_reg_inductive_bias_sup_causal,
)


def train_strategy_sofar_reg_causal_per_epoch_per_bkt(
    epoch,
    indx_opt_unique_bkt_sofar,
    spk_selected_strategy_sofar,
    early_stopping,
):

    train_selective_reg_inductive_bias_sup_causal_sofar = partial(
        train_selective_sofar_reg_inductive_bias_sup_causal,
        epoch,
        indx_opt_unique_bkt_sofar,
        spk_selected_strategy_sofar,
        early_stopping,
    )

    return train_selective_reg_inductive_bias_sup_causal_sofar