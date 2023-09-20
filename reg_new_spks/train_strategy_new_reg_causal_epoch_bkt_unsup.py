from functools import partial
from .train_selective_new_reg_inductive_bias_unsupervised_causal import (
    train_selective_new_reg_inductive_bias_unsup_causal,
)


def train_strategy_new_reg_causal_per_epoch_per_bkt_unsup(
    epoch,
    indx_opt_unique_bkt_sofar,
    spk_selected_strategy_sofar,
    indx_opt_unique_bkt,
    indx_selected_id,
    early_stopping,
):

    train_selective_reg_inductive_bias_unsup_causal_new = partial(
        train_selective_new_reg_inductive_bias_unsup_causal,
        epoch,
        indx_opt_unique_bkt_sofar,
        spk_selected_strategy_sofar,
        indx_opt_unique_bkt,
        indx_selected_id,
        early_stopping,
    )

    return train_selective_reg_inductive_bias_unsup_causal_new
