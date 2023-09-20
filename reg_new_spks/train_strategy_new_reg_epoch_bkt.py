from functools import partial
from .train_selective_reg_inductive_bias_supervised import (
    train_selective_reg_inductive_bias_sup,
)


def train_strategy_new_reg_per_epoch_per_bkt(
    epoch,
    indx_opt_unique_bkt,
    indx_selected_id,
    early_stopping,
):

    train_selective_reg_inductive_bias_sup_new = partial(
        train_selective_reg_inductive_bias_sup,
        epoch,
        indx_opt_unique_bkt,
        indx_selected_id,
        early_stopping,
    )

    return train_selective_reg_inductive_bias_sup_new
