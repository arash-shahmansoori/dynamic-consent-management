from functools import partial
from .train_selective_reg_inductive_bias_unsupervised import (
    train_selective_reg_inductive_bias_unsup,
)


def train_strategy_new_reg_per_epoch_per_bkt_unsup(
    epoch,
    indx_opt_unique_bkt,
    indx_selected_id,
    early_stopping,
):

    train_selective_reg_inductive_bias_unsup_new = partial(
        train_selective_reg_inductive_bias_unsup,
        epoch,
        indx_opt_unique_bkt,
        indx_selected_id,
        early_stopping,
    )

    return train_selective_reg_inductive_bias_unsup_new