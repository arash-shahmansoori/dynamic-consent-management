from functools import partial
from .train_selective_reg_inductive_bias_unsupervised_scratch import (
    train_selective_reg_inductive_bias_unsup_scratch,
)


def train_strategy_new_reg_per_epoch_per_bkt_unsup_scratch(
    epoch,
    indx_opt_unique_bkt,
    indx_selected_id,
    early_stopping,
):

    train_selective_reg_inductive_bias_unsup_new_scratch = partial(
        train_selective_reg_inductive_bias_unsup_scratch,
        epoch,
        indx_opt_unique_bkt,
        indx_selected_id,
        early_stopping,
    )

    return train_selective_reg_inductive_bias_unsup_new_scratch