from functools import partial
from .train_reg_unsupervised_old_scratch import train_reg_unsup_old_scratch


def train_strategy_old_reg_per_epoch_per_bkt_unsup_scratch(
    epoch,
    early_stopping_status,
):

    train_reg_unsup_old_partial_scratch = partial(
        train_reg_unsup_old_scratch,
        epoch,
        early_stopping_status,
    )

    return train_reg_unsup_old_partial_scratch