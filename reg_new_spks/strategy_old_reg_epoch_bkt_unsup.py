from functools import partial
from .reg_unsupervised_old import reg_unsup_old


def strategy_old_reg_per_epoch_per_bkt_unsup(early_stopping_status):

    reg_unsup_old_partial = partial(reg_unsup_old, early_stopping_status)

    return reg_unsup_old_partial