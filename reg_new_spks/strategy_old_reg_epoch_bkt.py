from functools import partial
from .reg_supervised_old import reg_sup_old


def strategy_old_reg_per_epoch_per_bkt(early_stopping_status):

    reg_sup_old_partial = partial(reg_sup_old, early_stopping_status)

    return reg_sup_old_partial