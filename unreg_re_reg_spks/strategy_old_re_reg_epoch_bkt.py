from functools import partial
from .re_reg_supervised_old import re_reg_sup_old


def strategy_old_re_reg_per_epoch_per_bkt(early_stopping_status):

    re_reg_sup_old_partial = partial(re_reg_sup_old, early_stopping_status)

    return re_reg_sup_old_partial