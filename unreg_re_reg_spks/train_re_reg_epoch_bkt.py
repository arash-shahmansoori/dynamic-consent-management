from .strategy_old_re_reg_epoch_bkt import strategy_old_re_reg_per_epoch_per_bkt
from .train_strategy_re_reg_epoch_bkt import train_strategy_re_reg_per_epoch_per_bkt


def train_re_reg_per_epoch_per_bkt(
    args,
    device,
    outputs,
    outputs_updated,
    bucket_id,
    epoch,
    early_stopping_status,
    **kwargs_training,
):

    strategy_keys = {
        "strategy_1": len(outputs_updated[bucket_id]) == len(outputs[bucket_id]),
        "strategy_2": len(outputs_updated[bucket_id]) != len(outputs[bucket_id]),
        "strategy_3": len(outputs_updated[bucket_id]) == 0
        and len(outputs[bucket_id]) != 0,
    }

    if strategy_keys["strategy_1"]:
        train_strategy = strategy_old_re_reg_per_epoch_per_bkt(early_stopping_status)
    elif strategy_keys["strategy_2"]:
        train_strategy = train_strategy_re_reg_per_epoch_per_bkt(
            epoch,
            kwargs_training["early_stop"][bucket_id],
        )
    elif strategy_keys["strategy_3"]:
        train_strategy = train_strategy_re_reg_per_epoch_per_bkt(
            epoch,
            kwargs_training["early_stop"][bucket_id],
        )
    else:
        raise ValueError

    sup_re_reg = train_strategy(
        args,
        outputs,
        bucket_id,
        device,
        kwargs_training,
    )

    x = sup_re_reg["x"]
    spk = sup_re_reg["y"]

    xe = kwargs_training["dvectors"][bucket_id](x).detach()

    feat_props = {
        "feat_bkt": xe.view(-1, args.dim_emb),
        "label_bkt": spk.view(-1),
    }

    return feat_props