import json
import torch


def save_model_ckpt_dvec(epoch, model, optimizer, criterion, loss, bucket_id, filename):
    """Save the d-vector checkpoints per bucket."""

    criterion_state = criterion.state_dict() if criterion is not None else None
    opt_stat = optimizer.state_dict() if optimizer is not None else None

    state = {
        "epoch": epoch,
        "bucket_id": bucket_id,
        "loss": loss,
        "criterion": criterion_state,
        "model": model.state_dict(),
        "optimizer": opt_stat,
    }

    torch.save(state, filename)


def save_model_ckpt_cls(
    epoch,
    round_num,
    model,
    model_ma,
    optimizer,
    criterion,
    loss,
    ma_n,
    filename,
):
    """Save the classifier checkpoints."""

    criterion_state = criterion.state_dict() if criterion is not None else None
    opt_stat = optimizer.state_dict() if optimizer is not None else None

    state = {
        f"start_epoch_round_{round_num}": epoch,
        "epoch": epoch,
        "loss": loss,
        "criterion": criterion_state,
        "model": model.state_dict(),
        "optimizer": opt_stat,
        "model_ma": model_ma.state_dict(),
        "ma_n": ma_n,
    }

    torch.save(state, filename)


def save_model_ckpt_scratch_cls(
    epoch,
    classifier,
    optimizer,
    ce_loss,
    loss,
    filename,
):
    """Save the classifier checkpoints."""

    criterion_state = ce_loss.state_dict() if ce_loss is not None else None
    opt_stat = optimizer.state_dict() if optimizer is not None else None

    state = {
        "epoch": epoch,
        "loss": loss,
        "criterion": criterion_state,
        "model": classifier.state_dict(),
        "optimizer": opt_stat,
    }

    torch.save(state, filename)


def save_as_json(result_dir_path, file_name, metric):
    """Save the metric as a JSON file"""

    with open(result_dir_path / file_name, "w") as filename:
        json.dump(metric, filename, indent=2)
