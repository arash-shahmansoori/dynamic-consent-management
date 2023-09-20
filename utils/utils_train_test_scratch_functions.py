import torch
from .utils_metric import num_correct


def train_cls_scratch(model, inputs, opt, criterion):

    # Set up model for training
    model.train()

    opt.zero_grad()

    _, out, _ = model(inputs["x"])

    loss = criterion(out, inputs["y"])

    # Backward
    loss.backward()

    opt.step()

    return loss


def test_cls_scratch(model, inputs, criterion):

    total_num_correct = 0
    with torch.no_grad():
        prob, out, _ = model(inputs["x_val"])
        loss = criterion(out, inputs["y_val"])
        corr_spk, _, _ = num_correct(prob, inputs["y_val"].view(-1), topk=1)

        total_num_correct += corr_spk

        acc = (total_num_correct / len(inputs["y_val"])) * 100

    return acc, loss
