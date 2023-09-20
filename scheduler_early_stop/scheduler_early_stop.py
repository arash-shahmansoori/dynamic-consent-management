import torch


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, params):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = params.patience
        self.min_lr = params.min_lr
        self.factor = params.factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
            threshold=5e-2,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, params):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = params.patience_stopping
        self.min_delta = params.min_delta

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset the counter
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


class EarlyStoppingCustom:
    """Early stops the training if accuracy is in a certain range."""

    def __init__(self, args):
        """
        args:
                necessary arguments for early stopping including =>
                patience (int): How long to wait after last time improved; Default: 5
                verbose (bool): If True, prints a message for each improvement; Default: False
                min_delta (float): Minimum change in the monitored quantity to qualify as an improvement; Default: 0

        """
        self.patience = args.patience_stopping
        self.verbose = args.verbose_stopping
        self.delta = args.min_delta
        self.threshold_val_acc = args.threshold_val_acc

        # Initializations
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, acc, epoch, bucket_id):
        score = acc

        if self.best_score is None:
            self.best_score = score
        elif acc - torch.tensor(self.threshold_val_acc) >= self.delta:
            # elif acc >= torch.tensor(97):
            self.counter += 1

            # if self.counter < self.patience:
            #     print(
            #         f"Early stopping at ep:{epoch} for the bkt:{bucket_id}, cnt:{self.counter}/{self.patience}"
            #     )
            # else:
            #     print(f"Early stopping at ep:{epoch} for the bkt:{bucket_id}")

            if self.counter < self.patience and self.early_stop == False:
                print(
                    f"Early stopping at ep:{epoch} for the bkt:{bucket_id}, cnt:{self.counter}/{self.patience}"
                )
            elif self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0


class EarlyStoppingCustomLoss:
    """Early stops the training if loss is in a certain range."""

    def __init__(self, args):
        """
        args:
                necessary arguments for early stopping including =>
                patience (int): How long to wait after last time improved; Default: 5
                verbose (bool): If True, prints a message for each improvement; Default: False
                min_delta (float): Minimum change in the monitored quantity to qualify as an improvement; Default: 0

        """
        self.patience = args.patience_stopping
        self.verbose = args.verbose_stopping
        self.delta = args.min_delta_loss
        self.threshold_val_loss = args.threshold_val_loss

        # Initializations
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss, epoch, bucket_id):
        score = loss

        if self.best_score is None:
            self.best_score = score
        elif torch.abs(loss - torch.tensor(self.threshold_val_loss)) <= self.delta:
            self.counter += 1

            if self.counter < self.patience and self.early_stop == False:
                print(
                    f"Early stopping at ep:{epoch} for the bkt:{bucket_id}, cnt:{self.counter}/{self.patience}"
                )
            elif self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0


class EarlyStoppingCustomLossAcc:
    """Early stops the training if loss/accuracy is in a certain range."""

    def __init__(self, args):
        """
        args:
                necessary arguments for early stopping including =>
                patience (int): How long to wait after last time improved; Default: 5
                verbose (bool): If True, prints a message for each improvement; Default: False
                min_delta (float): Minimum change in the monitored quantity to qualify as an improvement; Default: 0

        """
        self.patience = args.patience_stopping
        self.verbose = args.verbose_stopping

        self.delta_acc = args.min_delta
        self.threshold_val_acc = args.threshold_val_acc

        self.delta_loss = args.min_delta_loss
        self.threshold_val_loss = args.threshold_val_loss

        # Initializations
        self.counter = 0
        self.best_score_acc = None
        self.best_score_loss = None
        self.early_stop = False

    def __call__(self, acc, loss, epoch, bucket_id):
        score_acc = acc
        score_loss = loss

        if (self.best_score_acc is None) and (self.best_score_loss is None):
            self.best_score_acc = score_acc
            self.best_score_loss = score_loss

        elif (self.best_score_acc is None) and (self.best_score_loss is not None):
            self.best_score_acc = score_acc

        elif (self.best_score_acc is not None) and (self.best_score_loss is None):
            self.best_score_loss = score_loss

        elif (
            torch.abs(loss - torch.tensor(self.threshold_val_loss)) <= self.delta_loss
        ) or (acc - torch.tensor(self.threshold_val_acc) >= self.delta_acc):
            self.counter += 1

            if self.counter < self.patience and self.early_stop == False:
                print(
                    f"Early stopping at ep:{epoch} for the bkt:{bucket_id}, cnt:{self.counter}/{self.patience}"
                )
            elif self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score_acc = score_acc
            self.best_score_loss = score_loss
            self.counter = 0


class EarlyStoppingCustomUnreg:
    """Early stops the training if accuracy is in a certain range after unregistering."""

    def __init__(self, args):
        """
        args:
                necessary arguments for early stopping including =>
                patience (int): How long to wait after last time improved; Default: 5
                verbose (bool): If True, prints a message for each improvement; Default: False
                min_delta (float): Minimum change in the monitored quantity to qualify as an improvement; Default: 0

        """
        self.base_spk_per_bkt = args.spk_per_bucket
        self.patience = args.patience_stopping
        self.verbose = args.verbose_stopping
        self.delta = args.min_delta
        self.delta_unreg = args.min_delta_unreg
        self.threshold_val_acc = args.threshold_val_acc

        # Initializations
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, acc, spk_per_bkt, epoch, bucket_id):
        score = acc

        if self.best_score is None:
            self.best_score = score

        elif spk_per_bkt == self.base_spk_per_bkt:
            if acc - torch.tensor(self.threshold_val_acc) >= self.delta:
                self.counter += 1

                if self.counter < self.patience and self.early_stop == False:
                    print(
                        f"Early stopping at ep:{epoch} for the bkt:{bucket_id}, cnt:{self.counter}/{self.patience}"
                    )
                elif self.counter >= self.patience:
                    self.early_stop = True

            else:
                self.best_score = score
                self.counter = 0

        elif spk_per_bkt != self.base_spk_per_bkt:
            acc_diff = acc - torch.tensor((spk_per_bkt / self.base_spk_per_bkt) * 100)

            if abs(acc_diff) <= self.delta_unreg:
                self.counter += 1

                if self.counter < self.patience and self.early_stop == False:
                    print(
                        f"Early stopping at ep:{epoch} for the bkt:{bucket_id}, cnt:{self.counter}/{self.patience}"
                    )
                elif self.counter >= self.patience:
                    self.early_stop = True

            else:
                self.best_score = score
                self.counter = 0

        else:
            self.best_score = score
            self.counter = 0


def swa_schedule(swa_start, swa_lr, lr_cls, ep):
    """Schedule the learning rate for stochastic weight averaging (SWA)."""

    t = ep / swa_start

    lr_ratio = swa_lr / lr_cls

    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio

    return lr_cls * factor


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def swa_scheduling(**kwargs):
    """Weight averaging per epoch."""

    # Stochastic
    lr_updated_swa = swa_schedule(
        kwargs["swa_start"], kwargs["swa_lr"], kwargs["lr_cls"], kwargs["epochs"]
    )
    adjust_learning_rate(kwargs["optimizer"], lr_updated_swa)

    kwargs["moving_average"](
        kwargs["classifier_ma"],
        kwargs["classifier"],
        1.0 / (kwargs["ma_n"] + 1),
    )
    kwargs["ma_n"] += 1


def swa_scheduling_unsup(**kwargs):
    """Weight averaging per epoch."""

    # Stochastic
    lr_updated_swa = swa_schedule(
        kwargs["swa_start"], kwargs["swa_lr"], kwargs["lr_cls"], kwargs["epochs"]
    )
    adjust_learning_rate(kwargs["optimizer"], lr_updated_swa)

    kwargs["moving_average"](
        kwargs["dvec_latent_ma"],
        kwargs["dvec_latent"],
        1.0 / (kwargs["ma_n"] + 1),
    )
    kwargs["ma_n"] += 1


def no_ma_scheduling(**kwargs):
    """Update without moving average per epoch"""
    kwargs
