from dataclasses import dataclass


@dataclass
class HyperParams:
    """A class that maintains the hyper-parameters for training."""

    num_of_buckets: int = 8  # Number of buckets

    # Initialize the round for registrations
    round_num: int = -1  # Set to -1 for training before dynamic registrations

    lr_cls: float = 1e-2  # Learning rate of the classifier

    # Hyper parameters of the contrastive feature extraction optimizer
    lr: float = 1e-3  # Use 1e-2 for supervised, and 1e-3 for unsupervised
    momentum: float = 0.9
    nesterov: bool = True
    dampening: int = 0
    weight_decay: float = 1e-4

    # Determine the states of the model chechpoints
    model_str: str = "model"
    opt_str: str = "optimizer"
    contloss_str: str = "criterion"
    start_epoch: str = "epoch"

    # Determine the states and mode of the moving average checkpoints
    ma_mode: str = "no_ma"  # options: swa/no_ma
    model_ma_str: str = "model_ma"
    ma_n_str: str = "ma_n"

    # Determine the mode for training unsupervised contrastive loss as:
    # - "train_dvec": from literature;
    # - "train_dvec_adapted": from literature adapted;
    # - "train_dvec_proposed": proposed;
    # - "train_dvec_latent_proposed": proposed;
    # - "train_dvec_latent_proposed_swa": proposed with including swa;

    train_dvec_mode: str = "train_dvec_proposed"
    train_dvec_latent_mode: str = "train_dvec_latent_proposed"

    # Determine the percentage of old utterances for dynamic registrations
    pcnt_old: str = "full"
