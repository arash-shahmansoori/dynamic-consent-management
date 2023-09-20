from utils import parse_args, HyperParams
from create_disjoint_train_test_dataset import preprocess


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Clean utts for corresponding utts percentage
    # output_dir = args.output_dir_train_ninty_pcnt  # For training
    # output_dir = args.output_dir_val  # For validation

    # Noisy utts
    # output_dir = args.output_dir_nonreduced_train_other  # For training
    output_dir = args.output_dir_val_other  # For validation

    pcnt_old = hparams.pcnt_old  # Select the appropriate utts percentage (%)

    # Clean utts
    # root_name = (
    #     f"./data/LibriSpeech_modular/agnt_{args.agnt_num}_spks_{args.n_speakers}"
    # )
    # file_name = f"data/LibriSpeech_modular/agnt_{args.agnt_num}_spks_{args.n_speakers}/Speakers.txt"

    # Noisy utts
    root_name = f"./data/LibriSpeech_modular_other/agnt_{args.agnt_num}_spks_{args.n_speakers_other}"
    file_name = f"data/LibriSpeech_modular_other/agnt_{args.agnt_num}_spks_{args.n_speakers_other}/Speakers.txt"

    preprocess(args, output_dir, root_name, file_name, pcnt_old)
