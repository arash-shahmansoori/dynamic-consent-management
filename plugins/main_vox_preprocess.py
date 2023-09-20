from utils import parse_args
from create_disjoint_train_test_dataset import vox_preprocess


def main_execute():
    args = parse_args()

    root_dir = "data\\voxceleb1"

    logic_name = "train"
    output_dir = args.output_dir_vox_train

    vox_preprocess(args, root_dir, output_dir, logic_name)
