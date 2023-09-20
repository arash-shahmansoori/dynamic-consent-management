from utils import parse_args, HyperParams
from create_plots import create_plot
import importlib


def main():
    args = parse_args()
    hparams = HyperParams()

    # Select the plot of interest
    RESULT_NAME = "create_plots.result_dyn_reg_val_acc_old_utts_pcnt_vs_agnt_sup_unsup"
    result_module = importlib.import_module(RESULT_NAME, ".")

    plt_fn_name = (
        result_module.plot_result_dyn_reg_val_acc_old_utts_pcnt_vs_agnt_sup_unsup
    )

    plot_result_fn = create_plot(plt_fn_name)

    plot_result_fn(args, hparams)


if __name__ == "__main__":
    main()
