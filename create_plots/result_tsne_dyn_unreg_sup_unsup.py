import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from matplotlib import style
from matplotlib.legend import Legend
from pathlib import Path
from utils import create_filenames_unreg_tsne_results


def plot_result_tsne_dyn_unreg_sup_unsup(bucket, args, hparams):

    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_unreg_tsne_results(args, hparams, bucket)

    with open(
        Path(paths_filenames["dir_tsne"], paths_filenames["tsne_pred_feats"]),
        "r",
    ) as pred_feats:
        Xsup = json.load(pred_feats)

    with open(
        Path(paths_filenames["dir_tsne"], paths_filenames["tsne_pred_labels"]),
        "r",
    ) as pred_labels:
        Ysup = json.load(pred_labels)

    # with open(
    #     Path(
    #         paths_filenames_unsup["dir_tsne"], paths_filenames_unsup["tsne_pred_feats"]
    #     ),
    #     "r",
    # ) as pred_feats_unsup:
    #     Xunsup = json.load(pred_feats_unsup)

    # with open(
    #     Path(
    #         paths_filenames_unsup["dir_tsne"], paths_filenames_unsup["tsne_pred_labels"]
    #     ),
    #     "r",
    # ) as pred_labels_unsup:
    #     Yunsup = json.load(pred_labels_unsup)

    dfXsup = pd.DataFrame(Xsup)
    dfXsup[r"dynamic removal"] = Ysup
    dfXsup.columns = ["dim-1", "dim-2", "dynamic removal"]

    # dfXunsup = pd.DataFrame(Xunsup)
    # dfXunsup[r"dynamic registrations"] = Yunsup
    # dfXunsup.columns = ["dim-1", "dim-2", "dynamic registrations"]

    makers = {"Unregistered speakers": "1", "Residual speakers": "+"}

    # Settings
    x = 4  # Want figures to be A4
    plt.rc("figure", figsize=[46.82 * 0.5 ** (0.5 * x), 33.11 * 0.5 ** (0.5 * x)])
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    style.use("ggplot")
    fig = plt.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(111)
    # sns.scatterplot(
    #     data=dfXsup,
    #     x="dim-1",
    #     y="dim-2",
    #     hue=dfXsup["dynamic registrations"],
    #     style="dynamic removal",
    #     ax=ax1,
    #     s=100,
    #     # marker="x",
    # )
    sns.scatterplot(
        data=dfXsup,
        x="dim-1",
        y="dim-2",
        hue=dfXsup["dynamic removal"],
        style="dynamic removal",
        ax=ax1,
        s=50,
        markers=makers,
        alpha=0.5,
        # marker="+",
    )

    # ax2 = fig.add_subplot(212)
    # sns.scatterplot(
    #     data=dfXunsup,
    #     x="dim-1",
    #     y="dim-2",
    #     hue=dfXunsup["dynamic removal"],
    #     style="dynamic removal",
    #     ax=ax2,
    # )

    ax1.legend(prop={"size": 11})
    # ax1.legend()

    plt.tight_layout()

    # fig.savefig(
    #     f"{result_dir_plot_path}/tsne_eval_dyn_unreg_agnt_2.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     dpi=1200,
    # )

    plt.show()
    plt.close()
