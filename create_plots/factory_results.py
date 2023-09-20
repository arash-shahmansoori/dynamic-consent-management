from typing import Callable, Any
from functools import partial


from typing import Callable


plot_result_function: dict[str, Callable[..., None]] = {}


def register(result_type: str, plt_fn):
    plot_result_function[result_type] = plt_fn


def create(plt_type: str, plt_params: Any):

    try:
        if type(plt_params).__qualname__ == "list":
            creator_func = partial(plot_result_function[plt_type], *plt_params)
        elif type(plt_params).__qualname__ == "dict":
            creator_func = partial(plot_result_function[plt_type], plt_params)
    except KeyError:
        raise ValueError
    return creator_func


# plot_result_function["plot_metrics_sup"] = partial(
#     plot_metrics_sup,
#     spk_per_bkt_collection,
#     create_filenames_results_collection,
# )
# plot_result_function["plot_result_dyn_reg_val_acc_vs_pcnt_old_sup"]
# plot_result_function["plot_result_dyn_reg_val_acc_vs_elapsed_time_sup"]
# plot_result_function["plot_result_tsne_new_reg_sup"]
# plot_result_function["plot_result_unreg_re_reg_sup"]
# plot_result_function["plot_cfm_unreg_sup"]

# plot_result_function["plot_metrics_unsup"]
# plot_result_function["plot_result_dyn_reg_val_acc_vs_pcnt_old_unsup"]
# plot_result_function["plot_result_dyn_reg_val_acc_vs_elapsed_time_unsup"]
# plot_result_function["plot_result_tsne_new_reg_unsup"]
# plot_result_function["plot_result_unreg_re_reg_unsup"]