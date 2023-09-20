import json


from .factory_results import register, create


def create_plot(plt_fn_name):

    plt_type = plt_fn_name.__name__
    plt_fn = plt_fn_name

    register(plt_type, plt_fn)

    with open("create_plots/level.json") as file:
        func_params = json.load(file)

    plot_result_fn = create(plt_type, func_params[plt_type]["params"])

    return plot_result_fn
