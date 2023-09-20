import importlib


def main():
    # Select the appropriate plugin from "plugins" to run different simulations
    PLUGIN_NAME = "plugins.main_model_params"

    plugin_module = importlib.import_module(PLUGIN_NAME, ".")

    plugin_module.main_execute()


if __name__ == "__main__":
    main()
