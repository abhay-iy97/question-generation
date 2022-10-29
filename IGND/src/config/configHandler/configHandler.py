import yaml
import argparse
from .configInjector import ConfigInjector

def _get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=False, type=str, help='path to the config file')
    args = vars(parser.parse_args())
    return args


def _print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def get_config_from_args():
    args = _get_args()
    config_path = args['config'] or 'config/msmarco/graph2seq_static_bert.yml'
    try:
        config = _get_config(config_path)
        ConfigInjector.config = config
        return config
    except Exception as e:
        print(f'error: config could not be loaded from path: {config_path}:')
        print(e)
        print('exiting')
        exit(-1)
