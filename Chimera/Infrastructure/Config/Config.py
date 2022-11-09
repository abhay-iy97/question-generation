import json
import argparse
from Chimera.Infrastructure.ConfigInjector import ConfigInjectorBase


def FromJsonFile(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config


def FromArgs():  # todo specify type of file if needed
    parser = argparse.ArgumentParser(description='parse the path to config file')
    parser.add_argument('-c', '--config', dest='configPath', default='Chimera/Infrastructure/Config/config.json')
    args = parser.parse_args()
    path = args.configPath
    config = FromJsonFile(path)
    ConfigInjectorBase.config = config
    return config
