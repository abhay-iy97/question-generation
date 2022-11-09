import argparse
import json
from Chimera.Infrastructure.Core import FromConfig

def FromJsonFile(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config

class ArgResolver:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='parse the path to config file')
        self.config = {}

    def Parse(self):
        self.parser.add_argument('-c', '--config', dest='configPath', default='Chimera/Infrastructure/Config/config.json')
        self.parser.add_argument('-d', '--data-path', dest='dataPath', default='Chimera/Data/sample_dataset.csv')
        self.parser.add_argument('--base_config', dest='baseConfigPath', default='Chimera/Infrastructre/Config/config')
        args = self.parser.parse_args()
        config = FromJsonFile(args.configPath)
        baseConfig = FromJsonFile(args.baseConfigPath)
        config = {**FromJsonFile(args.baseConfigPath), **FromJsonFile(args.configPath)}
        config['data_path'] = args.dataPath
        self.config = config
        FromConfig(config)
        return config


