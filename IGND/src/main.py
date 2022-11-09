import torch
import numpy as np
import random
import os
from config.configHandler.configHandler import get_config_from_args

from core.model_handler import ModelHandler


################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)

    if config['only_test']:
        model.test()
    else:
        model.train()
        model.test()


################################################################################
# Module Command-line Behavior #
################################################################################


if __name__ == '__main__':
    config = get_config_from_args()
    main(config)
