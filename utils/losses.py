import torch
from deep_utils import log_print
from torch import nn
import numpy as np


def get_loss(config, logger=None, verbose=1):
    if config.loss.class_weight is not None:
        class_weights = torch.FloatTensor(np.array(config.loss.class_weight)).to(config.device)
        criterion = nn.CrossEntropyLoss(class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    log_print(logger,
              f"Successfully Created loss: {config.loss.name} with class_weights: {config.loss.class_weight is not None}", verbose=verbose)
    return criterion
