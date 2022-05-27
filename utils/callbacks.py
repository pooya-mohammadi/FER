from pathlib import Path
from typing import Union
import torch
from deep_utils import ModelCheckPointTorch, log_print, CSVLogger, TensorboardTorch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.config_utils import Config


def get_opt_callbacks(config: Config,
                      model_path: Union[str, Path],
                      csv_path: Union[str, Path],
                      tensorboard_dir: Union[str, Path],
                      net,
                      monitor_val=None,
                      logger=None,
                      verbose=1):
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=config.optimizer.lr,
                                momentum=config.optimizer.momentum,
                                nesterov=config.optimizer.nesterov,
                                weight_decay=config.optimizer.weight_decay)

    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     mode=config.optimizer.mode,
                                     factor=config.optimizer.lr_decay,
                                     patience=config.optimizer.lr_patience,
                                     min_lr=config.optimizer.min_lr,
                                     verbose=config.optimizer.verbose)

    model_checkpointer = ModelCheckPointTorch(model_path,
                                              net,
                                              optimizer=optimizer,
                                              scheduler=lr_scheduler,
                                              monitor_val=monitor_val,
                                              logger=logger,
                                              verbose=verbose
                                              )
    csv_logger = CSVLogger(csv_path, logger=logger, verbose=verbose)
    tensorboard = TensorboardTorch(tensorboard_dir, logger=logger, verbose=verbose)
    log_print(logger,
              "Successfully created optimizer & callbacks [lr_scheduler,"
              " model_checkpointer, optimizer, csv_logger, tensorboard]!",
              verbose=verbose)
    return model_checkpointer, optimizer, lr_scheduler, csv_logger, tensorboard
