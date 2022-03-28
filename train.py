import os
import time
import torch
from data.fer2013 import get_data_loaders
from models import get_model
from utils.loops import train, evaluate
from deep_utils import get_logger, mkdir_incremental, easy_argparse, yaml_config2yaml_file, log_print, color_str
from utils.config_utils import Config
from utils.callbacks import get_opt_callbacks
from utils.losses import get_loss


def run(config,
        optimizer,
        model,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        model_checkpointer,
        lr_scheduler,
        csv_logger,
        tensorboard,
        logger=None):
    # Train with FP16
    scaler = torch.cuda.amp.GradScaler()
    total_tic = time.time()
    for epoch in range(config.epochs):
        tic = time.time()
        acc, loss = train(model,
                          train_loader,
                          criterion,
                          optimizer,
                          scaler,
                          epoch=epoch,
                          config=config)
        val_acc, val_loss = evaluate(model, val_loader, criterion, epoch, config)

        # update callbacks
        logs = dict(acc=acc, loss=loss, val_acc=val_acc, val_loss=val_loss)
        tensorboard(epoch, **logs)
        csv_logger(**logs)
        model_checkpointer(val_loss)
        lr_scheduler.step(val_loss, epoch=epoch)

        # logger
        lr = optimizer.param_groups[0]['lr']
        log_print(logger,
                  color_str(
                      f"[Epoch] {epoch}/{config.epochs} -> [ETA]: {int(time.time() - tic)}S - [train-acc]: {acc:0.2f} - [train-loss]: {loss:0.4f} - [val-acc]: {val_acc:0.2f} - [val-loss]: {val_loss:0.4f} - [lr]: {lr:0.6f}",
                      color='green'))

    # Calculate the model's performance on the test set
    test_acc, test_loss = evaluate(model, test_loader, criterion, config.epochs, config)
    log_print(logger, color_str(
        f"[ETA]: {time.time() - total_tic}S - [test-acc]: {test_acc:0.2f} - [test-loss]: {test_loss:0.4f}",
        color='red'))


def main():
    args = easy_argparse(
        dict(name="--config_path", default='configs/vgg.yml', type=str, help='the path to config file!'),
        dict(name="--fix_name", default=None, help="Whether to save a model in the provided fixed directory,"
                                                   " default is False!")
    )
    config = Config.load_config(args.config_path)
    checkpoint_path = mkdir_incremental(dir_path=os.path.join(config.model_path, config.model.name),
                                        fix_name=args.fix_name)
    # write config.yml to disc
    yaml_config2yaml_file(config, checkpoint_path / f"{config.model.name}.yaml")
    # load logger
    logger = get_logger(f"FER_{config.network}", log_path=checkpoint_path / 'FER.log')
    logger.propagate = False
    # get data_loader
    train_loader, val_loader, test_loader = get_data_loaders(config, logger=logger)
    # load model
    model = get_model(model_name=config.model.name, device=config.device, model_config=config.model)

    model_checkpointer, optimizer, lr_scheduler, csv_logger, tensorboard = get_opt_callbacks(config,
                                                                                             model_path=checkpoint_path / "weights" / "model.pt",
                                                                                             csv_path=checkpoint_path / "log.csv",
                                                                                             tensorboard_dir=checkpoint_path / 'tensorboard',
                                                                                             net=model,
                                                                                             monitor_val=config.optimizer.monitor_val,
                                                                                             logger=logger
                                                                                             )
    loss_fn = get_loss(config, logger=logger)
    run(config, optimizer, model, loss_fn, train_loader, val_loader, test_loader, model_checkpointer, lr_scheduler,
        csv_logger, tensorboard, logger=logger)


if __name__ == "__main__":
    main()
