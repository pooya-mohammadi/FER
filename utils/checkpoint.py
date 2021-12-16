import os
import torch


def save(net, logger, hps, optimizer, scheduler, name):
    # Create the path the checkpint will be saved at using the epoch number

    # path = os.path.join(hps['model_save_dir'], 'epoch_' + str(epoch))

    # create a dictionary containing the logger info and model info that will be saved
    checkpoint = {
        'logs': logger.get_logs(),
        'params': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict()
    }

    # save checkpoint
    best_path = os.path.join(hps['model_save_dir'], name)
    torch.save(checkpoint, best_path)
    # torch.save(checkpoint, path)


def restore(net, logger, hps, optimizer, scheduler, get_best):
    """ Load back the model and logger from a given checkpoint
        epoch detailed in hps['restore_epoch'], if available"""
    if get_best:
        path = os.path.join(hps['model_save_dir'], 'best')
    else:
        path = os.path.join(hps['model_save_dir'], 'epoch_' + str(hps['restore_epoch']))

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)

            logger.restore_logs(checkpoint['logs'])
            net.load_state_dict(checkpoint['params'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Network Restored from {path}!")

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e)
            hps['start_epoch'] = 0

    else:
        print("Restore point unavailable. Training from scratch.")
        hps['start_epoch'] = 0


def model_restore(save_path, net, optimizer, scheduler, restore_from: str):
    path = os.path.join(save_path, restore_from)

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)

            net.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Network Restored from {path}!")

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            raise e
            # hps['start_epoch'] = 0

    else:
        print("Restore point unavailable. Training from scratch.")
        # hps['start_epoch'] = 0


def load_features(model, params):
    """ Load params into all layers of 'model'
        that are compatible, then freeze them"""
    model_dict = model.state_dict()

    imp_params = {k: v for k, v in params.items() if k in model_dict}

    # Load layers
    model_dict.update(imp_params)
    model.load_state_dict(imp_params)

    # Freeze layers
    for name, param in model.named_parameters():
        param.requires_grad = False
