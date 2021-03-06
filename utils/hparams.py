import os

hps = {
    'network': 'vgg',  # which network do you want to train
    'name': 'my_vgg',  # whatever you want to name your run
    'n_epochs': 300,
    'restore_epoch': None,  # continue training from a specific saved point
    'start_epoch': 0,
    'lr': 0.01,  # starting learning rate
    'lr_decay': 0.75,
    'weight_decay': 0.0001,
    'save_freq': 20,  # how often to create checkpoints
    'drop': 0.1,
    'bs': 2,
    'data_path': '../data/fer2013.csv',
    'model_save_dir': ".",
    'crop_size': 40,

}

possible_nets = set(filename.split(".")[0] for filename in os.listdir('../models'))


def setup_hparams(name, network, **kwargs):
    hps['name'] = name
    hps['network'] = network
    hps.update(kwargs)

    if hps['network'] not in possible_nets:
        raise ValueError("Invalid network.\nPossible ones include:\n - " + '\n - '.join(possible_nets))

    try:
        hps['n_epochs'] = int(hps['n_epochs'])
        hps['start_epoch'] = int(hps['start_epoch'])
        hps['save_freq'] = int(hps['save_freq'])
        hps['lr'] = float(hps['lr'])
        hps['drop'] = float(hps['drop'])
        hps['bs'] = int(hps['bs'])

        if hps['restore_epoch']:
            hps['restore_epoch'] = int(hps['restore_epoch'])
            hps['start_epoch'] = int(hps['restore_epoch'])

        # make sure we can save checkpoints regularly or at least at the end of training
        if hps['n_epochs'] < 20:
            hps['save_freq'] = min(5, hps['n_epochs'])

    except Exception:
        raise ValueError("Invalid input parameters")

    # create checkpoint directory
    hps['model_save_dir'] = os.path.join(hps['model_save_dir'], 'checkpoints', hps['name'])

    os.makedirs(hps['model_save_dir'], exist_ok=True)
    print("[INFO] hyper-parameters: ", hps)
    return hps
