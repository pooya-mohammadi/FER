from deep_utils import load_yaml
import os
from datetime import datetime
import sys
from deep_utils import KeyValStruct, YamlConfig


# from .mobilenets import mobilenets
# from .resnet50 import resnet50

# configs = dict(
#     mobilenets=mobilenets,
#     resnet50=resnet50
# )


def get_main_config(model_name='mobilenet'):
    id_ = str(datetime.now().date()) + "_" + str(datetime.now().time())
    main_config = dict(
        model_name=model_name,
        model=dict(
            seed=42,
            base_model=dict(
                include_top=False,
            ),
            input_shape=(128, 128),
            input_channel=3
        ),
        mlflow=dict(mlflow_source='./mlruns',
                    ngrok=False,
                    experiment_name=f"skin-cls-{model_name}",
                    run_name=id_),
        callbacks=dict(
            tb_log_dir='logs/',
            early_stopping_p=8,
            save_weight_only=False,
            plateau_min_lr=1e-6,
            plateau_patience=3,
            monitor='val_loss',
            mode='auto',
            model_path=f"weights/{model_name}/skin-cls-{model_name}"
        ),
        train=dict(
            train=True,
            optimizer=dict(
                type='adam',
                lr=0.001
            ),
            epochs=20,
            batch_size=32,
            kfold=False,
            metrics=['recall', 'precision', 'auc', 'accuracy'],
            from_checkpoint=False,
            checkpoint_path=f"weights/{model_name}/skin-cls-{model_name}",
            trainable=False,
            workers=4,
            use_multiprocessing=True,
        ),
        finetune=dict(
            optimizer=dict(
                type='adam',
                lr=0.0001
            ),
            finetune=True,
            epochs=30,
            batch_size=32,
            trainable=54,
            from_checkpoint=False,
            checkpoint_path=f"weights/{model_name}/skin-cls-{model_name}",
            workers=4,
            use_multiprocessing=True,
        ),

        data=dict(
            records_path='data/train',
            val_size=0.1,
            shuffle_buffer=1024 * 8,
            batch_method=1,
            load_all=True,
            train_idx=[0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14],
            val_idx=[1, 5, 12],
            test_idx=[],
            load_test=False,
            train_aug=True,
            val_aug=True,
            test_aug=False,
            cache=True,
        ),

        evaluate=dict(
            plot_roc=True,
            plot_pr=True
        ),

        loss=dict(
            loss_name="binary_crossentropy",
            label_smoothing=0,
            focal_loss_gamma=0,
        ),

        detect=dict(
            trainable=True,
            weight_path=f"weights/{model_name}/skin-cls-{model_name}"
        ),

    )
    return main_config


def update_dict(org: dict, target: dict):
    for target_k, target_v in target.items():
        org_v = org.get(target_k, None)
        if isinstance(target_v, dict) and isinstance(org_v, dict):
            update_dict(org_v, target_v)
        elif org_v is None or type(org_v) in [str, int, list, tuple, bool, float]:
            org[target_k] = target_v
        else:
            print(org_v)
            raise ValueError("This isn't considered")
    return org


def add_args(config: dict):
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        k, v = args[i], args[i + 1]
        k = k.replace('--', "").replace('-', '_')
        hierarchy = k.split('.')
        hierarchy = "".join(f"['{h}']" for h in hierarchy)
        conf_h = "config" + hierarchy
        try:
            exec_str = """conf_val = {0}
type_ = type(conf_val)
if type_ is bool:
    v=True if v == 'True' else False
elif v == 'None':
    v = None
else:
    v = type_(v)
{0}=v""".format(conf_h)
            exec(exec_str)
        except KeyError:
            raise ValueError(f"{k} is not a valid config parameter!")

    return config


def create_weight_paths(model_name):
    os.makedirs(f'weights/{model_name}', exist_ok=True)


def get_config(model_name):
    target_config = configs.get(model_name, dict())
    config = update_dict(get_main_config(model_name), target_config)
    config = add_args(config)
    config = Config(**config)
    create_weight_paths(model_name)
    return config


class AugmentationStruct(KeyValStruct):
    def __init__(self, **kwargs):
        self.name = None
        self.mean = None
        self.std = None
        self.cutmix = None
        super(AugmentationStruct, self).__init__(**kwargs)


class Loss(KeyValStruct):
    def __init__(self, **kwargs):
        self.name = None
        self.class_weight = None
        super(Loss, self).__init__(**kwargs)


class Optimizer(KeyValStruct):
    def __init__(self, **kwargs):
        self.lr = None
        self.momentum = None
        self.nesterov = None
        self.weight_decay = None
        self.lr_decay = None
        self.lr_patience = None
        self.min_lr = None
        self.verbose = None
        self.mode = None
        self.monitor_val = None
        super(Optimizer, self).__init__(**kwargs)


class Dataset(KeyValStruct):
    def __init__(self, **kwargs):
        self.train_path = None
        self.val_path = None
        self.test_path = None

        self.img_w = None
        self.img_h = None
        self.img_channel = None

        self.train_shuffle = None
        self.val_shuffle = None
        self.test_shuffle = None
        self.num_workers = None
        self.batch_size = None

        super(Dataset, self).__init__(**kwargs)


class ModelConfig(KeyValStruct):
    def __init__(self, **kwargs):
        self.name = None
        self.dropout_p = None
        self.n_classes = None
        self.in_channels = None
        super(ModelConfig, self).__init__(**kwargs)


class Config(YamlConfig):
    def __init__(self, **kwargs):
        self.network = None
        self.epochs = None
        self.model_path = None
        self.device = None

        self.dataset = Dataset(**kwargs.get("dataset", dict()))
        self.augmentation = AugmentationStruct(**kwargs.get("augmentation", dict()))
        self.loss = Loss(**kwargs.get("class_weights", dict()))
        self.optimizer = Optimizer(**kwargs.get("optimizer", dict()))
        self.model = ModelConfig(**kwargs.get('model', dict()))
        super(Config, self).__init__(**kwargs)
