from easydict import EasyDict as edict
import yaml


cfg = edict()

cfg.MODEL = edict()
cfg.MODEL.ARCH = "FPN"
cfg.MODEL.ENCODER_NAME = "resnet34"
cfg.MODEL.ENCODER_WEIGHTS = "imagenet"

cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 108
cfg.TRAIN.WORKERS = 16
cfg.TRAIN.EPOCHS = 80

cfg.DATA = edict()
# DATA.SEARCH
cfg.DATA.FACTOR = 2.0
cfg.DATA.OUTPUT_SIZE = 128
cfg.DATA.CENTER_JITTER = 4.5
cfg.DATA.SCALE_JITTER = 0.5


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


