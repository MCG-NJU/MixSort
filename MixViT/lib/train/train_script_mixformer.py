import os
# loss function related
from lib.utils.box_ops import giou_loss, ciou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mixformer import build_mixformer, build_mixformer_online_score
from lib.models.mixformer_vit import build_mixformer_vit, build_mixformer_vit_multi, build_mixformer_vit_multi_score, build_mixformer_vit_decoder, build_mixformer_deit_multi_score, build_mixformer_deit
from lib.models.mixformer_vit import build_mixformerpp_vit_multi_score, build_mixformer_vit_window, build_mixformer_sparse_vit, build_mixformer_vit_learn_pos, build_mixformer_vit_decoder_multi_score
# forward propagation related
from lib.train.actors import MixFormerActor
# for import modules
import importlib


def prepare_input(res):
    res_t, res_s = res
    t = torch.FloatTensor(1, 3, res_t, res_t).cuda()
    s = torch.FloatTensor(1, 3, res_s, res_s).cuda()
    return dict(template=t, search=s)


def run(settings):
    settings.description = 'Training script for Mixformer'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    settings.generate_sep_labels = False

    # Create network
    if settings.script_name == "mixformer":
        net = build_mixformer(cfg)
    elif settings.script_name == "mixformer_online":
        net = build_mixformer_online_score(cfg, settings)
    elif settings.script_name == 'mixformerpp_vit_multi' and 'decoder' in settings.config_name:
        net = build_mixformer_vit_decoder_multi_score(cfg, settings)
    elif settings.script_name == 'mixformerpp_vit_multi':
        net = build_mixformerpp_vit_multi_score(cfg, settings)
        # settings.generate_sep_labels = True
    elif settings.script_name == 'mixformer_deit_multi':
        net = build_mixformer_deit_multi_score(cfg, settings)
    elif settings.script_name == 'mixformer_vit_learn_pos':
        net = build_mixformer_vit_learn_pos(cfg)
    elif settings.script_name == 'mixformer_vit' and 'sparse' in settings.config_name:
        net = build_mixformer_sparse_vit(cfg)
    elif settings.script_name == 'mixformer_vit' and 'decoder' in settings.config_name:
        net = build_mixformer_vit_decoder(cfg)
    elif settings.script_name == 'mixformer_vit':
        net = build_mixformer_vit(cfg)
    elif settings.script_name == 'mixformer_vit_multi' and 'score' not in settings.config_name:
        net = build_mixformer_vit_multi(cfg)
    elif settings.script_name == 'mixformer_vit_multi' and 'score' in settings.config_name:
        net = build_mixformer_vit_multi_score(cfg, settings)
    elif settings.script_name == 'mixformer_deit':
        net = build_mixformer_deit(cfg)
    else:
        raise ValueError("illegal script name")

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == 'mixformer' or settings.script_name == 'mixformer_vit' or settings.script_name == 'mixformer_vit_learn_pos' \
            or (settings.script_name == 'mixformer_vit_multi' and 'score' not in settings.config_name) \
            or (settings.script_name == 'mixformer_deit_multi' and 'score' not in settings.config_name) \
            or (settings.script_name == 'mixformerpp_vit_multi' and 'score' not in settings.config_name):
        objective = {'iou': ciou_loss, 'l1': l1_loss}
        loss_weight = {'iou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    # elif (settings.script_name == 'mixformerpp_vit_multi' and 'score' not in settings.config_name):
    #     objective = {'ciou': REGLoss(loss_type='ciou'), 'l1': REGLoss(loss_type='l1'), 'hinge': LBHinge(threshold=0.05)}  # 2d
    #     loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'hinge': cfg.TRAIN.HINGE_WEIGHT}
    #     actor = MixFormerppActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == 'mixformer_online' or 'score' in settings.config_name:
        objective = {'iou': ciou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss()}
        loss_weight = {'iou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)
    elif settings.script_name == 'mixformer_deit':
        actor = MixFormerActor(net=net)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    print("USE_AMP: {}".format(use_amp))
    # Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
    accum_iter = getattr(cfg.TRAIN, "ACCUM_ITER", 1)
    settings.cfg = cfg
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler,
                         accum_iter=accum_iter, use_amp=use_amp)

    # warmup lr
    # settings.epochs = cfg.TRAIN.EPOCH
    # settings.warmup_epochs = cfg.TRAIN.WARMUP_EPOCHS
    # settings.lr = cfg.TRAIN.LR
    # settings.min_lr = cfg.TRAIN.MIN_LR
    # trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, None,
    #                      accum_iter=accum_iter, use_amp=use_amp, shed_args=settings)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
