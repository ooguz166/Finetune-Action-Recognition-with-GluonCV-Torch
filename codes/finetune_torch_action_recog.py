
import torch
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter

from gluoncv.torch.data import build_dataloader
from gluoncv.torch.utils.model_utils import save_model
from gluoncv.torch.utils.task_utils import train_classification, validation_classification
from gluoncv.torch.engine.config import get_cfg_defaults

from gluoncv.torch.utils.utils import build_log_dir
from gluoncv.torch.utils.lr_policy import GradualWarmupScheduler
from gluoncv.torch.model_zoo.action_recognition.slowfast import slowfast_4x16_resnet50_kinetics400

from config.paths import SLOW_FAST_4_16_RES50_FINETUNE_UCF_YAML

def main_worker(cfg):
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    cfg.freeze()

    # create model
    # from my own zoo imported from the gluoncv codes to "action_recognition_models"
    model = slowfast_4x16_resnet50_kinetics400(cfg)

    # Handle gradient ops according to the pretraining flag
    requires_grad=False
    if cfg.CONFIG.MODEL.PRETRAINED == False:
        requires_grad=True

    # # Finetune
    for param in model.parameters():
        param.requires_grad = requires_grad
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(cfg.CONFIG.MODEL.FC_NUM, cfg.CONFIG.MODEL.FINETUNE_CLASS)

    # Use cuda
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    # create dataset and dataloader
    train_loader, val_loader, train_sampler, val_sampler, mg_sampler = build_dataloader(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.CONFIG.TRAIN.LR, momentum=cfg.CONFIG.TRAIN.MOMENTUM,
                                weight_decay=cfg.CONFIG.TRAIN.W_DECAY)


    # Optimize the learning rate
    if cfg.CONFIG.TRAIN.LR_POLICY == 'Step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=cfg.CONFIG.TRAIN.LR_MILESTONE,
                                                         gamma=cfg.CONFIG.TRAIN.STEP)
    elif cfg.CONFIG.TRAIN.LR_POLICY == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=cfg.CONFIG.TRAIN.EPOCH_NUM - cfg.CONFIG.TRAIN.WARMUP_EPOCHS,
                                                               eta_min=0,
                                                               last_epoch=cfg.CONFIG.TRAIN.RESUME_EPOCH)
    else:
        print('Learning rate schedule %s is not supported yet. Please use Step or Cosine.')

    # Warm-up
    if cfg.CONFIG.TRAIN.USE_WARMUP:
        scheduler_warmup = GradualWarmupScheduler(optimizer,
                                                  multiplier=(cfg.CONFIG.TRAIN.WARMUP_END_LR / cfg.CONFIG.TRAIN.LR),
                                                  total_epoch=cfg.CONFIG.TRAIN.WARMUP_EPOCHS,
                                                  after_scheduler=scheduler)


    criterion = nn.CrossEntropyLoss().cuda()

    base_iter = 0
    for epoch in range(cfg.CONFIG.TRAIN.EPOCH_NUM):
        # Not using DISTRIBUTED but gluon-torch uses, just turn of in your cfg file
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        # Train using gluon-torch functions
        base_iter = train_classification(base_iter, model, train_loader, epoch, criterion, optimizer, cfg, writer=writer)
        if cfg.CONFIG.TRAIN.USE_WARMUP:
            scheduler_warmup.step()
        else:
            scheduler.step()

        # if cfg.CONFIG.TRAIN.MULTIGRID.USE_LONG_CYCLE:
        #     if epoch in cfg.CONFIG.TRAIN.MULTIGRID.LONG_CYCLE_EPOCH:
        #         mg_sampler.step_long_cycle()

        if epoch % cfg.CONFIG.VAL.FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1:
            validation_classification(model, val_loader, epoch, criterion, cfg, writer)

        if epoch % cfg.CONFIG.LOG.SAVE_FREQ == 0:
            if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 or cfg.DDP_CONFIG.DISTRIBUTED == False:
                save_model(model, optimizer, epoch, cfg)

    if writer is not None:
        writer.close()


if __name__ == '__main__':

    cfg = get_cfg_defaults()
    cfg.merge_from_file(SLOW_FAST_4_16_RES50_FINETUNE_UCF_YAML)
    main_worker(cfg)
