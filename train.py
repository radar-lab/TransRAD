#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
#from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

## For radar dataset
import utils.radar_loader as loader
from utils.radar_dataset_generator import RadarDataset, radar_dataset_collate



if __name__ == "__main__":
    
    torch.cuda.empty_cache()

    Cuda            = True

    seed            = 11

    distributed     = False

    sync_bn         = False

    fp16            = False

    classes_path    = 'model_data/radar_classes.txt' 

    model_path      = 'model_data/best_epoch_weights.pth'
    #------------------------------------------------------#
    #   input_shape  
    #------------------------------------------------------#
    input_shape     = [256, 256, 64]  

    phi             = 'n'

    pretrained      = False

    mosaic              = False
    mosaic_prob         = 0.5
    mixup               = False
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   label_smoothing 
    #------------------------------------------------------------------#
    label_smoothing     = 0

    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32

    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4 #12 #16

    Freeze_Train        = False


    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "adam" #"sgd"
    momentum            = 0.937
    weight_decay        = 0 #5e-4

    lr_decay_type       = "cos"

    save_period         = 10

    save_dir            = 'logs'

    eval_flag           = True
    eval_period         = 10

    num_workers         = 0



    seed_everything(seed)

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0


    class_names, num_classes = get_classes(classes_path)


    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)    


    
    ###################     Radar Dataset Pre-Processing    ######################
    #----------------------------------------------------#
    #   Load Radar parameters with using .json
    #----------------------------------------------------#    
    config = loader.readConfig()
    config_data = config["DATA"]
    # config_radar = config["RADAR_CONFIGURATION"]
    # config_model = config["MODEL"]
    # config_train = config["TRAIN"]    

    train_annotation_path   = "train_set_dir" #config_data["train_set_dir"]
    val_annotation_path     = train_annotation_path
    
    #---------------------------#
    #   Split train and validation
    #---------------------------#
    RAD_sequences_train = loader.readSequences(config_data,mode="train")
    RAD_sequences_test = loader.readSequences(config_data,mode="test")
    RAD_sequences_train, RAD_sequences_validate = loader.splitTrain(RAD_sequences_train,validate=True, ratio=0.05)
    num_train   = len(RAD_sequences_train)
    num_val     = len(RAD_sequences_validate) 
    num_test    = len(RAD_sequences_test)    
    
    
    
    ##########################################  Our  Model   #########################        

    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != '':

        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mWarning: Backbone should be loaded successfully!\033[0m")


    yolo_loss = Loss(model)

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    ema = ModelEMA(model_train)
    


    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Not enough data')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1



    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False


        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})


        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Not enough data")

        if ema:
            ema.updates     = epoch_step * Init_Epoch

        train_dataset   = RadarDataset(RAD_sequences_train, config_data, train_annotation_path,input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        train=True, mode = 'RA', target_shape=[256,256,256], interpolation='nearest')
        val_dataset     = RadarDataset(RAD_sequences_validate, config_data, val_annotation_path, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        train=False, mode = 'RA', target_shape=[256,256,256], interpolation='nearest')
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=radar_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=radar_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))


        if local_rank == 0:
            # eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, RAD_sequences_validate, log_dir, Cuda, \
            #                                 eval_flag=eval_flag, period=eval_period)
            eval_callback   = None
        else:
            eval_callback   = None
        

        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size


                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Not enough data")
                    
                if ema:
                    ema.updates     = epoch_step * epoch

                if distributed:
                    batch_size  = batch_size // ngpus_per_node
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=radar_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=radar_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag   = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            print('epoch',epoch,'\n')
            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
