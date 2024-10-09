import datetime
import numpy as np
import time
import torch
import utils
import model   
import torch.backends.cudnn as cudnn

from engine import *
from pathlib import Path 
from base_args import get_args
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import get_model, sel_criterion_train, sel_criterion_test, load_checkpoint
from datasets import build_dataset_train, build_dataset_test, BatchSchedulerSampler, collate_fn, build_dataloader

############################################################
def seed_initial(seed=0):
    seed += utils.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    ### Configuration
    utils.init_distributed_mode(args)  # 初始化分布式训练模式
    device = torch.device(args.device)
    seed_initial(seed=args.seed)  # 初始化随机种子
    ####################################### Get the model
    model = get_model(args)
    if args.resume:
        print(args.resume)
        checkpoint_model = load_checkpoint(model, args)
        
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        
    model.to(device)
    model_without_ddp = model  # 保存未经过分布式训练的模型
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module  
    
    print("------------------------------------------------------")
    ############## Get the data and dataloader 获取数据和数据加载器

    # ta_sel表明当前选中的任务类型，分别包含 textr/textc, msa/textr, imgr
    # ta_sel = ['textr','textc']
    ta_sel = ['msa', 'textr']
    # ta_sel = ['imgr']
    trainset_group = build_dataset_train(is_train=True, ta_sel=ta_sel, args=args)  # 构建训练所用数据集组
    trainloader_group= build_dataloader(ta_sel,trainset_group, args=args)  # 依据数据集组构建数据加载器

    ############################################## Get the test dataloader
    valset = None
    if args.ta_perform:
        valset = build_dataset_test(is_train=False, args=args)
        sampler_val = torch.utils.data.SequentialSampler(valset)
    else:
        valset = None

    if valset is not None:
        Collate_fn = collate_fn if args.ta_perform.startswith('msa') else None 
        dataloader_val = torch.utils.data.DataLoader(
            valset, sampler=sampler_val, batch_size=int(1.0 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False, collate_fn=Collate_fn)
    else:
        dataloader_val = None
    
    ############################# Get the optimizer and the other training settings
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = args.num_samples // total_batch_size

    optimizer = create_optimizer(args, model)  # 使用optim_factory中的创建优化器来初始化优化器
    loss_scaler = NativeScaler()  # 动态损失缩放器，用来在反向传播时对损失进行缩放，确保梯度的数值稳定

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(  # 生成一个学习率调度表，学习率随着训练过程呈现余弦衰减
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay  # 指定正则化技术的参数，即训练结束时的权重衰减值
    wd_schedule_values = utils.cosine_scheduler(  # 生成一个weight_decay的调度表，该调度表和lr调度表一致，都呈现余弦衰减
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    
    
    ###################################################### Get the criterion
    """
    根据不同的任务类型，选择不同的损失函数
    """
    criterion_train = sel_criterion_train(args,ta_sel, device)
    criterion_test = sel_criterion_test(args, device)
    
    ################################## Auto load the model in the model record folder
    if args.eval:
        
        if args.ta_perform.startswith('img') or args.ta_perform.startswith('text'):
            test_stats = evaluate(ta_perform=args.ta_perform, 
                                net=model, dataloader=dataloader_val, 
                                device=device, criterion=criterion_test)
            if args.ta_perform.startswith('imgc') or args.ta_perform.startswith('textc'):
                print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
            elif args.ta_perform.startswith('imgr'):
                print(f"Average PSNR on the {len(valset)} test samples: {test_stats['psnr']:.3f}dB")
            elif args.ta_perform.startswith('textr'):
                print(f"Average BLEU on the {len(valset)} test samples: {test_stats['bleu']:.3f}")
        elif args.ta_perform.startswith('msa'):
            test_stats = evaluate_msa(ta_perform=args.ta_perform, 
                                net=model, dataloader=dataloader_val, 
                                device=device, criterion=criterion_test)
            print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
        
        elif args.ta_perform.startswith('vqa'):
            test_stats = evaluate_vqa(ta_perform=args.ta_perform, 
                                net=model, dataloader=dataloader_val, 
                                device=device, criterion=criterion_test)
            print("Overall Accuracy is: %.02f" % (test_stats['overall']))
            print("Per Answer Type Accuracy is the following:")
            for ansType in test_stats['perAnswerType']:
                print("%s : %.02f" % (ansType, test_stats['perAnswerType'][ansType]))
        exit(0)

    ################################## Start Training the T-DeepSC
    print(f"Start training for {args.epochs} epochs")
    max_accuracy = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for trainloader in trainloader_group.values():
                trainloader.sampler.set_epoch(epoch)

        # 训练模型
        train_stats = train_epoch_uni(
                model, criterion_train, trainloader_group, optimizer, device, epoch, loss_scaler, 
                ta_sel, args.clip_grad,  start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
                update_freq=args.update_freq)
   

        # 打印训练时间
        inter_time = time.time() - start_time
        inter_time_str = str(datetime.timedelta(seconds=int(inter_time)))
        print('Training time {}'.format(inter_time_str))

        # 训练中定期保存模型的检查点
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=None)
        # 根据不同任务类型，打印相应的模型指标
        if dataloader_val is not None:
            print(args.output_dir)
            if args.ta_perform.startswith('img') or args.ta_perform.startswith('text'):
                test_stats = evaluate(ta_perform=args.ta_perform, 
                                    net=model, dataloader=dataloader_val, 
                                    device=device, criterion=criterion_test)
            elif args.ta_perform.startswith('vqa'):
                test_stats = evaluate_vqa(ta_perform=args.ta_perform, 
                                    net=model, dataloader=dataloader_val, 
                                    device=device, criterion=criterion_test)
            else:
                test_stats = evaluate_msa(ta_perform=args.ta_perform, 
                                    net=model, dataloader=dataloader_val, 
                                    device=device, criterion=criterion_test)
            if args.ta_perform.startswith('imgc') or args.ta_perform.startswith('textc'):
                print(f"Accuracy of the network on the {len(valset)} test images: {test_stats['acc']*100:.3f}")
            elif args.ta_perform.startswith('imgr'):
                print(f"Average PSNR on the {len(valset)} test images: {test_stats['psnr']:.3f} dB")
            elif args.ta_perform.startswith('textr'):
                print(f"Average BLEU on the {len(valset)} test samples: {test_stats['bleu']:.3f}")
            elif args.ta_perform.startswith('msa'):
                print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
            elif args.ta_perform.startswith('vqa'):
                print("Overall Accuracy is: %.02f" % (test_stats['overall']))
                print("Per Answer Type Accuracy is the following:")
                for ansType in test_stats['perAnswerType']:
                    print("%s : %.02f" % (ansType, test_stats['perAnswerType'][ansType]))
       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
