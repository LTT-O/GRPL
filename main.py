import logging
import os
import torch.nn as nn
import torch
import time
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from tqdm import tqdm
from config import arg_parse
from utils.utlits import show_args
from dataloader import get_data_loader
from model.model_label import MoProGCN
from train import train
from utils.utlits import AverageMeter, accuracy, save_checkpoint, adjust_learning_rate

ImageFile.LOAD_TRUNCATED_IMAGES = True
cudnn.benchmark = True  # 会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
#cudnn.enabled = True


def main():
    args = arg_parse()
    logging.basicConfig(level=logging.INFO, filename='exp/Log/Web-{}'.format(args.data_type), filemode='a')
    writer = SummaryWriter('exp/tensorboard/log-{}'.format(args.data_type))

    show_args(args)
    print("======> Dataloader")
    end = time.time()
    train_loader, test_loader = get_data_loader(args)
    print(f"======> Dataloader finish({time.time()-end:.2f})\n")
    end = time.time()
    print("======> Create model")
    model = MoProGCN(inputDim=2048, nodeNum=10, num_classes=args.num_class)
    # model.load_state_dict(
    #     {k.replace('module.', ''): v for k, v in torch.load(args.resume).items()})
    # model = MoProGCN(inputDim=2048, low_dim=128, num_classes=args.num_class)
    #model.cuda()
    model = nn.DataParallel(model)
    model.cuda()
    print(f"======> Create model finish({time.time()-end:.2f})")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    best_acc = 0
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        train_cls_acc, train_inst_acc, train_cls_loss, train_inst_loss = \
            train(train_loader, model, criterion, optimizer, epoch, args)
        val_acc_top1, val_acc_top5, val_loss = test(model, test_loader, args, criterion)
        if epoch == 20:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=(val_acc_top1 > best_acc), args=args,
                filename='{}/checkpoint/checkpoint_{}_epoch{}.pth'.format(args.exp_dir, args.data_type, epoch+1))
        if val_acc_top1 > best_acc:
            best_acc = val_acc_top1
            torch.save(model.state_dict(),  "{}/model/best_{}_model.pth".format(args.exp_dir, args.data_type))


        logging.info(f'Epoch:[{epoch+1}]\n'
                     f'train_Acc  cls:[{train_cls_acc:6.2f}]   inst:[{train_inst_acc:6.2f}]\n'
                     f'train_Loss cls:[{train_cls_loss:6.2f}]   inst:[{train_inst_loss:6.2f}]\n'
                     f'val_acc  top1:[{val_acc_top1:6.2f}]   top5:[{val_acc_top5:6.2f}]\n'
                     f'val_Loss cls:[{val_loss:6.2f}]\n'
                     f'best_acc: [{best_acc:6.2f}]\n'
                     f'use time: [{time.time() - end:}]'
                     )
        end = time.time()
        writer.add_scalars('Train_Acc',
                           {'cls_acc': train_cls_acc, 'inst_acc': train_inst_acc}, epoch + 1)
        writer.add_scalars('Train_Loss',
                           {'cls_loss': train_cls_loss, 'inst_loss': train_inst_loss}, epoch + 1)
        writer.add_scalars('Val_Acc',
                           {'top1': val_acc_top1, 'top5': val_acc_top5}, epoch + 1)
        writer.add_scalar('Val_Loss', val_loss, epoch + 1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=(val_acc_top1 > best_acc), args=args,
            filename='{}/checkpoint/checkpoint_{}.pth'.format(args.exp_dir, args.data_type))


def test(model, test_loader, args, criterion):
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        cls_Loss = AverageMeter()

        # evaluate on webVision val set
        test_loader = tqdm(test_loader)
        for batch_idx, (img, img_target) in enumerate(test_loader):
            img, img_target = img.cuda(), img_target.cuda()
            img_temp = img
            batch = [img, img_temp, img_target]
            outputs = model(batch=batch, args=args, is_eval=True)
            # outputs = model(batch=batch, args=args, is_proto=True, is_clean=True)
            acc1, acc5 = accuracy(outputs, img_target, topk=(1, 5))
            top1_acc.update(acc1.item(), img.shape[0])
            top5_acc.update(acc5.item(), img.shape[0])
            cls_Loss.update(criterion(outputs, img_target), img.shape[0])

        # average across all processes


        print('WebVision Accuracy is %.2f%% (%.2f%%)' % (top1_acc.avg, top5_acc.avg))

    return top1_acc.avg, top5_acc.avg, cls_Loss.avg


if __name__ == '__main__':
    main()