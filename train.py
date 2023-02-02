import time
import torch
from utils.utlits import AverageMeter, accuracy
from tqdm import tqdm


def train(train_loader, model, criterion, optimizer, epoch, args):
    acc_cls = AverageMeter()
    acc_inst = AverageMeter()
    acc_pro = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    cls_Loss = AverageMeter()
    inst_Loss = AverageMeter()
    cls_pro_Loss = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    train_loader = tqdm(train_loader)
    for i, (img, img_target) in enumerate(train_loader):
        img, img_target = img.cuda(), img_target.cuda()
        temp = img_target
        data_time.update(time.time() - end)
        # compute output
        cls_out, cls_pro, img_target, similarity_prototypes, N_similarity_prototypes = \
            model([img, img_target], args, is_proto=(epoch > 0), is_clean=(epoch >= args.start_clean_epoch))

        # cls loss
        cls_loss = criterion(cls_out, img_target)
        cls_Loss.update(cls_loss.item(), img_target.shape[0])

        # pro loss
        if epoch > 0:
            cls_pro_loss = criterion(cls_pro, img_target)
            cls_pro_Loss.update(cls_pro_loss.item(), img_target.shape[0])
            # cls_pro_loss = 0
        else:
            cls_pro_loss = 0

        prototypes_loss = 0
        N_prototypes_loss = 0

        if epoch > 0:
            # prototypical match
            prototypes_loss = 1 - torch.mean(similarity_prototypes)
            N_prototypes_loss = 1 + torch.mean(N_similarity_prototypes)

        loss = cls_loss + cls_pro_loss + args.w_proto * (
        prototypes_loss + N_prototypes_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(cls_out, img_target)[0]
        acc_cls.update(acc.item(), img_target.shape[0])

        acc = accuracy(cls_pro, img_target)[0]
        acc_pro.update(acc.item(), img_target.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                f"[Epoch{epoch + 1}/{args.epochs}]  cls_loss:{cls_Loss.avg:6.2f}  Pro_loss:{cls_pro_Loss.avg:6.2f}"
                f"  cls_acc:{acc_cls.avg:6.2f}  pro_acc:{acc_pro.avg:6.2f}"
                f"  data_time:{data_time.avg:6.2f}  batch_time:{batch_time.avg:6.2f}  tar_num:{img_target.shape[0]}")
            if epoch >= args.start_clean_epoch:
                print(temp)
                print(img_target)

    return acc_cls.avg, acc_inst.avg, cls_Loss.avg, inst_Loss.avg

# def train(train_loader, train_strong_loader, model, criterion, optimizer, epoch, args):
#     acc_cls = AverageMeter()
#     acc_inst = AverageMeter()
#     data_time = AverageMeter()
#     batch_time = AverageMeter()
#     cls_Loss = AverageMeter()
#     inst_Loss = AverageMeter()
#
#     # switch to train mode
#     model.train()
#     end = time.time()
#     train_loader = tqdm(train_loader)
#     for (i, (img, img_target)), (j, (img_str, img_str_target)) in zip(enumerate(train_loader),
#                                                                       enumerate(train_strong_loader)):
#         """
#             batch[0] = img
#             batch[2] = img_aug-增强
#         """
#         img, img_target, img_str = img.cuda(), img_target.cuda(), img_str.cuda()
#         data_time.update(time.time()-end)
#         # compute output
#         cls_out, mom_out, similarity_net, similarity_prototypes = model(batch=[img, img_str, img_target],
#                                                                         args=args, is_proto=(epoch >= 0))
#
#         # cls loss
#         cls_loss = criterion(cls_out, img_target)
#         cls_Loss.update(cls_loss.item(), img.shape[0])
#         inst_loss = criterion(mom_out, img_target)
#         inst_Loss.update(inst_loss, img.shape[0])
#         prototypes_loss = 0
#
#         if epoch >= 0:
#             # prototypical loss
#             prototypes_loss = 1 - torch.sum(similarity_prototypes) / img.shape[0]
#
#         momNet_loss = 1 - torch.sum(similarity_net) / img.shape[0]
#         # print(similarity_net)
#         # print(similarity_prototypes)
#
#         loss = cls_loss + args.w_proto * prototypes_loss  + args.w_inst * momNet_loss
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         acc = accuracy(cls_out, img_target)[0]
#         acc_cls.update(acc.item(), img.shape[0])
#         acc = accuracy(mom_out, img_target)[0]
#         acc_inst.update(acc.item(), img.shape[0])
#
#
#
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             print(f"[Epoch{epoch+1}/{args.epochs}]  cls_loss:{cls_Loss.avg:6.2f}  inst_loss:{inst_Loss.avg:6.2f}"
#                   f"  cls_acc:{acc_cls.avg:6.2f}  inst_acc:{acc_inst.avg:6.2f}"
#                   f"  data_time:{data_time.avg:6.2f}  batch_time:{batch_time.avg:6.2f}")
#
#     return acc_cls.avg, acc_inst.avg, cls_Loss.avg, inst_Loss.avg
