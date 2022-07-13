import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from GPNet import GPNet

from datasets import RandomDataset, BatchDataset, BalancedBatchSampler
from utils import accuracy, AverageMeter, save_checkpoint

import logging
import time

name = 'test'
time_run = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
save_dir = 'log'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

logfile = save_dir + '/' + name + '_' + time_run + '.log'
logging.basicConfig(format='%(asctime)s-%(pathname)s-%(levelname)s:%(message)s', level=logging.INFO, filename=logfile,
                    filemode='a')
logging.info("OK!")

parser = argparse.ArgumentParser(description='GPNet')
parser.add_argument('exp_name', default=None, type=str,
                    help='name of experiment')
parser.add_argument('data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-ep', '--after_epoch', default=8, type=int,
                    help='number of fine-tune')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--evaluate-freq', default=10, type=int,
                    help='the evaluation frequence')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('n_classes',  type=int,
                    help='the number of classes')
parser.add_argument('n_samples',  type=int,
                    help='the number of samples per class')
parser.add_argument('seed', type=int,
                    help='random seed')

parser.add_argument('circle_mode', default=False, type=bool,
                    help='the mode of circle_loss')

parser.add_argument('m', default=0.25, type=float,
                    help='the margin of circle_loss')
parser.add_argument('gamma', default=128, type=float,
                    help='the gamma of circle_loss')

parser.add_argument('weight_circle', default=0.001, type=float,
                    help='the weight of circle_loss')
parser.add_argument('weight_multi', default=0.01, type=float,
                    help='the weight of muti_loss')
parser.add_argument('weight_rank', default=1, type=float,
                    help='the weight of rank_loss')

parser.add_argument('dimensionality', default=200, type=float,
                    help='the dimensionality of map1')

parser.add_argument('mask', default='B', type=str,
                    help='the type of mask')
parser.add_argument('mask_thred', default=0.2, type=float,
                    help='the thred of mask')
parser.add_argument('mask_bia', default=0.2, type=float,
                    help='the bia of mask')

parser.add_argument('GP', default=True, type='str',
                    help='generality and particularity')


best_prec1 = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def convert_label_to_similarity(feature, label):  # -> Tuple[Tensor, Tensor]:

    eudist = pdist(feature)

    max = torch.max(eudist)
    logging.info("max\n{}".format(max))
    similarity_matrix = torch.div((max - eudist),max)
    logging.info('similarity_matrix\n{}'.format(similarity_matrix))

    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m, gamma):  # -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn):  # -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class CircleLossLikeCE(nn.Module):
    def __init__(self, m, gamma):
        super(CircleLossLikeCE, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inp, label):
        a = torch.clamp_min(inp + self.m, min=0).detach()
        src = torch.clamp_min(
            - inp.gather(dim=1, index=label.unsqueeze(1)) + 1 + self.m,
            min=0,
        ).detach()
        a.scatter_(dim=1, index=label.unsqueeze(1), src=src)

        sigma = torch.ones_like(inp, device=inp.device, dtype=inp.dtype) * self.m
        src = torch.ones_like(label.unsqueeze(1), dtype=inp.dtype, device=inp.device) - self.m
        sigma.scatter_(dim=1, index=label.unsqueeze(1), src=src)

        return self.loss(a * (inp - sigma) * self.gamma, label)


args = parser.parse_args()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    global args, best_prec1

    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # create model
    model = GPNet()
    logging.info(model)
    model = model.to(device)


    parameters_conv=model.conv.parameters()


    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    criterion = nn.CrossEntropyLoss().to(device)

    global likece
    # likece=True
    likece = args.circle_mode

    if likece:
        criterion2 = CircleLossLikeCE(m=args.m, gamma=args.gamma).to(device)
    else:
        criterion2 = CircleLoss(m=args.m, gamma=args.gamma).to(device)

    optimizer_conv = torch.optim.SGD(parameters_conv, args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    fc_parameters = [value for name, value in model.named_parameters() if 'conv' not in name]
    optimizer_fc = torch.optim.SGD(fc_parameters, args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print ('loading checkpoint {}'.format(args.resume))
            logging.info('loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_conv.load_state_dict(checkpoint['optimizer_conv'])
            optimizer_fc.load_state_dict(checkpoint['optimizer_fc'])
            print ('loaded checkpoint {}(epoch {})'.format(args.resume, checkpoint['epoch']))
            logging.info('loaded checkpoint {}(epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print ('no checkpoint found at {}'.format(args.resume))
            logging.info('no checkpoint found at {}'.format(args.resume))

    cudnn.benchmark = True
    # Data loading code
    train_dataset = BatchDataset(transform=transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.RandomCrop([448, 448]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )]))

    train_sampler = BalancedBatchSampler(train_dataset, args.n_classes, args.n_samples)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)
    scheduler_conv = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_conv, 100 * len(train_loader))
    scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fc, 100 * len(train_loader))

    step = 0
    print('START TIME:', time.asctime(time.localtime(time.time())))
    for epoch in range(args.start_epoch, args.epochs):
        step = train(train_loader, model, criterion, criterion2, optimizer_conv, scheduler_conv, optimizer_fc,
                     scheduler_fc, epoch, step)


def train(train_loader, model, criterion, criterion2, optimizer_conv, scheduler_conv, optimizer_fc, scheduler_fc, epoch,
          step):
    global best_prec1

    batch_time = AverageMeter()
    data_time = AverageMeter()
    softmax_losses = AverageMeter()

    softmax_losses_circle = AverageMeter()
    multi_softmax_losses = AverageMeter()

    rank_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    end = time.time()
    rank_criterion = nn.MarginRankingLoss(margin=0.05)
    softmax_layer = nn.Softmax(dim=1).to(device)

    criterion3 = nn.MultiLabelMarginLoss().to(device)

    for i, (input, target) in enumerate(train_loader):
        model.train()

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = input.cuda()

        target_var = target.cuda() - 1

        # compute output
        logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2, classsfier_out_for_circle_loss, map1_out_sigmoid = model(
            input_var, target_var, flag='train')
        batch_size = logit1_self.shape[0]
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)

        self_logits = torch.zeros(2 * batch_size, 200).to(device)
        other_logits = torch.zeros(2 * batch_size, 200).to(device)
        self_logits[:batch_size] = logit1_self
        self_logits[batch_size:] = logit2_self
        other_logits[:batch_size] = logit1_other
        other_logits[batch_size:] = logit2_other

        # compute loss
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        softmax_loss = criterion(logits, targets)

        target_var = target_var.squeeze()

        if likece:
            softmax_loss_circle = criterion2(classsfier_out_for_circle_loss, target_var)
        else:
            matrix1, matrix2 = convert_label_to_similarity(classsfier_out_for_circle_loss, target_var)
            softmax_loss_circle = criterion2(matrix1, matrix2)

        self_scores = softmax_layer(self_logits)[torch.arange(2 * batch_size).to(device).long(),
                                                 torch.cat([labels1, labels2], dim=0)]
        other_scores = softmax_layer(other_logits)[torch.arange(2 * batch_size).to(device).long(),
                                                   torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones([2 * batch_size, ]).to(device)
        rank_loss = rank_criterion(self_scores, other_scores, flag)

        label_neg = torch.full((map1_out_sigmoid.size(0), map1_out_sigmoid.size(1)), -1).to(device).long()

        label_neg[:, 0] = labels1
        label_neg[:, 1] = labels2

        multi_loss = criterion3(map1_out_sigmoid, label_neg)

        loss = softmax_loss + args.weight_rank*rank_loss + args.weight_circle* softmax_loss_circle + args.weight_multi * multi_loss


        # measure accuracy and record loss
        prec1 = accuracy(logits, targets, 1)
        prec5 = accuracy(logits, targets, 5)
        losses.update(loss.item(), 2 * batch_size)
        softmax_losses.update(softmax_loss.item(), 4 * batch_size)
        rank_losses.update(rank_loss.item(), 2 * batch_size)

        softmax_losses_circle.update(softmax_loss_circle.item(), classsfier_out_for_circle_loss.size(0))
        multi_softmax_losses.update(multi_loss.item(), 4 * batch_size)

        top1.update(prec1, 4 * batch_size)
        top5.update(prec5, 4 * batch_size)

        # compute gradient and do SGD step
        optimizer_conv.zero_grad()
        optimizer_fc.zero_grad()
        loss.backward()
        if epoch >= args.ep:
            optimizer_conv.step()
        optimizer_fc.step()
        scheduler_conv.step()
        scheduler_fc.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Time: {time}\nStep: {step}\t Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                  'RankLoss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'

                  'circleSoftmaxLoss {softmax_loss_circle.val:.4f} ({softmax_loss_circle.avg:.4f})\t'
                  'Multi_softmax_losses {multi_softmax_losses.val:.4f} ({multi_softmax_losses.avg:.4f})\t'

                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, softmax_loss=softmax_losses, rank_loss=rank_losses,
                softmax_loss_circle=softmax_losses_circle, multi_softmax_losses=multi_softmax_losses,
                top1=top1, top5=top5, step=step, time=time.asctime(time.localtime(time.time()))))
            logging.info('Time: {time}\nStep: {step}\t Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                         'RankLoss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'

                         'circleSoftmaxLoss {softmax_loss_circle.val:.4f} ({softmax_loss_circle.avg:.4f})\t'
                         'Multi_softmax_losses {multi_softmax_losses.val:.4f} ({multi_softmax_losses.avg:.4f})\t'

                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, softmax_loss=softmax_losses, rank_loss=rank_losses,
                softmax_loss_circle=softmax_losses_circle, multi_softmax_losses=multi_softmax_losses,
                top1=top1, top5=top5, step=step, time=time.asctime(time.localtime(time.time()))))

        if i == len(train_loader) - 1:
            val_dataset = RandomDataset(transform=transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.CenterCrop([448, 448]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )]))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1

            if is_best == True:
                print('best')
                logging.info('best')
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer_conv': optimizer_conv.state_dict(),
                'optimizer_fc': optimizer_fc.state_dict(),
            }, is_best)

        step = step + 1
    return step


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    softmax_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input_var = input.to(device)

            target_var = target.to(device).squeeze()

            # compute output
            logits = model(input_var, targets=None, flag='val')


            target_var = target_var - 1
            softmax_loss = criterion(logits, target_var)


            prec1 = accuracy(logits, target_var, 1)
            prec5 = accuracy(logits, target_var, 5)
            softmax_losses.update(softmax_loss.item(), logits.size(0))
            top1.update(prec1, logits.size(0))
            top5.update(prec5, logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Time: {time}\nTest: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                    top1=top1, top5=top5, time=time.asctime(time.localtime(time.time()))))
                logging.info('Time: {time}\nTest: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                    top1=top1, top5=top5, time=time.asctime(time.localtime(time.time()))))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    main()
