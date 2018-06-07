import shutil
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
from resnet import resnet50
from cnn_dataset import cnn_dataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='hw5_cnn',metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nframe', default=16, type=int, metavar='N',
                    help='number of frames of a video')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='checkpoint_cnn_4.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--finetune', default = True,dest='finetune', action='store_true',
                    help='fine tune pre-trained model')

best_prec1 = 0

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
             
            self.classifier = nn.Sequential(
                #nn.Linear(16384, 2048),
                nn.Linear(2048, num_classes),
                nn.Softmax()
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True



    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        #print('y.shape={}'.format(y.shape))
        return y


def main():
    global args, best_prec1
    args = parser.parse_args()

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'valid')
    # Get number of classes from train directory
    num_classes = 11
    print("num_classes = '{}'".format(num_classes))
    # create model
    if args.finetune:
        print("=> using pre-trained model '{}'".format(args.arch))
        #original_model = models.__dict__[args.arch](pretrained=True)
        original_model = resnet50(pretrained=True)
        model = FineTuneModel(original_model, args.arch, num_classes)
        print(model)
    
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     #model = models.__dict__[args.arch]()
    #     model = resnet50()
    
    # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #     model.features = torch.nn.DataParallel(model.features)
    #     model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    #model.fc=nn.Linear(512*4,11)
    cudnn.benchmark = True

    # Data loading code
    print('train_dataset: start')
    start_time = time.time()
    train_dataset = cnn_dataset(traindir)
    elapsed_time = time.time() - start_time
    print('train_dataset loading takes: {}'.format(elapsed_time))
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    print('val_dataset: start') 
    start_time = time.time()
    val_dataset = cnn_dataset(valdir)
    elapsed_time = time.time() - start_time
    print('val_dataset loading takes: {}'.format(elapsed_time))
    

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=65, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
    #                            args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 args.lr, 
                                 betas=(0.9, 0.999), 
                                 eps=1e-08, 
                                 weight_decay=args.weight_decay, 
                                 amsgrad=False)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        print('before training, epoch = {}'.format(epoch))
        train(train_loader, model, criterion, optimizer, epoch)
        print('after training, epoch = {}'.format(epoch))

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end) 
        
        # compute output and mean every nframe features
        #print('about to inference model') 
        #print('input.shape={}'.format(input.shape)) 
        input = input.permute((0,1,4,2,3)) 
        #print('input.shape={}'.format(input.shape)) 
        output = torch.zeros((args.nframe,input.shape[0],11))
        for f_id in range(args.nframe):
          input_var = torch.squeeze(input[:,f_id,:,:,:],1)
          #print('input_var.shape={}'.format(input_var.shape))
          input_var = torch.autograd.Variable(input_var,requires_grad=False) 
          #print('model(input_var).shape={}'.format(model(input_var).shape))
          output[f_id,:,:] = model(input_var)
        output = torch.mean(output,dim=0)
        #print('output.shape={}'.format(output.shape))
        #print('after inference model')

        # compute loss
        #target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target,requires_grad=False) 
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        
        # compute output and mean every nframe features
        #print('about to inference model') 
        #print('input.shape={}'.format(input.shape)) 
        input = input.permute((0,1,4,2,3))
        #print('input.shape={}'.format(input.shape)) 
        output = torch.zeros((args.nframe,input.shape[0],11))
        for f_id in range(args.nframe):
          input_var = torch.squeeze(input[:,f_id,:,:,:],1)
          #print('input_var.shape={}'.format(input_var.shape))
          input_var = torch.autograd.Variable(input_var,requires_grad=False) 
          #print('model(input_var).shape={}'.format(model(input_var).shape))
          output[f_id,:,:] = model(input_var)
        output = torch.mean(output,dim=0)
        #print('output.shape={}'.format(output.shape))
        #print('after inference model')

        # compute loss
        #target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target,requires_grad=False) 
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint_cnn_4.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_cnn_4.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
