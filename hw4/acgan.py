"""""""""
Pytorch implementation of Conditional Image Synthesis with Auxiliary Classifier GANs (https://arxiv.org/pdf/1610.09585.pdf).
This code is based on Deep Convolutional Generative Adversarial Networks in Pytorch examples : https://github.com/pytorch/examples/tree/master/dcgan
"""""""""
from __future__ import print_function
import argparse
import os
import random
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from folder import Face
import model

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default = 'hw4_data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--pairNum',type=int,default=10,help='pair of male/female faces to save')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='acgan_output', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")



# folder dataset
dataset = Face(
    root=opt.dataroot,
    transform=transforms.Compose([
        transforms.Scale(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))





nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nb_label = 2

netG = model.netG(nz, ngf, nc)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.netD(ndf, nc, nb_label)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

s_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(2*opt.pairNum, nz, 1, 1).normal_(0, 1)
s_label = torch.FloatTensor(opt.batchSize)
c_label = torch.LongTensor(opt.batchSize)

real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    s_criterion.cuda()
    c_criterion.cuda()
    input, s_label = input.cuda(), s_label.cuda()
    c_label = c_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_noise_ = np.random.normal(0, 1, (opt.pairNum, nz))
fixed_noise_ = np.vstack((fixed_noise_,fixed_noise_))
label_onehot = np.zeros((2*opt.pairNum, nb_label))
label_onehot[np.arange(opt.pairNum), 0] = 1
label_onehot[np.arange(opt.pairNum,2*opt.pairNum), 1] = 1
fixed_noise_[np.arange(2*opt.pairNum), :nb_label] = label_onehot[np.arange(2*opt.pairNum)]

fixed_noise_ = np.reshape(fixed_noise_,(2*opt.pairNum, nz, 1, 1))
fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise.data.copy_(fixed_noise_)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def class_acc(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return 100.*correct/len(labels.data)

def rf_acc(predict, labels):
    pred = torch.round(predict)
    correct = torch.eq(pred.data,labels.data).cpu().sum()
    return 100.*correct/len(labels.data)



# log csv file
with open(os.path.join(opt.outf,'acgan_log.csv'), 'w') as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["500-steps", "s_errD_real","s_errD_fake","s_errG","rf_acc_real","rf_acc_fake","c_errD_real","c_errD_fake","c_errG","class_acc_real","class_acc_fake","G_loss","D_loss"])
    writer.writeheader()



for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        
        ###########################
        # (1) Update D network
        ###########################
        # train with real
        netD.zero_grad()
        img, label = data
        batch_size = img.size(0)
        input.data.resize_(img.size()).copy_(img)
        s_label.data.resize_(batch_size).fill_(real_label)
        c_label.data.resize_(batch_size).copy_(label)
        s_output, c_output = netD(input)
        s_errD_real = s_criterion(s_output, s_label)
        c_errD_real = c_criterion(c_output, c_label)
        errD_real = s_errD_real + c_errD_real
        errD_real.backward()
        D_x = s_output.data.mean()
        
        class_acc_real = class_acc(c_output, c_label)
        rf_acc_real = rf_acc(s_output, s_label) 
        
        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)

        label = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
        
        noise_ = np.reshape(noise_,(batch_size, nz, 1, 1))
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_)

        c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)
        s_label.data.fill_(fake_label)
        s_output,c_output = netD(fake.detach())
        s_errD_fake = s_criterion(s_output, s_label)
        c_errD_fake = c_criterion(c_output, c_label)
        errD_fake = s_errD_fake + c_errD_fake

        errD_fake.backward()
        D_G_z1 = s_output.data.mean()
        errD = s_errD_real + s_errD_fake
        optimizerD.step()

        class_acc_fake = class_acc(c_output, c_label)
        rf_acc_fake = rf_acc(s_output, s_label)
        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        s_label.data.fill_(real_label)  # fake labels are real for generator cost
        s_output,c_output = netD(fake)
        s_errG = s_criterion(s_output, s_label)
        c_errG = c_criterion(c_output, c_label)
       
        errG = s_errG + c_errG
        errG.backward()
        D_G_z2 = s_output.data.mean()
        optimizerG.step()
        
        if i % 500 == 0:
            vutils.save_image(img,
                    '%s/real_samples.png' % opt.outf)
            #fake = netG(fixed_cat)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                              '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),nrow=10)

            print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, class_acc_real: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[-1], errG.data[0], D_x, D_G_z1, D_G_z2,
                 class_acc_real))
            # wirte rows into acgan_logs.csv
            with open(os.path.join(opt.outf,'acgan_log.csv'), 'a') as outcsv:
                writer = csv.DictWriter(outcsv, fieldnames = ["500-steps", "s_errD_real","s_errD_fake","s_errG","rf_acc_real","rf_acc_fake","c_errD_real","c_errD_fake","c_errG","class_acc_real","class_acc_fake","G_loss","D_loss"])
                writer.writerow({"500-steps":10*epoch+(i/500),"s_errD_real":s_errD_real.data[0],"s_errD_fake":s_errD_fake.data[0],"s_errG":s_errG.data[0],"rf_acc_real":rf_acc_real,"rf_acc_fake":rf_acc_fake,"c_errD_real":c_errD_real.data[0],"c_errD_fake":c_errD_fake.data[0],"c_errG":c_errG.data[0],"class_acc_real":class_acc_real,"class_acc_fake":class_acc_fake,"G_loss":errG.data[0],"D_loss":errD.data[-1]})
                '''
                print("500-steps :{}".format(10*epoch+(i/500)),"s_errD_real:{}".format(s_errD_real.data[0]),"s_errD_fake:{}".format(s_errD_fake.data[0]),"s_errG:{}".format(s_errG.data[0]),"rf_acc_real:{}".format(rf_acc_real),"rf_acc_fake:{}".format(rf_acc_fake),"c_errD_real:{}".format(c_errD_real.data[0]),"c_errD_fake:{}".format(c_errD_fake.data[0]),"c_errG:{}".format(c_errG.data[0]),"class_acc_real:{}".format(class_acc_real),"class_acc_fake:{}".format(class_acc_fake),"G_loss:{}".format(errG.data[0]),"D_loss:{}".format(errD.data[-1]))
                '''
         
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
