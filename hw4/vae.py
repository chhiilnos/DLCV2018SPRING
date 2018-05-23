from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from face import FACE

plt.switch_backend('agg')
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--data_root',default = 'hw4_data',help='path to dataset')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--code_len', type=int, default=512, help ='length of code')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', default='vae_output',help='folder to output images and model checkpoints')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    FACE(args.data_root, train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    FACE(args.data_root, train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
tsne_loader = torch.utils.data.DataLoader(
    FACE(args.data_root, train=False, transform=transforms.ToTensor()),
    batch_size=1, shuffle=True, **kwargs)




class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(4096, 400)
        self.fc21 = nn.Linear(400, args.code_len)
        self.fc22 = nn.Linear(400, args.code_len)
        self.fc3 = nn.Linear(args.code_len, 400)
        self.fc4 = nn.Linear(400, 4096)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 4096))
        #print('mu.data.shape = {}'.format(mu.data.shape))
        #print('logvar.data.shape = {}'.format(logvar.data.shape))
        z = self.reparameterize(mu, logvar)
        #print("z.data.[:20] = {}".format(z.data[:20]))
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()
print(model)

def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4096))
    loss = torch.nn.MSELoss()
    #print("x.shape = {}".format(x.shape))
    #print("recon_x.shape = {}".format(recon_x.shape))
    MSE = loss(recon_x,x.view(-1,4096))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 4096
    k_weight = 0.00001
    return MSE + k_weight*KLD,MSE,KLD



optimizer = optim.Adam(model.parameters(), lr=1e-3)

# log csv file
with open(os.path.join(args.outf,'vae_log.csv'), 'w') as outcsv:
  writer = csv.DictWriter(outcsv, fieldnames = ["250-steps","mse","kld"])
  writer.writeheader()



def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss,mse,kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
            # write rows into vae_logs.csv
            with open(os.path.join(args.outf,'vae_log.csv'), 'a') as outcsv:
                writer = csv.DictWriter(outcsv, fieldnames = ["250-steps","mse","kld"])
                writer.writerow({"250-steps":10*epoch+(batch_idx/args.log_interval),"mse":mse.data[0],"kld":kld.data[0]}) 
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def run_tsne():
    codes_0 = []
    codes_1 = []
    for i, (data,label) in enumerate(tsne_loader):
        
        if i>300:
            break
        
        print("in tsne:{}".format(i))
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        fc21,fc22 = model.encode(data.view(-1,4096))
        #print("type(fc21) = {}".format(type(fc21)))
        #print("fc21.shape = {}".format(fc21.shape))
        #print("type(fc22) = {}".format(type(fc22)))
        #print("fc22.shape = {}".format(fc22.shape))
        code = np.hstack((fc21,fc22))
        #print("code.shape = {}".format(code.shape))
        code = np.squeeze(np.asarray(code))
        code = code.flatten()
        #print("after squeeze, code.shape = {}".format(code.shape))
        #print("type(label)={}".format(type(label)))
        #print("len(label)={}".format(len(label)))
        #print("label={}".format(label))
        #print("label[7][0]={}".format(label[7][0]))
        if label[7][0]==0:
            codes_0.append(code)
        elif label[7][0]==1:
            codes_1.append(code)
    #print("codes_0.shape = {}".format(codes_0.shape))
    #print("codes_1.shape = {}".format(codes_1.shape))
    #codes_0 = np.squeeze(np.asarray(codes_0))
    #codes_1 = np.squeeze(np.asarray(codes_1))
    
    print("codes_0.shape = {}".format(np.asarray(codes_0).shape))
    print("codes_1.shape = {}".format(np.asarray(codes_1).shape))
    codes = np.vstack((codes_0,codes_1)) 
    print("type(codes)={}".format(type(codes)))
    #codes = codes.reshape(codes.shape[0],-1)
    print("start PCA")
    pca=PCA(n_components=80)  
    codes = pca.fit_transform(codes)  
    print("start PCA")
    print("start tsne")
    tsne = TSNE(n_jobs=4)
    codes_tsne = tsne.fit_transform(codes)
    print("done tsne")
    #plt.figure(figsize=(16, 16))
    print("len(codes_0)={}".format(len(codes_0)))
    print("len(codes_1)={}".format(len(codes_1)))
    color_0 = np.zeros((len(codes_0),1))
    color_0.fill(13)
    #print(color_0[0][0])
    color_1 = np.ones((len(codes_1),1))
    color = np.concatenate((color_0,color_1))
    print("np.zeros((len(codes_0),1)).shape={}".format(np.zeros((len(codes_0),1)).shape))
    print("np.ones((len(codes_1),1)).shape={}".format(np.ones((len(codes_1),1)).shape))
    #print(np.concatenate((np.zeros((len(codes_0)),1),np.ones((len(codes_1),1)))).shape)

    plt.scatter(codes_tsne[:, 0], codes_tsne[:, 1], c =color)
    #plt.scatter(codes_tsne[:, 0], codes_tsne[:, 1])
    #plt.colorbar()
    #plt.show()
    plt.savefig('tsne.png')
 

def test(epoch):
    model.eval()
    test_loss = 0
    MSE =0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        loss,mse,kld= loss_function(recon_batch, data, mu, logvar)
        MSE+=mse.data[0]
        test_loss+=loss[0]
        if i%160==0:
            print("i here :{}".format(i))
            n = min(data.size(0), 10)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 3, 64, 64)[:n]])
            save_image(comparison.data.cpu(),
                     args.outf + '/reconstruction_' + str(epoch) + '.png', nrow=n)

    MSE/=163
    print("MSE on test set ={}".format(MSE))
    test_loss /= len(test_loader.dataset)
    print(test_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss.data[0]))

    
for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    sample_mu = Variable(torch.randn(32*3, args.code_len))
    sample_logvar = Variable(torch.randn(32*3, args.code_len))
    if args.cuda:
        sample_mu = sample_mu.cuda()
        sample_logvar = sample_logvar.cuda()

    sample = model.reparameterize(sample_mu, sample_logvar)
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    
    save_image(sample.data.view(32, 3, 64, 64),
               args.outf + '/sample_' + str(epoch) + '.png')

run_tsne()

