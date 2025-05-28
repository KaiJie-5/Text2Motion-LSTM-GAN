import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from embedding_net import EmbeddingNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_iter(target_data, net, optim):

    # zero gradients
    optim.zero_grad()

    # reconstruction loss
    feat, recon_data = net(target_data)

    #print(f"Target Data Shape: {target_data.shape}")  # Should be (batch_size, 32, 24)
    #print(f"Reconstructed Data Shape: {recon_data.shape}")  # Should also be (batch_size, 32, 24)

    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if True:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    recon_loss.backward()
    optim.step()

    ret_dict = {'loss': recon_loss.item()}
    return ret_dict


def make_tensor(path, n_frames=None, stride=None):
    # Load the single .npy file
    data = np.load(path)  # Shape: (31919, 32, 24)
    
    # Normalize the dataset using precomputed mean and std

    # Load the mean vector
    mean_vec = np.array([0.017727887,0.024600636,0.018993093,0.15804641,0.04801874,0.514795,-0.03391232,-0.4199248,-0.17036547,0.054599937,-0.1828417,-0.39755738,0.29908618,0.12296839,-0.14594543,-0.0045795976,0.4695487,
                         -0.18657716,0.045856882,0.23338486,-0.39653006,0.31735834,-0.06377233,-0.12065148])
    
    # Load the std vector
    std_vec = np.array([0.1589865,0.033021253,0.07455321,0.12996687,0.06274846,0.035091236,0.15307683,0.10665058,0.0727581,0.134405,0.13801084,0.1414812,0.16514629,0.25570178,0.2708101,0.14427245,0.10523983,0.06939567,
                        0.13445261,0.13619617,0.1473774,0.15804584,0.25936827,0.2703309])
    
    data = (data - mean_vec) / std_vec  # Normalize

    # Return as a PyTorch tensor
    return torch.Tensor(data)  # Shape: (31919, 32, 24)


def main(tier, n_frames):
    code = 'F' if tier == 'fullbody' else 'U'

    # dataset
    train_dataset = TensorDataset(make_tensor('../src/Data/train_action.npy'))
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

    '''
    # Sanity check: Print the shape of one batch
    for batch in train_loader:
        print(batch[0].shape)  # Should print (batch_size, 32, 24)
        break
    '''

    # train
    loss_meters = [AverageMeter('loss')]

    # interval params
    print_interval = int(len(train_loader) / 5)

    pose_dim = 24  # Update to your dataset's pose dimension

    generator = EmbeddingNet(pose_dim, n_frames).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    # training
    for epoch in range(50):
        for iter_idx, target in enumerate(train_loader, 0):
            target = target[0]
            batch_size = target.size(0)
            target_vec = target.to(device)
            loss = train_iter(target_vec, generator, gen_optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | '.format(epoch, iter_idx + 1)
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                print(print_summary)

    # save model
    gen_state_dict = generator.state_dict()
    save_name = f'../src/Evaluate/FGD/model_checkpoint_{tier}_{n_frames}.bin'
    torch.save({'pose_dim': pose_dim, 'n_frames': n_frames, 'gen_dict': gen_state_dict}, save_name)

if __name__ == '__main__':
    tier = 'upperbody'  # fullbody, upperbody
    n_frames = 32 
    main(tier, n_frames)
