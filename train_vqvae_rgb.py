######################################################################
# Script for training VQ-VAE with RGB images.                        #
# Supports 8-bit 64x64x3 and 128x128x3 images (0-255 values)         #
# Creates 4x4 latent codes for both image sizes.                     #
# @author: Patrick Ribu Gorton                                       #
######################################################################

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from scheduler import CycleScheduler

from tqdm import tqdm
import argparse
import numpy as np
from utils import util


def train(epoch, loader_rgb, loader_seg, model, optimizer, scheduler, device, img_size):
    loader_rgb = tqdm(loader_rgb)
    model.train()
    criterion = nn.MSELoss(reduction='none')
    #criterion = pytorch_ssim.SSIM3D(window_size = 33)
    criterion.to(device)

    latent_loss_weight = 0.25
    sample_size = 8

    loss_sum = 0
    loss_n = 0

    seg_iter = iter(loader_seg)
    for ep_idx, (episode, _) in enumerate(loader_rgb):
        episode_seg, _ = next(seg_iter)

        # Since we don't shuffle the dataloaders, we create and shuffle an order
        # to be used for both the rgb and segmentation episodes
        order = np.arange(episode.shape[1])
        np.random.shuffle(order)

        for i in range(episode.shape[1]):
            model.zero_grad()

            # Get and resize current batch of images
            img = episode[:,order[i]]
            img = util.resize_img(img, type='rgb', size=img_size)
            img = img.to(device)

            # Getting the bce_weights
            img_seg = episode_seg[:,order[i]]
            img_seg = util.resize_img(img_seg, type='seg', size=img_size).to(device)
            class_weights = util.seg_weights(img_seg).to(device)

            out, latent_loss, top = model(img)

            # Calculate loss
            recon_loss = criterion(out, img)
            recon_loss = (recon_loss*class_weights).mean()
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss

            loss.backward()
            optimizer.step()

            recon_loss_item = recon_loss.item()
            latent_loss_item = latent_loss.item()
            loss_sum += recon_loss_item * img.shape[0]
            loss_n += img.shape[0]


            lr = optimizer.param_groups[0]['lr']

            loader_rgb.set_description(
                (
                    f'epoch: {epoch + 1}; mse: {recon_loss_item:.5f}; '
                    f'latent: {latent_loss_item:.3f}; avg mse: {loss_sum / loss_n:.5f}; '
                    f'lr: {lr:.5f}; '
                )
            )

            if i == 100:
                model.eval()
                sample = img[:sample_size]

                with torch.no_grad():

                    out, _, top = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f'sample/{str(epoch + 1).zfill(4)}_{img_size}x{img_size}_GAN.png',
                    nrow=sample_size,
                )
                model.train()

def val(epoch, loader, model, optimizer, scheduler, device, img_size):

    model.eval()
    loader = tqdm(loader)
    criterion = nn.MSELoss()
    criterion.to(device)

    sample_size = 8

    loss_sum = 0
    loss_n = 0

    with torch.no_grad():
        for ep_idx, (episode, _) in enumerate(loader):

            for i in range(episode.shape[1]):

                # Get, resize and reconstruct current batch of images
                img = episode[:,i]
                img = util.resize_img(img, type='rgb', size=img_size)
                img = img.to(device)

                out, latent_loss, _ = model(img)

                recon_loss = criterion(out, img)
                latent_loss = latent_loss.mean()

                loss_sum += recon_loss.item() * img.shape[0]
                loss_n += img.shape[0]

                lr = optimizer.param_groups[0]['lr']

                loader.set_description(
                    (
                        f'validation; mse: {recon_loss.item():.5f}; '
                        f'latent: {latent_loss.item():.3f}; avg mse: {loss_sum/loss_n:.5f}; '
                        f'lr: {lr:.5f}'
                    )
                )

    model.train()

    return loss_sum/loss_n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--train', type=str, default='/shared/users/patriri/carla_data/rgb/vqvae/train/')
    parser.add_argument('--val', type=str, default='/shared/users/patriri/carla_data/rgb/vqvae/val/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--decoder_only', type=bool, default=False)

    args = parser.parse_args()
    print(args)

    device = args.device
    torch.cuda.set_device(device)

    def npy_loader(path):
        sample = torch.from_numpy(np.load(path))
        return sample

    dataset_train = datasets.DatasetFolder(
        root=args.train,
        loader=npy_loader,
        extensions='.npy',
    )
    dataset_val = datasets.DatasetFolder(
        root=args.val,
        loader=npy_loader,
        extensions='.npy',
    )
    dataset_seg = datasets.DatasetFolder(
        root='/shared/users/patriri/carla_data/seg/vqvae/more_ped_vehic/',
        loader=npy_loader,
        extensions='.npy',
    )

    loader_train = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=False)
    loader_val = DataLoader(dataset_val, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=False)
    loader_seg = DataLoader(dataset_seg, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=False)

    model = util.load_model('rgb', device, args.ckpt, args.img_size)
    if args.ckpt is not None : print('Resuming training...')
    model.train()


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )


    current_best = 999999
    for i in range(args.epoch):
        train(i, loader_train, loader_seg, model, optimizer, scheduler, device, args.img_size)
        torch.save(model.state_dict(), f'checkpoint/rgb/4mars/vqvae_{args.img_size}x{args.img_size}.pt')


        #val_loss = val(i, loader_val, model, optimizer, scheduler, device, args.img_size)

        # if val_loss < current_best:
        #     current_best = val_loss
        #     print('Model improved.')
        #     torch.save(
        #         model.state_dict(), f'checkpoint/rgb/4mars/vqvae_{args.img_size}x{args.img_size}_improved.pt'
        #     )
