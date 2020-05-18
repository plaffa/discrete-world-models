######################################################################
# Code for training a VQ-VAE with semantic segmentation images.      #
# Supports 64x64x1 and 128x128x1 images (13 classes used here.)      #
# Creates 4x4 latent codes for both image sizes.                     #
#                                                                    #
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


def train(epoch, loader, model, optimizer, scheduler, device, img_size):

    loader = tqdm(loader)
    model.train()

    criterion = nn.BCELoss(reduction='none')
    criterion.to(device)

    latent_loss_weight = 0.25
    sample_size = 8

    bce_sum = 0
    bce_n = 0

    for ep_idx, (episode, _) in enumerate(loader):
        for i in range(episode.shape[1]):

            model.zero_grad()
            # Get, resize and one-hot encode current batch of images
            img = episode[:,i]
            img = util.resize_img(img, type='seg', size=img_size)
            img = img.to(device)
            #bce_weight = util.loss_weights(img).to(device)
            bce_weight = util.seg_weights(img, out_channel=13).to(device)
            img = util.one_hot(img, device=device)


            out, latent_loss, _ = model(img)

            recon_loss = criterion(out, img)
            recon_loss = (recon_loss*bce_weight).mean()
            latent_loss = latent_loss.mean()

            loss = recon_loss + latent_loss_weight * latent_loss


            loss.backward()
            if scheduler is not None:
                scheduler.step()
            optimizer.step()

            recon_loss_item = recon_loss.item()
            latent_loss_item = latent_loss.item()
            bce_sum += recon_loss.item() * img.shape[0]
            bce_n += img.shape[0]

            lr = optimizer.param_groups[0]['lr']

            loader.set_description(
                (
                    f'epoch: {epoch + 1}; bce: {recon_loss_item:.5f}; '
                    f'latent: {latent_loss_item:.3f}; avg bce: {bce_sum/bce_n:.5f}; '
                    f'lr: {lr:.5f}; '
                )
            )

            if i % 100 == 0:
                model.eval()
                sample = img[:sample_size]

                with torch.no_grad():
                    out, _, _ = model(sample)
                    #print('id[0]: ', id[0])
                # Convert one-hot semantic segmentation to RGB
                sample = util.seg_to_rgb(sample)
                out = util.seg_to_rgb(out)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f'sample/seg/{str(epoch + 1).zfill(4)}_{img_size}x{img_size}.png',
                    nrow=sample_size,
                )
                model.train()


def val(epoch, loader, model, optimizer, scheduler, device, img_size):

    model.eval()
    loader = tqdm(loader)
    sample_size = 8

    criterion = nn.BCELoss(reduction='none').to(device)
    bce_sum = 0
    bce_n = 0

    with torch.no_grad():
        for ep_idx, (episode, _) in enumerate(loader):
            for i in range(episode.shape[1]):

                img = episode[:,i]
                img = util.resize_img(img, type='seg', size=img_size)
                img = img.to(device)

                bce_weight = util.loss_weights(img).to(device)
                img = util.one_hot(img, device=device)

                out, latent_loss, _ = model(img)

                recon_loss = criterion(out, img)
                recon_loss = (recon_loss*bce_weight).mean()
                latent_loss = latent_loss.mean()

                bce_sum += recon_loss.item() * img.shape[0]
                bce_n += img.shape[0]

                lr = optimizer.param_groups[0]['lr']

                loader.set_description(
                    (
                        f'validation; bce: {recon_loss.item():.5f}; '
                        f'latent: {latent_loss.item():.3f}; avg bce: {bce_sum/bce_n:.5f}; '
                        f'lr: {lr:.5f}'
                    )
                )

    model.train()
    return bce_sum/bce_n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--train', type=str, default='/shared/users/patriri/carla_data/seg/vqvae/more_ped_vehic/')
    parser.add_argument('--val', type=str, default='/shared/users/patriri/carla_data/seg/vqvae/more_ped_vehic/')
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
    loader_train = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=False)
    loader_val = DataLoader(dataset_val, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=False)

    model = util.load_model('seg', device, args.ckpt, args.img_size)
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
        train(i, loader_train, model, optimizer, scheduler, device, args.img_size)
        #val_loss = val(i, loader_val, model, optimizer, scheduler, device, args.img_size)
        torch.save(model.state_dict(), f'checkpoint/seg/4mars/vqvae_{args.img_size}x{args.img_size}.pt')
        # if val_loss < current_best:
        #     current_best = val_loss
        #     print('Model improved.')
        #     torch.save(
        #         model.state_dict(), f'checkpoint/seg/n_embed_128_embed_dim_128/vqvae_{args.img_size}x{args.img_size}_improved.pt'
        #     )
