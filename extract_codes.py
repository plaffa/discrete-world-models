import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import lmdb
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE_seg, VQVAE_seg2, VQVAE_rgb, VQVAE_rgb2
from utils import util
import numpy as np
import os


def extract(loader, model, device, img_size, img_type, destination):

    loader = tqdm(loader)

    for ep_idx, (episode, _) in enumerate(loader):
        sequence = None

        for i in range(episode.shape[1]):
            img = episode[:,i]
            img = util.resize_img(img, type=img_type, size=img_size)
            img = img.to(device)
            if img_type == 'seg' : img = util.one_hot(img, device=device)

            with torch.no_grad() : _, _, code = model.encode(img)

            if i == 0:
                sequence = code.reshape(1, img.shape[0], -1)
            else:
                sequence = torch.cat((sequence, code.reshape(1,img.shape[0], -1)), 0)

        sequence = sequence.cpu().numpy()
        np.save(destination + f'/episode_{ep_idx}', sequence)

def npy_loader(path):
        sample = torch.from_numpy(np.load(path))
        return sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--img_type', type=str, default='rgb')
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()

    device = args.device
    torch.cuda.set_device(device)

    # Initialise model
    model = util.load_model(args.img_type, device, args.ckpt, args.img_size)
    model.eval()

    if args.dataset == None : dataset = {'train', 'val'}
    else : dataset = {args.dataset}

    for ds in dataset:
        dataset_path = f'/shared/users/patriri/carla_data/{args.img_type}/rnn/{ds}/'
        dataset = datasets.DatasetFolder(
            root=dataset_path,
            loader=npy_loader,
            extensions='.npy',
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

        destination = f'extracted_codes/{args.img_type}/{ds}/{args.img_size}/episodes/'
        if not os.path.exists(destination):
            os.makedirs(destination)

        print(f'Extracting latent codes from dataset: {ds} ...')
        extract(loader, model, device, args.img_size, args.img_type, destination)
        print(f'Done! Saved under ~/{destination}')
