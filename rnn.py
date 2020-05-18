######################################################################
# Code for training an RNN (and predicting) with seuences of latent  #
# codes extraced with the VQ-VAE.                                    #
# Currently supports only 1x64 codes (8x8) with num_embed=32,        #
# though this is fairly easy to modify.                              #
#                                                                    #
# @author: Patrick Ribu Gorton                                       #
######################################################################


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm
import argparse
import numpy as np
from vqvae import VQVAE, VQVAE_seg, VQVAE_rgb, VQVAE_rgb2
from utils import util
import imageio


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0, batch_first=False):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.inp = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), batch_first=batch_first)
        self.out = nn.Linear(hidden_size, input_size)

        self.drop = nn.Dropout(dropout)


    def forward(self, input, hidden=None):


        x = self.drop(self.inp(input))
        x, hidden = self.lstm(x, hidden)
        x = self.out(self.drop(x))
        x = torch.sigmoid(x)

        return x, hidden


# ----- DEFINE TRAINING PROCEDURE ----- #

def train(epoch, loader_train, loader_val, model, optimizer, scheduler, args):

    model.train()
    #criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    #bce_loss = nn.BCELoss()
    train_n = 0
    train_total = 0
    train_acc = 0

    for ep_idx, (episode, d) in enumerate(loader_train):

        loss = 0
        loss2 = 0
        seq_len = episode.shape[1]
        episode = episode.squeeze(2)
        optimizer.zero_grad()

        rng = np.arange(0,seq_len-9,3)

        # Get sequence inputs and target
        for i in range(3):
            optimizer.zero_grad()
            inputs = episode[:,i+rng].long().to(args.device)
            inputs = F.one_hot(inputs, num_classes=args.n_embed).float()
            inputs = inputs.view(episode.shape[0], -1, args.n_embed*64)

            targets = episode[:,i+rng+3].long().to(args.device)


            # Forward pass through the rnn
            outputs, _ = model(inputs)

            outputs = outputs.view(-1,args.n_embed,64)
            targets = targets.view(-1,episode.shape[2]) 

            loss = criterion(outputs, targets)
            loss = (loss*util.embedding_movement(targets).to(args.device)).mean()
            loss = loss.mean()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            train_total += loss.item()
            train_n += 1

            # Compute accuracy
            pred = torch.argmax(outputs, dim=1)
            correct = (pred == targets).float()
            acc = correct.sum() / targets.numel()
            train_acc += acc


    # ---- VALIDATION ----- #
    model.eval()
    with torch.no_grad():

        val_n = 0
        val_total = 0
        val_acc = 0

        for ep_idx, (episode, d) in enumerate(loader_val):

            loss = 0
            seq_len = episode.shape[1]
            episode = episode.squeeze(2)
            # Get sequence inputs and targets

            rng = np.arange(0,seq_len-9,3)
            for i in range(3):
                inputs = episode[:,i+rng].long().to(args.device)
                inputs = F.one_hot(inputs, num_classes=args.n_embed).float()
                inputs = inputs.view(episode.shape[0], -1, args.n_embed*64) 
                targets = episode[:,i+rng+3].long().to(args.device)

                # Forward pass through the rnn
                outputs, _ = model(inputs)
                outputs = outputs.view(-1,args.n_embed,64) # 16
                targets = targets.view(-1,episode.shape[2]) # uncomment
                # Compute loss
                loss = criterion(outputs, targets)
                loss = (loss*util.embedding_movement(targets).to(args.device)).mean()
                loss = loss.mean()
                val_n += 1
                val_total += loss.item()

                # Compute accuracy
                pred = torch.argmax(outputs, dim=1)
                correct = (pred == targets).float()
                acc = correct.sum() / targets.numel()
                val_acc += acc

    # Return train loss, train accuracy, val loss, val accuracy
    return train_total/train_n, train_acc/train_n, val_total/val_n, val_acc/val_n


def predict(loader, model_rnn, model_vqvae, args):

    model_rnn.eval()
    model_vqvae.eval()

    start = 75*3#0 for original

    with torch.no_grad():
        for ep_idx, (episode, _) in enumerate(loader):
            sequence_top = None

            for i in range(start,(args.in_steps+args.steps)*3+start,3):
                img = episode[:,i].to(args.device)
                img = util.resize_img(img, type=args.img_type, size=args.img_size)

                if args.img_type == 'seg' : img = util.one_hot(img, device=args.device)

                if i == start:
                    seq_in = img
                else:
                    seq_in = torch.cat((seq_in, img))

                _, _, top = model_vqvae.encode(img.to(args.device))


                if i == start:
                    sequence_top = top.reshape(1, img.shape[0], -1)
                else:
                    sequence_top = torch.cat((sequence_top, top.reshape(1,img.shape[0], -1)), 0)

            seq_len = sequence_top.shape[0]

            inputs_top = sequence_top.long()
            inputs_top = F.one_hot(inputs_top, num_classes=args.n_embed).float()
            inputs_top = inputs_top.view(-1, img.shape[0],args.n_embed*64) # 16

            # Forward pass through the rnn
            out, hidden = model_rnn(inputs_top[:args.in_steps])

            # Reshape, argmax and prepare for decoding image
            out = out[-1].unsqueeze(0)
            out_top = out.view(-1,args.n_embed,64) # 16
            out_top = torch.argmax(out_top, dim=1)
            out_top_seq = out_top.view(-1,8,8) # 4,4


            out = out_top
            for t in range(args.steps-1):
                # One-hot encode previous prediction
                out = out.long()
                out = F.one_hot(out, num_classes=args.n_embed).float()
                out = out.view(-1, img.shape[0],args.n_embed*64) # 16
                # Predict next frame
                out, hidden = model_rnn(out, hidden=hidden)
                # Argmax and save
                out = out.view(-1,args.n_embed,64) # 16
                out = torch.argmax(out, dim=1)
                out_top_seq = torch.cat((out_top_seq, out.view(-1,8,8)), 0) # 4,4

            decoded_samples = model_vqvae.decode_code(out_top_seq) # old vqvae


            channels = 13 if args.img_type == 'seg' else 3
            seq_out = torch.zeros(args.in_steps, channels, args.img_size, args.img_size).to(device)
            #print('seq_out: ', seq_out.shape, 'decoded_samples: ', decoded_samples[0].shape)
            seq_out = torch.cat((seq_out, decoded_samples),0)

            sequence = torch.cat((seq_in.to(args.device), seq_out.to(args.device)))
            sequence_rgb = util.seg_to_rgb(sequence) if args.img_type == 'seg' else sequence



            ########## save images and measure IoU ############
            if args.save_images is True:
                save_individual_images(sequence_rgb, ep_idx, args)
                utils.save_image(
                    sequence_rgb,
                    f'predictions/test_pred_{ep_idx}.png',
                    nrow=(args.in_steps+args.steps),
                )


def save_individual_images(seq, pred, args):
    # dir for gt and pred concatenated
    dir_together = f'predictions/{args.img_type}/{args.img_size}/together/sequence_{pred}/'
    if not os.path.exists(dir_together):
        os.makedirs(dir_together)
    # dir for ground truth
    dir_gt = f'predictions/{args.img_type}/{args.img_size}/gt/sequence_{pred}/'
    if not os.path.exists(dir_gt):
        os.makedirs(dir_gt)
        os.makedirs(dir_gt + 'gif/')
    # dir for predictions
    dir_pred = f'predictions/{args.img_type}/{args.img_size}/pred/sequence_{pred}/'
    if not os.path.exists(dir_pred):
        os.makedirs(dir_pred)
        os.makedirs(dir_pred + 'gif/')

    # Save individual images
    for i in range(seq.shape[0]//2):
        img = torch.cat([seq[i].unsqueeze(0), seq[i+60].unsqueeze(0)], 0)
        # Save gt and pred concatenated
        utils.save_image(
            img,
            dir_together + f'img_{i}.png',
            nrow=2,
        )
        #if i >= 10:
        # Save only gt
        utils.save_image(
            seq[i],
            dir_gt + f'img_{i}.png',
            nrow=1,
        )
        if i >= 10:
            # Save only pred
            utils.save_image(
                seq[i+60],
                dir_pred + f'img_{i}.png',
                nrow=1,
            )

    # Save GIFs. This will provoke warnings (conversion loss), but it does not matter
    #print(seq[70:].shape)
    if args.save_gif is True:
        # Save GIFs: gt
        seq_gt = torch.cat((seq[10:60].cpu(), torch.zeros(10,3,args.img_size,args.img_size).cpu()), 0)
        images = []
        for img in seq_gt:
            images.append(img.permute(1,2,0).numpy())
        imageio.mimsave(dir_gt + 'gif/' + 'video.gif', images, duration=0.1)
        # Save GIFs: pred
        seq_pred = torch.cat((seq[70:].cpu(), torch.zeros(10,3,args.img_size,args.img_size).cpu()), 0)
        images = []
        for img in seq_pred:
            images.append(img.permute(1,2,0).numpy())
        imageio.mimsave(dir_pred + 'gif/' + 'video.gif', images, duration=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--input_size', type=int, default=2048) # 512
    parser.add_argument('--hidden_size', type=int, default=1024) #512
    parser.add_argument('--n_lstm_layers', type=int, default=2) #2
    parser.add_argument('--n_embed', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.3) #0.2
    parser.add_argument('--epoch', type=int, default=700)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--vqvae', type=str, default=None)
    parser.add_argument('--save_images', type=bool, default=True)
    parser.add_argument('--save_gif', type=bool, default=True)
    #parser.add_argument('--train', type=str)
    #parser.add_argument('--val', type=str)
    parser.add_argument('--img_type', type=str, default='rgb')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pred', type=bool, default=False)
    parser.add_argument('--in_steps', type=int, default=10)
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()
    print(args)

    device = args.device
    torch.cuda.set_device(device)

    def npy_loader(path):
        sample = torch.from_numpy(np.load(path))
        return sample

    # Load rnn model
    model_rnn = RNN(args.input_size, args.hidden_size, args.n_lstm_layers, args.dropout) if args.pred == True else RNN(args.input_size, args.hidden_size, args.n_lstm_layers, args.dropout, batch_first=True)
    if args.ckpt is not None:
        model_rnn.load_state_dict(torch.load(args.ckpt))
        print('Resuming training.')
    model_rnn.train()
    model_rnn.to(device)

    # Set optimizer and learning rate scheduler
    optimizer = optim.Adam(model_rnn.parameters(), lr=args.lr, amsgrad=True)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=1e-2, cycle_momentum=False)
    print(model_rnn)

    # Prepare data loaders
    path_train = f'extracted_codes/{args.img_type}/train/{args.img_size}/'
    dataset_train = datasets.DatasetFolder(
        root=path_train,
        loader=npy_loader,
        extensions='.npy',
    )
    path_val = f'extracted_codes/{args.img_type}/val/{args.img_size}/'
    dataset_val = datasets.DatasetFolder(
        root=path_val,
        loader=npy_loader,
        extensions='.npy',
    )
    loader_train = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, num_workers=4)
    loader_val = DataLoader(dataset_val, batch_size=args.batch, shuffle=True, num_workers=4)

    if args.pred is True:
        model_vqvae = util.load_model(args.img_type, args.device, args.vqvae, args.img_size)
        model_vqvae.eval()
        path_test = f'/shared/users/patriri/carla_data/{args.img_type}/rnn/val'
        dataset_test = datasets.DatasetFolder(
            root=path_test,
            loader=npy_loader,
            extensions='.npy',
        )
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

        predict(loader_test, model_rnn, model_vqvae, args)

    else:

        # Training procedure
        curr_best = 999999
        best_epoch = 0
        pbar = tqdm(range(args.epoch))
        for i in pbar:
            train_loss, train_acc, val_loss, val_acc = train(i, loader_train, loader_val, model_rnn, optimizer, scheduler, args)

            torch.save(model_rnn.state_dict(), f'checkpoint/{args.img_type}/29jan/rnn_{args.img_size}x{args.img_size}.pt')
            # if val_loss < curr_best:
            #     torch.save(model_rnn.state_dict(), f'checkpoint/{args.img_type}/29jan/rnn_{args.img_size}x{args.img_size}.pt')
            #     curr_best = val_loss
            #     best_epoch = i

            lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f'epoch: {i+1}; train loss: {train_loss:.2f}; train acc: {train_acc:.2f}; val loss: {val_loss:.2f}; val acc: {val_acc:.2f}; lr: {lr:.5f}; last improvement: epoch {best_epoch+1}')
