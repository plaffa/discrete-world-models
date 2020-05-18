######################################################################
# A number of useful pre/post-processing functions, visualizalion    #
# tools, and more.                                                   #
#                                                                    #
# @author: Patrick Ribu Gorton                                       #
######################################################################

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from vqvae import VQVAE_rgb, VQVAE_rgb2, VQVAE_seg, VQVAE_seg2, VQVAE_seg_256, VQVAE_256, VQVAE_128, VQVAE_GAN


def resize_img(img, type='rgb', normalize=True, crop=False, crop_prob=0.5, flip=False, flip_prob=0.5, size=64):
    """
    Resizes a batch of input images. Useful when torchvision does not
    support certain datatypes or tensor arrangementsself.
    If crop=True, elements in a batch is cropped with probability = crop_prob
    If flip=True, random elements in the batch are flipped with probability = flip_prob

    The image types can be either
    - RGB: returns shape (batch_size, 3, img_height, img_width)
    or
    - Segmentation map: returns shape (batch_size, 1, img_height, img_width)
    """

    if crop == True and np.random.rand() > crop_prob:
        offset = 5 # pixels
        h, w = img.shape[1], img.shape[2]
        y1, x1 = np.random.randint(offset, size=img.shape[0]), np.random.randint(offset, size=img.shape[0])
        y2, x2 = h-(offset-y1), w-(offset-x1)
        # This should probably be improved with some clever indexing..
        cropped_img = torch.zeros(img.shape[0], img.shape[1]-offset, img.shape[2]-offset, img.shape[3])
        for i in range(img.shape[0]):
            cropped_img[i] = img[i,int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
        img = cropped_img

    if type == 'seg':
        resized = F.interpolate(img[:,:,:,0].unsqueeze(0).float(), size=size, mode='nearest')
        resized = resized.reshape(-1,1,size,size) # (B,H,W,C) -> (B,C,H,W)

    elif type == 'rgb':
        resized = torch.zeros(img.shape[0],3,size,size)
        bgr = [2,1,0] # BGR -> RGB
        for i in range(3):
            tmp = F.interpolate(img[:,:,:,bgr[i]].unsqueeze(0).float(), size=size, mode='nearest').reshape(-1,1,size,size)
            resized[:,i] = tmp[:,0]
        if normalize is True:
            resized = resized/255

    if flip != False:
        idx_h = np.where(np.random.rand(resized.shape[0]) > flip_prob)
        idx_v = np.where(np.random.rand(resized.shape[0]) > flip_prob)

        if flip is 'horizontal':
            resized[idx_h] = flip_img(resized[idx_h], orientation='horizontal')

        elif flip is 'vertical':
            resized[idx_v] = flip_img(resized[idx_v], orientation='vertical')

        elif flip is 'random':
            resized[idx_h] = flip_img(resized[idx_h], orientation='horizontal')
            resized[idx_v] = flip_img(resized[idx_v], orientation='vertical')

        else:
            raise ValueError(f'{flip} is not a valid flip method.')

    return resized

def flip_img(img, orientation='horizontal'):
    """
    Flips a batch of images (dim=0) horizontally or vertically
    Input shape:  (batch_size, n_channels, img_height, img_width)
    Return shape: (batch_size, n_channels, img_height, img_width)
    """

    if orientation == 'horizontal':
        img = img.flip(3)

    elif orientation == 'vertical':
        img = img.transpose(2,3).flip(3).transpose(3,2)

    return img

def seg_weights(img, n_classes=13, eps=0.1, out_channel=3):
    """
    Estimates the class weights (from a batch distribution) to be
    used with a binary cross-entropy loss for segmentation images.
    Return shape: (batch_size, n_classes, img_height, img_width)
    """

    img_size = img.shape[-1]
    img = img.cpu()
    weights = torch.zeros(img.shape[0],1,img_size,img_size)

    for c in range(n_classes):

        if c == 4 or c == 10:
            weights += (torch.sum(img == c, dim=1).float() * (1 - torch.sum(img == c).float() / img.numel()) + 2.0).unsqueeze(1)
        else:
            weights += (torch.sum(img == c, dim=1).float() * (1 - torch.sum(img == c).float() / img.numel())).unsqueeze(1)

    weights = weights.repeat(1,out_channel,1,1)

    return weights

def loss_weights(img, n_classes=13, eps=0.1):
    """
    Estimates the class weights (from a batch distribution) to be
    used with a binary cross-entropy loss for segmentation images.
    Return shape: (batch_size, n_classes, img_height, img_width)
    """

    img_size = img.shape[-1]
    #weights = torch.Tensor(1,n_classes,1,1)
    weights = torch.Tensor(img.shape[0],n_classes,img_size,img_size)

    for c in range(n_classes):
        # weights[0,c,0,0] = 1 - torch.sqrt( (torch.sum(img == c).float() / img.view(-1,1,1).shape[0]) )
        # if c==4: # Pedestrians (class 4) is considered important
        #    weights[0,c,0,0]+=10.0

        #weights[0,c] = torch.sum(img == c).float() / img.view(-1,1,1).shape[0]
        weights[:,c] = (torch.sum(img == c, dim=1).float() * (1 - torch.sum(img == c).float() / img.numel()))#.unsqueeze(1)
        #weights[0,c] = 1 - weights[0,c]#/weights[0,c].max() + eps
        #if c==4: # Pedestrians (class 4) is considered important
        #    weights[0,c,0,0]+=10.0
        #weights[0,c,0,0] = (torch.sum(torch.Tensor(input == c)).float()) / input.view(-1,1,1).shape[0]
    #weights = 1 - weights/weights.max() + eps
    #weights[0,4,0,0] += 10.0
    #weights = weights.repeat(img.shape[0],1,img_size,img_size)
    return weights





def one_hot(img, device='cuda:0', n_classes=13):
    """
    One-hot encodes a batch of images.
    Input shape:  (batch_size, 1, img_height, img_width)
    Return shape: (batch_size, n_classes, img_height, img_width)
    """

    one_hot = torch.cuda.FloatTensor(img.size(0), n_classes, img.size(2), img.size(3), device=device).zero_()
    one_hot.scatter_(1, img.long(), 1)
    return one_hot


def seg_to_rgb(img):
    """
    Converts a batch of one-hot encoded segmentation
    maps to RGB images within the scale [0,1].
    Input shape:  (batch_size, n_classes, img_height, img_width)
    Return shape: (batch_size, 3, img_height, img_width)
    """

    # CARLA class labels to RGB conversion table
    table = [
          (0, 0, 0),       # 0 Unlabelled
          (70, 70, 70),    # 1 Building
          (190, 153, 153), # 2 Fence
          (250, 170, 160), # 3 Other
          (220, 20, 60),   # 4 Pedestrian
          (153, 153, 153), # 5 Pole
          (157, 234, 50),  # 6 Road line
          (128, 64, 128),  # 7 Road
          (244, 35, 232),  # 8 Sidewalk
          (107, 142, 35),  # 9 Vegetation
          (0, 0, 142),     # 10 Car
          (102, 102, 156), # 11 Wall
          (220, 220, 0)    # 12 Traffic sign
       ]
    converted = torch.zeros(img.shape[0], 3, img.shape[2], img.shape[3]).cpu()

    for b in range(img.shape[0]):
        argmax = img[b].max(0)[1].cpu()

        for c in range(img.shape[1]):
          idx = (argmax == c).cpu().int()
          converted[b,0] += idx*table[c][0] # R
          converted[b,1] += idx*table[c][1] # G
          converted[b,2] += idx*table[c][2] # B

    return converted.float()/255


def load_model(model_type, device, ckpt=None, img_size=64):
    """
    Initializes specified model, and loads weights if a checkpoint is provided
    """

    if model_type == 'rgb':
        if img_size == 128:
            model = VQVAE_128(in_channel=3, n_embed=32, embed_dim=64).to(device)

    elif model_type == 'seg':
        if img_size == 128:
            model = VQVAE_128(in_channel=13, n_embed=32 ,embed_dim=64).to(device)

    if model_type == 'rnn':
        model = RNN(args.latent_size, args.hidden_size, args.n_lstm_layers, args.dropout)

    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))

    return model

def to_gray_code(num):
    """
    Receives an 8-bit integer and converts it to its corresponding gray code value
    Input type: uint8
    Output type: uint8
    """
    return num ^ (num>>1)

def from_gray_code(num):
    """
    Receives an 8-bit gray code integer and converts it to its corresponding binary value
    Input type: uint8
    Output type: uint8
    """
    num = num ^ (num >> 8)
    num = num ^ (num >> 4)
    num = num ^ (num >> 2)
    num = num ^ (num >> 1)
    return num

def embedding_weights(input, batch_size=64, num_classes=32, eps=0.001):
    """
    Counts the occurrence of the different embeddings and creates loss weights
    Return shape: (batch_size, n_classes)
    """

    weights = torch.Tensor(num_classes)

    for c in range(num_classes):
        #weights[0,c] = 1 - torch.sqrt( (torch.sum(input == c).float() / input.view(-1,1,1).shape[0]) )
        weights[c] = (torch.sum(input == c).float() / input.view(-1,1,1).shape[0])
        #weights[0,c] = torch.sum(input == c).float()

    weights = 1 - weights/weights.max() + eps

    #weights = weights.repeat(batch_size,1)
    return weights


def embedding_movement(input, eps=0.1):
    """
    Counts the variance in usage of the different embeddings and creates loss weights
    Return shape: (batch_size, 16)
    """

    weights = torch.Tensor(1,64) # 16

    for i in range(64): # 16
        weights[0,i] = torch.sqrt(input[:,i].float().std())

    weights += eps
    weights = weights.repeat(input.shape[0],1)
    return weights








def plot_grad_flow(named_parameters):
    """
    Function written by Roshan Rane (https://discuss.pytorch.org/u/roshanrane/)

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
