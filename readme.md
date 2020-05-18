A VQ-VAE in conjunction with an RNN (LSTM) to predict future video

1. Train VQ-VAE with RGB or semantic segmentation
2. Extract latent codes
3. Train RNN with RGB or semantic segmentation
4. Predict video with RNN

-------------------------------------------------------------------------------

Not to be confused with:
'Deep Neuroevolution of Recurrent and Discrete World Models' (Risi et al. 2019)
https://arxiv.org/abs/1906.08857

VQ-VAE implementation inspired by rosinality
https://github.com/rosinality/vq-vae-2-pytorch

