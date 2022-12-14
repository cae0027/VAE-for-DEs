## Variational Autoencoders for Differential Equations

Variational Auto-Encoder (VAE) of ![Kingma and Welling](https://arxiv.org/pdf/1312.6114.pdf) is a neural networks based encoder-decoder type architecture to perform efficient inference and learning in directed probalistic models, in the presence of continuous latent variables with intratable posterior distributions and large datasets.

Amond the many variants of VAE is Conditional Variational Auto-Encoder (CVAE) of ![Sohn, Yan & Lee](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf). Conditioning on specific parts of images as attention mechanisms, CVAEs have been applied to generate new samples of specific MNIST digits. ![Gundersen et al](https://aip.scitation.org/doi/10.1063/5.0025779) used CVAE to reconstruct flow and quantify unceratinty from limited observations.

This work uses CVAE to reconstruct fine scale solutions from coarse solutions of differential equations, in the presence of rough or random parameters.

