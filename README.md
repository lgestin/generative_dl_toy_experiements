# Generative Modelling [toy experiments] - Deep Learning Techniques for Image Generation

This repo implements various deep learning techniques. The model are trained to generate MNIST images.

## 1. GANs 

Generative Adversarial Networks is a generative modelling technique designed to generate synthetic data that resembles the training data. A well trained generator can generate high quality samples. GANs have produced state of the art results in a variety of tasks and domains (images, audio, video, ...).
GANs are composed of two networks (Generator and Discriminator) that are trained adversarially until an equilibrium is reached. 

Pros:
- **High Quality output**
- **No explicit density estimation**. Unlike some other generative modelling, GANs don't need explicit density estimation which, when dealing with high dimensional spaces, can be a significant advantage.
- **Versatility**. Gans can be used in a lot of situations. It can be used to train most of the architectures, but more importantly it can be used in combination with other techniques. It is a way to compute perceptual features (via the discriminator's features) the can be used along with other approaches.
- **Sharp Output**. Usually the output of GANs is sharper than other approaches du to the impact of the discriminator (feature matching, patch discriminators, ...)
- **Variety in the generated output**. When properly trained, GANs can generate a greater variety of outputs, capturing a wide spectrum of the training data distribution. This contrasts with some other models that may suffer from overfitting to certain data modes.
- **Unsupervised Learning** capability. Training a GAN doesn't require any label.
- **Data augmentation**. Training a GAN on your data is a great way to augment your dataset

Cons:
- **Training instability**. GANs are usually hard to train. Finding the right balance between the discriminator and the generator can be a tedious task at training time. GANs can suffer from issues like mode collapse.
- **Lack of control over the output**. The only way to really control the output of a GAN is to train it with powerful conditioning, otherwise it can be hard to control.
- There is no explicit way to estimate the latent variables for a given, real sample.

In this implementation we use:
- Hinge Loss
- WeightNorm
- Conditioning is done with conditional batch normalization

## 2. VAEs

Variational Auto Encoders is another generative modelling technique. Similarly to regular auto encoders, it is usually composed of an encoder and a decoder. 

The Encoder's job is to map the input to a lower dimension space and the decoder has to reconstruct the data from that lower dim representation. 

This bottleneck is variational in the case of VAEs. That means that the encoder predicts the parameters that define a distribution. A latent vector is then sampled from this distribution before being fed to the decoder. 

VAEs usually require 2 losses to train: reconstruction loss and KL div (that ensures that the latent distribution doesn't diverge too much from a prior distribution in most of the case $\mathcal{N}(0, I)$)

Pros:
- **Stable training**. Contrary to GANs, the two losses ensure a stable and easy training.
- **Structured latent space**: VAEs enforce a structured latent space (usually Gaussian), which allows for nice properties like interpolation and sampling.
- **Theoretical Grounding**: VAEs have a solid theoretical foundation based on the variational inference framework and Bayesian principles.

Cons:
- **Quality of sample**: Usually samples are less sharp and detailed. (Reconstruction loss in output domain alongside with gaussian prior usually causes this). However the reconstruction loss can be replaced / improved by using a Discriminator network and train the whole system as a GAN.

In this implementation we used
- For the unconditional case: Prior $\mathcal{N}(0, I)$
- For the conditional case: A gaussian conditional prior. Another network is trained to predict the parameters of the prior distribution, which the encoder is asked to match as best as it can via the KL term

## 3. Diffusion models

Diffusion models is another type of Generative models. They work in two phases. The *forward* process, which gradually corrupts training data with gaussian noise, and the *reverse process* which uses a neural network to predict what noise was added at a given step (equivalent to predicting the original data). In other words, a model is trained to gradually denoise input, which was manually gradually corrupted until it can be considered gaussian noise.

Pros:
- **Training stability**: Diffusion models are also easy to train, they don't suffer from mode collapse.
- **Quality of the samples**: Diffusion models have shown to generate high-quality samples comparable to or even surpassing GANs. 

Cons:
- **Compute cost**: Because of the iterative nature of diffusion models, both training a sampling are expensive processes.

In this implementation we used:
- Linear beta schedule.
- Conditioning done with conditional batch normalization

## 4. [WIP] RCNN