
# Human Face Generator using GANs

This project implements a Generative Adversarial Network (GAN) to generate realistic human faces. GANs consist of two neural networks, a Generator and a Discriminator, that are trained together to create images that become increasingly realistic over time.

![images](https://eu-images.contentstack.com/v3/assets/blt6b0f74e5591baa03/bltfd36e68ac7a0f3b2/651b29bb3671b45abcc7e4c8/Generative_AI_(2).png?width=850&auto=webp&quality=95&format=jpg&disable=upscale)

## Features

- Introduction
- Dataset images
- Installation
- Architecture
- Training Progress
- Results
- Pre-trained Model
- Contributing



## Introduction
Generative Adversarial Networks (GANs) have revolutionized the field of generative modeling. This project focuses on generating high-quality human face images using a GAN architecture. The GAN is trained on a dataset of 50,000 celebrity faces from Kaggle and learns to generate new, unseen faces that are indistinguishable from real images.




## Dataset images

![App Screenshot](https://images.newscientist.com/wp-content/uploads/2022/02/14174128/PRI_223554170.jpg?width=1003)

This graph provides insight into how the generator and discriminator losses evolved during the training process. Ideally, the generator's loss decreases while the discriminator's loss stabilizes, indicating a balanced training process.


## Installation

To get started with this project, you need to clone the repository 

```bash
  git clone https://github.com/b0808/Human-Face-Generator-using-GANs-Generative-AI.git
```


## Architecture
I trained two GANs models: one with higher convolutional layers than the other to improve accuracy

### Small Architecture

### Generator
```
model = Sequential(name='generator1')

model.add(layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM))
model.add(layers.ReLU())

model.add(layers.Reshape((8, 8, 512)))

model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())

model.add(layers.Conv2D(CHANNELS, (4, 4), padding='same', activation='tanh'))

generator1 = model
generator1.summary()
```
### Discriminator
```
model = Sequential(name='discriminator1')
input_shape = (64, 64, 3)
alpha = 0.2

model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Flatten())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(1, activation='sigmoid'))

discriminator1 = model
discriminator1.summary()
```
## Bigger Architecture
### Generator
```
 model = Sequential(name='generator2')

model.add(layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM))
model.add(layers.ReLU())

model.add(layers.Reshape((8, 8, 512)))

model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())
model.add(layers.Conv2DTranspose(256, (4, 4), strides=(1, 1), padding='same', kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())
model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())
model.add(layers.Conv2DTranspose(156, (4, 4), strides=(1,1), padding='same', kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', kernel_initializer=WEIGHT_INIT))

model.add(layers.ReLU())

model.add(layers.Conv2D(CHANNELS, (4, 4), padding='same', activation='tanh'))

generator2 = model
generator2.summary()
```
### Discriminator
```
model = Sequential(name='discriminator2')
input_shape = (64, 64, 3)
alpha = 0.2


model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))
model.add(layers.Conv2D(64, (4, 4), strides=(1, 1), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))


model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(64, (4, 4), strides=(1, 1), padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Flatten())
model.add(layers.Dropout(0.3))


model.add(layers.Dense(1, activation='sigmoid'))

discriminator2 = model
discriminator2.summary()
```


## Training Progress
Below is the graph showing the training progress of the Generator vs. Discriminator for Bigger Architecture over 50 epochs:
![App Screenshot](https://github.com/b0808/Human-Face-Generator-using-GANs-Generative-AI/blob/main/download.png)

This graph provides insight into how the generator and discriminator losses evolved during the training process. Ideally, the generator's loss decreases while the discriminator's loss stabilizes, indicating a balanced training process.
## Results
### Generated Images :

![App Screenshot](https://github.com/b0808/Human-Face-Generator-using-GANs-Generative-AI/blob/main/images.png)

## Pre-trained Model
A pre-trained Generator model is available in the repository. You can download the  [Generator.h5](https://github.com/yourusername/face-generator-gan/blob/main/generator_final.h5) file

## Contributing

We welcome contributions to improve this project! If you have any suggestions or bug reports, please open an issue or submit a pull request.
