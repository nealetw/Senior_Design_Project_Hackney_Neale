import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import time

EPOCHS = 100

builder = tfds.ImageFolder("./catimages/")
ds = builder.as_dataset(
    split='train',
    shuffle_files=True
)
tfds.show_examples(ds, builder.info)


ds_numpy = tfds.as_numpy(ds)

arr = []
for x in ds_numpy:
  arr.append(list(x.values())[0])

#arr = tf.image.rgb_to_grayscale(arr)
train_images = np.asarray(arr)

#train_images = tf.data.Dataset.from_tensor_slices((ds, builder.info))

#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
#(train_images, train_labels) = (ds_numpy, builder.info)
print("train_images.shape=", train_images.shape)
train_images = train_images.reshape(train_images.shape[0], 128, 128, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 5153
BATCH_SIZE = 64

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential() # makes the model with 1 input and 1 output
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,))) # makes a dense layer with 100 inputs, in 1 dimension, and 7*7*256 outputs
    model.add(layers.BatchNormalization()) # layer to normalize data, making it between -1 and 1
    model.add(layers.LeakyReLU()) # layer that takes in data, and if data is positive, output data, if negative, output zero
                                  # leaky then means if negative, output a negative value close to zero instead of zero, this is to make sure it doesn't get caught on a value because of a zero value

    model.add(layers.Reshape((8, 8, 256))) # changes the 7*7*256 outputs to be in 3d space, 7x7x256
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    # Sets a 5x5 window to convolute the outputs of the previous layer, then strides by 1, to output with dimension of 7x7x128 ?
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization()) # then normalize and leakyReLU again
    model.add(layers.LeakyReLU())

    # kernal size of 5,5 with stride of 2,2 may not be optimal
    # its good to have a a kernal size divisible by the stride size https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba
    model.add(layers.Conv2DTranspose(64, (7, 7), strides=(4, 4), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(16, (4, 4), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # activation is tanh to not squash out the negatives thats we've been keeping through leakyReLU
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    print("model.output_shape=", model.output_shape)
    assert model.output_shape == (None, 128, 128, 3)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='Blues')

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]))
    # 28x28x1 is image size, no color. Convoluting to size of 64
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) # randomly set values to zero, exagerate non-dropped values by 1/(1-.3)
    # dropout reduces overfitting and improves generalization error for small sample sizes

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # outputs bool classification of the image

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  print('Starting training at', time.asctime(time.localtime(time.time())))
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))
  
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}_gray.png'.format(epoch))

train(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)