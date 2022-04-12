import tensorflow as tf

import glob
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import time

tf.config.run_functions_eagerly(True)

EPOCHS = 300

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

train_images = np.asarray(arr)

print("train_images.shape=", train_images.shape)
train_images = train_images.reshape(train_images.shape[0], 128, 128, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 5153
BATCH_SIZE = 64

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential() # makes the model with 1 input and 1 output
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization()) # layer to normalize data, making it between -1 and 1
    model.add(layers.LeakyReLU()) # layer that takes in data, and if data is positive, output data, if negative, output zero
                                  # leaky then means if negative, output a negative value close to zero instead of zero, this is to make sure it doesn't get caught on a value because of a zero value

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization()) # then normalize and leakyReLU again
    model.add(layers.LeakyReLU())

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

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) # randomly set values to zero, exagerate non-dropped values by 1/(1-.3)
    # dropout reduces overfitting and improves generalization error for small sample sizes

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # outputs boolean classification of the image

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
#print("Decision:", decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output, discriminator_losses):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    total_loss_numpy = real_loss.numpy() + fake_loss.numpy()
    discriminator_losses.append(total_loss_numpy)
    return total_loss

def generator_loss(fake_output, generator_losses):
    gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    generator_losses.append(gen_loss.numpy())
    return gen_loss

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

seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, discriminator_losses, generator_losses):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output, generator_losses)
      disc_loss = discriminator_loss(real_output, fake_output, discriminator_losses)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

averaged_disc_losses = {}
averaged_gen_losses = {}

def train(dataset, epochs):
  print('Starting training at', time.asctime(time.localtime(time.time())))
  for epoch in range(epochs):
    start = time.time()

    discriminator_losses = []
    generator_losses = []
    for image_batch in dataset:
      train_step(image_batch, discriminator_losses, generator_losses)

    #print(discriminator_losses)
    averaged_disc_losses[epoch] = np.mean(discriminator_losses)
    averaged_gen_losses[epoch] = np.mean(generator_losses)
    discriminator_losses = []
    generator_losses = []

    # Produce images for the GIF as you go
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 20 == 0:
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
      plt.imshow(predictions[i, :, :, :] / 2 + 0.5) # Normalizing the input from [-1, 1] to [0, 1]
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.close('all')
  fig.clf()

train(train_dataset, EPOCHS)

# Plotting loss functions
plt.plot(averaged_disc_losses.keys(), averaged_disc_losses.values(), label="Disc. Loss")
plt.plot(averaged_gen_losses.keys(), averaged_gen_losses.values(), label="Gen. Loss")
plt.legend()
plt.title('Loss Functions')
plt.xlabel('Epoch No.')
plt.ylabel('Loss')
plt.savefig('loss'+ '.png')

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number

anim_file = 'catgan.gif'

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