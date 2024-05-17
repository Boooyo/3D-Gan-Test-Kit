import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Generator 모델 정의
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 4 * 4 * 4, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 4, 128)))
    assert model.output_shape == (None, 4, 4, 4, 128)

    model.add(layers.Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3DTranspose(1, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 16, 16, 16, 1)

    return model

# Discriminator 모델 정의
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same', input_shape=[16, 16, 16, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# 손실 함수와 최적화기 정의
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

# 모델 생성
generator = build_generator()
discriminator = build_discriminator()

# 훈련 루프
@tf.function
def train_step(real_images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        for real_images in dataset:
            gen_loss, disc_loss = train_step(real_images)
        print(f'Epoch {epoch + 1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')

# 데이터셋 준비
def generate_real_samples(batch_size):
    # 여기서는 단순히 임의의 데이터를 생성
    return np.random.rand(batch_size, 16, 16, 16, 1).astype('float32')

batch_size = 32
epochs = 50
train_dataset = tf.data.Dataset.from_tensor_slices(generate_real_samples(1000)).shuffle(1000).batch(batch_size)

# 훈련 시작
train(train_dataset, epochs)

# 결과 확인
def generate_and_save_images(model, epoch):
    noise = tf.random.normal([1, 100])
    predictions = model(noise, training=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(predictions[0, :, :, :, 0] > 0.5, edgecolor='k')
    plt.title(f'Epoch {epoch}')
    plt.show()

generate_and_save_images(generator, epochs)
