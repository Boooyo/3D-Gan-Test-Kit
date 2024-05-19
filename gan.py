import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Self-Attention 레이어 정의
class SelfAttention(layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query_conv = layers.Conv3D(filters=channels // 8, kernel_size=1)
        self.key_conv = layers.Conv3D(filters=channels // 8, kernel_size=1)
        self.value_conv = layers.Conv3D(filters=channels, kernel_size=1)
        self.gamma = tf.Variable(0.0)

    def call(self, x):
        batch_size, depth, height, width, channels = x.shape

        queries = self.query_conv(x)
        keys = self.key_conv(x)
        values = self.value_conv(x)

        queries_flat = tf.reshape(queries, (batch_size, -1, self.channels // 8))
        keys_flat = tf.reshape(keys, (batch_size, -1, self.channels // 8))
        values_flat = tf.reshape(values, (batch_size, -1, self.channels))

        attention_scores = tf.matmul(queries_flat, keys_flat, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        attention_output = tf.matmul(attention_scores, values_flat)
        attention_output = tf.reshape(attention_output, (batch_size, depth, height, width, channels))

        return self.gamma * attention_output + x

# 하이퍼파라미터 정의
latent_dim = 100
batch_size = 32
epochs = 50
learning_rate = 1e-4
beta_1 = 0.5
beta_2 = 0.9
checkpoint_dir = './training_checkpoints'
log_dir = './logs'

# Generator 모델 정의
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 4 * 4 * 4, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 4, 128)))
    assert model.output_shape == (None, 4, 4, 4, 128)

    model.add(layers.Conv3DTranspose(128, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(SelfAttention(128))  # Self-Attention 추가

    model.add(layers.Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(SelfAttention(64))  # Self-Attention 추가

    model.add(layers.Conv3DTranspose(1, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 32, 1)

    return model

# Discriminator 모델 정의
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same', input_shape=[32, 32, 32, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(SelfAttention(64))  # Self-Attention 추가

    model.add(layers.Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(SelfAttention(128))  # Self-Attention 추가

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# WGAN-GP 손실 함수 정의
def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    epsilon = tf.random.normal([batch_size, 1, 1, 1, 1])
    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        pred = discriminator(interpolated_images)

    gradients = tape.gradient(pred, [interpolated_images])[0]
    gradients_sqr = tf.square(gradients)
    gradient_penalty = tf.reduce_sum(gradients_sqr, axis=[1, 2, 3, 4])
    return tf.reduce_mean(gradient_penalty)

def discriminator_loss(real_output, fake_output, gp):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2)

# 모델 생성
generator = build_generator()
discriminator = build_discriminator()

# 체크포인트 생성
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 텐서보드 설정
summary_writer = tf.summary.create_file_writer(log_dir)

# 훈련 루프
@tf.function
def train_step(real_images):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gp = gradient_penalty(discriminator, real_images, generated_images)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, gp)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        for real_images in dataset:
            gen_loss, disc_loss = train_step(real_images)

        # 로깅
        with summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

        print(f'Epoch {epoch + 1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')

        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1)
            checkpoint.save(file_prefix=checkpoint_prefix)

# 데이터셋 준비
def generate_real_samples(batch_size):
    return np.random.rand(batch_size, 32, 32, 32, 1).astype('float32')

train_dataset = tf.data.Dataset.from_tensor_slices(generate_real_samples(1000)).shuffle(1000).batch(batch_size)

# 훈련 시작
train(train_dataset, epochs)

# 결과 확인
def generate_and_save_images(model, epoch):
    noise = tf.random.normal([1, latent_dim])
    predictions = model(noise, training=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(predictions[0, :, :, :, 0] > 0.5, edgecolor='k')
    plt.title(f'Epoch {epoch}')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()
    plt.close()

    # Save the image using imageio
    image = imageio.imread(f'image_at_epoch_{epoch:04d}.png')
    imageio.imsave(f'image_epoch_{epoch:04d}.png', image)

generate_and_save_images(generator, epochs)