from keras.datasets.cifar10 import load_data
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, BatchNormalization, LeakyReLU, Dropout, Reshape
from keras.optimizers import Adam
from keras.models import load_model, Sequential
from keras.initializers import RandomNormal
import numpy as np

from matplotlib import pyplot

RNG = np.random.default_rng()

def define_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()

    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(0.2))

    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def load_real_samples():
    (X, _), (_, _) = load_data()
    X = X.astype('float32')
    X = X / 127.5
    X = X - 1
    return X

def generate_real_samples(dataset, n_samples):
    ix = RNG.integers(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))

    return X, y


def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 4 * 4 * 256
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4, 4, 256)))

    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(3, (3, 3), padding='same', activation='tanh'))
    return model

def generate_latent_points(latent_dim, n_samples):
    x_input = RNG.standard_normal(latent_dim * n_samples)
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

def show_images(samples, n):
    for i in range(n*n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis('off')
        pyplot.imshow(samples[i])
    pyplot.show()

def define_gan(g_model, d_model):
    d_model.trainable = False

    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = dataset.shape[0] // n_batch
    half_batch = n_batch // 2

    for epo in range(n_epochs):
        for b in range(bat_per_epo):
            x_real, y_real = generate_real_samples(dataset, half_batch)
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            real_loss, _ = d_model.train_on_batch(x_real, y_real)
            fake_loss, _ = d_model.train_on_batch(x_fake, y_fake)

            x_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))

            g_loss = gan_model.train_on_batch(x_gan, y_gan)

            print(f'>{epo+1}: {b+1}/{bat_per_epo}, r_loss: {real_loss:.3f}, f_loss: {fake_loss:.3f}, g_loss: {g_loss:.3f}')
        
        if (epo+1) % 10 == 0:
            summarize_performance(epo, g_model, d_model, dataset, latent_dim)

def save_plot(examples, epoch, n=7):
    # scale from [-1, 1] to [0, 1]
    examples = (examples + 1) / 2.0

    for i in range(n*n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis('off')
        pyplot.imshow(examples[i])
    
    filename = f'generated_plot_{epoch+1}.png'
    pyplot.savefig(filename)
    pyplot.close()

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    x_real, y_real = generate_real_samples(dataset, n_samples)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    print(f'>Accuracy real: {acc_real*100:.0f}%, fake: {acc_fake*100:.0f}%')

    save_plot(x_fake, epoch)

    filename = f'generator_model_{epoch+1}.h5'
    g_model.save(filename)


if __name__ == '__main__':
    latent_dim = 100
    dataset = load_real_samples()
    d_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    train(g_model, d_model, gan_model, dataset, latent_dim)