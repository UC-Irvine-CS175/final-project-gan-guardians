from matplotlib import pyplot
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, LeakyReLU, Flatten, Reshape, Conv2DTranspose, GaussianNoise, Embedding, Dense, Concatenate, Activation
import numpy as np

RNG = np.random.default_rng()

def load_real_samples():
    images, labels = np.load('x_data.npy'), np.load('y_data.npy')
    images = images.astype('float32')
    images = (images / 127.5) - 1

    return [images, labels]

def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = RNG.integers(0, images.shape[0], n_samples)
    images, labels = images[ix], labels[ix]
    y = np.ones((n_samples, 1))

    return [images, labels], y

def generate_latent_points(latent_dim, n_samples, n_classes):
    latent = RNG.standard_normal((n_samples, latent_dim))
    labels = RNG.integers(0, n_classes, n_samples)

    return [latent, labels]

def generate_fake_samples(g_model, latent_dim, n_samples, n_classes=28):
    latent, labels = generate_latent_points(latent_dim, n_samples, n_classes)
    X = g_model.predict([latent, labels])
    y = np.zeros((n_samples, 1))
    return [X, labels], y

def define_discriminator(in_shape=(28, 28, 1), n_classes=28):
    init = RandomNormal(mean=0.0, stddev=0.02, seed=RNG.integers(0, 10000))

    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 60)(in_label)
    li = Dense(in_shape[0] * in_shape[1])(li)
    li = Reshape((in_shape[0], in_shape[1], 1))(li)

    in_image = Input(shape=(28, 28, 1))
    merge = Concatenate()([in_image, li])

    # downsample to 14x14
    x = GaussianNoise(0.1)(merge)
    x = Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer=init, padding='same', input_shape=in_shape)(merge)
    x = LeakyReLU(0.2)(x)
    # downsample to 7x7
    x = GaussianNoise(0.1)(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # flatten
    x = Flatten()(x)
    x = GaussianNoise(0.1)(x)
    # output
    output = Dense(1, activation='sigmoid')(x)
    # define model
    model = Model([in_image, in_label], output)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def define_generator(latent_dim, n_classes=28):
    init = RandomNormal(mean=0.0, stddev=0.02, seed=RNG.integers(0, 10000))

    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 60)(in_label)
    li = Dense(7*7)(li)
    li = Reshape((7,7,1))(li)

    in_lat = Input(shape=(latent_dim,))
    gen = Dense(7*7)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((7, 7, 1))(gen)

    gen = Concatenate()([gen, li])
    # upsample to 14x14
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    # output 28x28x1
    output = Conv2D(1, (7, 7), padding='same', activation='tanh')(gen)
    model = Model([in_lat, in_label], output)
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False

    gen_noise, gen_label = g_model.input
    gen_output = g_model.output
    gan_output = d_model([gen_output, gen_label])

    model = Model([gen_noise, gen_label], gan_output)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def generate_specific_samples(generator, latent_dim, n_samples, class_n):
    x_input = RNG.standard_normal((n_samples, latent_dim))
    labels = np.asarray([class_n] * n_samples)
    
    images = generator.predict([x_input, labels])
    return images

def summarize_performance(epoch, g_model, latent_dim, n_samples=140):
    # [X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
    for i in range(14):
        X = generate_specific_samples(g_model, latent_dim, 10, i)
        X = (X + 1) / 2.0
        for j in range(10):
            pyplot.subplot(14, 10, i*10 + j + 1)
            pyplot.axis('off')
            pyplot.imshow(X[j, :, :, 0], cmap='gray_r')
    
    pyplot.savefig(f'generated_plot_{epoch+1}.png')
    pyplot.close()
    g_model.save(f'g_model_{epoch+1}.h5')

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128, n_classes=28):
    bat_per_epo = dataset[0].shape[0] // n_batch
    half_batch = n_batch // 2

    for i in range(n_epochs):
        for j in range(bat_per_epo):
            [X_real, label_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss1 = d_model.train_on_batch([X_real, label_real], y_real)

            [X_fake, label_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2 = d_model.train_on_batch([X_fake, label_fake], y_fake)

            [x_gan, label_gan] = generate_latent_points(latent_dim, n_batch, n_classes)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([x_gan, label_gan], y_gan)

            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        
        if (i % 5) == 0:
            summarize_performance(i, g_model, latent_dim)

    # save the generator model
    g_model.save('cgan_generator.h5')
    


if __name__ == '__main__':
    # dataset = load_real_samples()
    # latent_dim = 100
    # d_model = define_discriminator()
    # g_model = define_generator(latent_dim)
    # gan_model = define_gan(g_model, d_model)

    # train(g_model, d_model, gan_model, dataset, latent_dim)

    g_model = load_model('g_model_32.h5')
    count = 0
    for i in range(4):
        for j in range(7):
            image = generate_specific_samples(g_model, 100, 1, count)[0]
            pyplot.subplot(4, 7, count+1)
            pyplot.axis('off')
            pyplot.imshow(image[:, :, 0], cmap='gray_r')
            count += 1

    pyplot.show()