import numpy as np
import tensorflow as tf


def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    inputs_real = tf.placeholder(tf.float32,
                                 shape=(None, image_width, image_height, image_channels),
                                 name='input_real')

    inputs_z = tf.placeholder(tf.float32,
                              shape=(None, z_dim),
                              name='input_z')

    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

    return inputs_real, inputs_z, learning_rate


def discriminator(images,
                  reuse=False,
                  alpha=0.2,
                  keep_prob=0.5):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :param alpha

    :param keep_prob: if set will perform drop out on a few layers

    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """

    def leaky_relu(x):
        return tf.maximum(alpha * x, x)

    def drop_out(x):
        # drop out only if keep_prob is set
        return tf.nn.dropout(x, keep_prob=keep_prob) if keep_prob else x

    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 28x28x3
        x1 = tf.layers.conv2d(images, filters=64, kernel_size=5, strides=2, padding='same')
        relu1 = drop_out(leaky_relu(x1))

        #  we will reduce image dimensions by half and double the depth between each layers.

        # 14x14x64
        x2 = tf.layers.conv2d(relu1, filters=128, kernel_size=5, strides=2, padding='same')
        relu2 = drop_out(leaky_relu(tf.layers.batch_normalization(x2, training=True)))

        # 7x7x128
        x3 = tf.layers.conv2d(relu2, filters=256, kernel_size=5, strides=2, padding='same')
        relu3 = drop_out(leaky_relu(tf.layers.batch_normalization(x3, training=True)))

        # 4x4x256
        # Flatten it
        flat = tf.reshape(relu3, (-1, 4 * 4 * 256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits


def generator(z,
              out_channel_dim,
              is_train=True,
              alpha=0.2,
              keep_prob=0.5):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :param alpha:
    :param keep_prob: if set will perform drop out on a few layers

    :return: The tensor output of the generator
    """

    def leaky_relu(x):
        return tf.maximum(alpha * x, x)

    def drop_out(x):
        # drop out only if keep_prob is set
        return tf.nn.dropout(x, keep_prob=keep_prob) if keep_prob else x

    with tf.variable_scope('generator', reuse=not is_train):
        # First fully connected layer

        #  to 2x2x512
        x1 = tf.layers.dense(z, 2 * 2 * 512)
        x1 = tf.reshape(x1, (-1, 2, 2, 512))
        x1 = drop_out(leaky_relu(tf.layers.batch_normalization(x1, training=is_train)))

        # to 7x7x256
        x2 = tf.layers.conv2d_transpose(x1, filters=256, kernel_size=5, strides=2, padding='valid')
        x2 = drop_out(leaky_relu(tf.layers.batch_normalization(x2, training=is_train)))

        # to 14x14x128
        x3 = tf.layers.conv2d_transpose(x2, filters=128, kernel_size=5, strides=2, padding='same')
        x3 = drop_out(leaky_relu(tf.layers.batch_normalization(x3, training=is_train)))

        # to 28x28 x out_channel_dim
        logits = tf.layers.conv2d_transpose(x3, filters=out_channel_dim, kernel_size=5, strides=2, padding='same')

        out = tf.tanh(logits)

        return out


def model_loss(input_real, input_z, out_channel_dim, alpha=0.2):
    """
    Get the loss for the discriminator and generator
    :param alpha:
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    g_model = generator(input_z, out_channel_dim, alpha=alpha)
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    from matplotlib import pyplot
    import helper

    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()


def train(epoch_count,
          batch_size,
          z_dim,
          learning_rate,
          beta1, get_batches,
          data_shape,
          data_image_mode,
          print_every=10,
          show_every=100,
          show_images=25):
    """
    Train the GAN
    :param show_images:
    :param show_every:
    :param print_every:
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    image_width, image_height, image_channels = data_shape[1:]

    input_real, input_z, input_lr = model_inputs(image_width=image_width,
                                                 image_height=image_height,
                                                 image_channels=image_channels,
                                                 z_dim=z_dim)

    d_loss, g_loss = model_loss(input_real=input_real,
                                input_z=input_z,
                                out_channel_dim=image_channels)

    d_opt, g_opt = model_opt(d_loss=d_loss,
                             g_loss=g_loss,
                             learning_rate=learning_rate,
                             beta1=beta1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for steps, batch_images in enumerate(get_batches(batch_size), start=1):
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                # Scale to  (-1, 1) from (-0.5,  0.5)
                batch_images = batch_images * 2.0

                # Run optimizers
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_z: batch_z, input_real: batch_images})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = d_loss.eval({input_z: batch_z,
                                                input_real: batch_images,
                                                input_lr:learning_rate})

                    train_loss_g = g_loss.eval({input_z: batch_z,
                                                input_lr: learning_rate})

                    print("Epoch {}/{}...".format(epoch_i + 1, epoch_count),
                          "Batch: {}...".format(steps),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                if steps % show_every == 0:
                    show_generator_output(sess,
                                          n_images=show_images,
                                          input_z=input_z,
                                          out_channel_dim=image_channels,
                                          image_mode=data_image_mode)


def main():
    import problem_unittests as tests

    tests.test_model_inputs(model_inputs)
    tests.test_discriminator(discriminator, tf)
    tests.test_generator(generator, tf)
    tests.test_model_loss(model_loss)
    tests.test_model_opt(model_opt, tf)


if __name__ == '__main__':
    main()
