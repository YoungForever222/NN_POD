'''

Data mining from

'''
# Common imports
import os
import sys
from data import *
from tqdm import *
import numpy as np
import numpy.random as rnd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def reset_graph(seed=42):
    """
    to make this notebook's output stable across runs
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    rnd.seed(seed)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
PROJECT_ROOT_DIR = "."
CODE_ID = "autoencoders"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CODE_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # make the minimum == 0, so the padding looks white
    w, h = images.shape[1:]
    image = np.zeros(((w + pad) * n_rows + pad, (h + pad) * n_cols + pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y * (h + pad) + pad):(y * (h + pad) + pad + h), (x *
                                                                    (w + pad) + pad):(x * (w + pad) + pad + w)] = images[y * n_cols + x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")

def run():
    '''
    Main function of the Auto-Encoder
    input:
        X_train: The input and the target Tensor.
    '''
    reset_graph()
    linear = False
    n_inputs = 400
    n_hidden = 128  # codings
    n_mid    = 100
    n_outputs = n_inputs
    learning_rate = 0.001
    datasets = DataVel().Vel
    X_train  = datasets
    X = tf.placeholder(tf.float32, shape=[64*64*3, n_inputs])
    # Net
    if linear:
        mid     = tf.layers.dense(X,n_mid)
        outputs = tf.layers.dense(mid, n_outputs)
    else:
        hidden1 = tf.layers.dense(X, n_hidden,activation=tf.tanh)
        mid     = tf.layers.dense(hidden1,n_mid)
        hidden2 = tf.layers.dense(mid,n_hidden,activation=tf.tanh)
        outputs = tf.layers.dense(hidden2, n_outputs)
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    accuracy_loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - X)))/tf.sqrt(tf.reduce_mean(tf.square(X)))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(reconstruction_loss)
    init = tf.global_variables_initializer()
    n_iterations = 10000
    codings = mid
    with tf.Session() as sess:
        init.run()
        for step in tqdm(range(n_iterations)):
            training_op.run(feed_dict={X: X_train})
            loss_train = accuracy_loss.eval(feed_dict={X: X_train})
            if step % (n_iterations/10) == 0:
                print("\r{}".format(step), "Train MSE:", loss_train) 
        codings_val = codings.eval(feed_dict={X: X_train})
        loss_train  = accuracy_loss.eval(feed_dict={X: X_train})
        print("Test MSE:",loss_train)
    print(codings_val.shape)
    result_vel = codings_val.reshape((64,64,3,n_mid))
    for i in range(n_mid):
        energy = .5*(np.sqrt(result_vel[:,:,0,i]**2 + result_vel[:,:,1,i]**2 + result_vel[:,:,2,i]**2))
        fig = plt.figure(figsize=(6.,6.))
        ax = fig.add_axes([.0, .0, 1., 1.])
        contour = ax.contour(z, y, energy, 30)
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        plt.clabel(contour, inline=1, fontsize=10)
        plt.title('Energy contours, layer = {0}'.format(i))
        fig.savefig('./middle_layer/'+figname + str(i)+'.eps', format = 'eps', bbox_inches = 'tight')




if __name__ == '__main__':
    run()
