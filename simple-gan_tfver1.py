import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.datasets import mnist # ���� mnist�����ͼ� �ҷ����� ��Ű���� keras�� �ٲ�����ϴ�.

# Eager execution�� ��Ȱ��ȭ�Ͽ� TensorFlow 1.x ����� ���
tf.compat.v1.disable_eager_execution()

# Xavier �ʱ�ȭ�� ���� �Լ� ����
def xavier_init(size):
    input_dim = size[0]
    xavier_variance = 1. / tf.sqrt(input_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_variance)

# plot �Լ�
def plot(samples):
    fig = plt.figure(figsize=(4, 4))  # 1. �׸��� ũ�⸦ �����Ͽ� ���ο� figure ��ü ����
    gs = gridspec.GridSpec(4, 4)  # 2. 4x4 ����(grid) ����
    gs.update(wspace=0.05, hspace=0.05)  # 3. ���� ���� ���� ����

    for i, sample in enumerate(samples):  # 4. ���� �̹����� ��ȸ�ϸ鼭 �� ���ڿ� ��ġ
        ax = plt.subplot(gs[i])  # 5. ���� ��ġ�� subplot ����
        plt.axis('off')  # 6. ���� ��
        ax.set_xticklabels([])  # 7. x�� ���̺��� ���
        ax.set_yticklabels([])  # 8. y�� ���̺��� ���
        ax.set_aspect('equal')  # 9. ���� ������ �����ϰ� ����
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')  # 10. �̹����� 28x28 ũ��� �����Ͽ� ������� ǥ��

    return fig  # 11. ������ figure ��ü ��ȯ


# �������� ���� ���� �õ� ����
tf.random.set_seed(100)

# GPU ��� ���� ���� Ȯ��
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# GPU �޸� ��� ����
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# ������ ��Ʈ��ũ ����
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

# �Ǻ��� ��Ʈ��ũ ����
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

# Placeholder �� ���� ����
Z = tf.compat.v1.placeholder(tf.float32, shape=[None, 100], name='Z')
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='X')

# ���� �ʱ�ȭ
G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros([128]), name='G_b1')
G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros([784]), name='G_b2')
theta_G = [G_W1, G_W2, G_b1, G_b2]

D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros([128]), name='D_b1')
D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros([1]), name='D_b2')
theta_D = [D_W1, D_W2, D_b1, D_b2]

# ��Ʈ��ũ�� �ս� �Լ� ����
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# ���̺� �������� ����Ͽ� �Ʒ� ����ȭ
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real) * 0.9))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# Gradient clipping ����
D_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
G_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

D_grads_and_vars = D_optimizer.compute_gradients(D_loss, var_list=theta_D)
D_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in D_grads_and_vars if grad is not None]
D_solver = D_optimizer.apply_gradients(D_grads_and_vars)

G_grads_and_vars = G_optimizer.compute_gradients(G_loss, var_list=theta_G)
G_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in G_grads_and_vars if grad is not None]
G_solver = G_optimizer.apply_gradients(G_grads_and_vars)

# ���� ���� ���ø� �Լ� ����
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# ������ �ε�
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.
train_images = train_images.reshape(-1, 784)

# ���ǰ� ���� �ʱ�ȭ
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# �α� ����
D_losses = []
G_losses = []
sampled_images = {}
batch_size = 128
Z_dim = 100

# �Ʒ� ����
for itr in range(60001):
    if itr % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        if itr in [0, 10000, 60000]:
            sampled_images[itr] = samples
            fig = plot(samples)
            plt.savefig('sample_{}.png'.format(itr), bbox_inches='tight')
            plt.close(fig)
            print(f"Saved sample images at iteration {itr}")

    indices = np.random.randint(0, train_images.shape[0], batch_size)
    X_mb = train_images[indices]
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

    if itr % 1000 == 0:
        D_losses.append(D_loss_curr)
        G_losses.append(G_loss_curr)
        print(f'Iter: {itr}, D loss: {D_loss_curr:.4f}, G loss: {G_loss_curr:.4f}')

# �ս� �Լ� �׷��� �׸���
fig, ax = plt.subplots()
ax.plot(D_losses, label='Discriminator Loss')
ax.plot(G_losses, label='Generator Loss')
ax.set_xlabel('Iterations (thousands)')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig('loss_plot.png')
plt.show()

# ���� �̹��� ����
for itr, samples in sampled_images.items():
    fig = plot(samples)
    plt.savefig('sample_{}.png'.format(itr), bbox_inches='tight')
    plt.close(fig)
