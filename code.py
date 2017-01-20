from scipy.misc import imread, imresize, imsave, fromimage, toimage
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model

# 预先定义的参数，这里先定义防止中途出错
# 不过有一些还是没用的，原项目是当输入参数处理器
base_image_path = "content.jpg" 
style_image_paths = "style.jpg"
result_prefix = "out_"  # 输出图片的前缀
color_mask = None
image_size = 400
content_weight = 0.025
style_weight = [1]  # 只有一个style图片，简化就只有一个权值
style_scale = 1.0
total_variation_weight = 8.5e-5
num_iter = 10  # 迭代次数
content_loss_type = 0  # 0, 1, 2三种content loss
maintain_aspect_ratio = "True"
content_layer = "conv5_2"  # 选取的content层计算
init_image = "content"  # content, noise, gray三种初始
pool_type = "max"  # 池化类型
preserve_color = "False"
min_improvement = 0.0

# 图片的预处理和解析
def preprocess_image(image_path):
    img = imread(image_path)
    img = imresize(img, (400, 400)).astype('float32')

    img = img[:, :, ::-1]
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    img = img.transpose((2, 0, 1)).astype("float32")
    img = np.expand_dims(img, axis=0)
    return img

def deprocess_image(x):
    x = x.reshape((3, 400, 400))
    x = x.transpose((1, 2, 0))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 加载原始图片，分割图片和合成图片的存放空间
base_image = K.variable(preprocess_image('content.jpg'))

style_reference_images = K.variable(preprocess_image('style.jpg'))

combination_image = K.placeholder((1, 3, 400, 400))

image_tensors = [base_image, style_reference_images, combination_image]

nb_tensors = len(image_tensors)
nb_style_images = nb_tensors - 2

# 把输入转化成keras的一维输入
input_tensor = K.concatenate(image_tensors, axis=0)

shape = (nb_tensors, 3, 400, 400)

ip = Input(tensor=input_tensor, shape=shape)

# 建立vgg16模型
x = Convolution2D(64, 3, 3, activation='relu', name='conv1_1', border_mode='same')(ip)
x = Convolution2D(64, 3, 3, activation='relu', name='conv1_2', border_mode='same')(x)
x = AveragePooling2D((2, 2), strides=(2, 2))(x)

x = Convolution2D(128, 3, 3, activation='relu', name='conv2_1', border_mode='same')(x)
x = Convolution2D(128, 3, 3, activation='relu', name='conv2_2', border_mode='same')(x)
x = AveragePooling2D((2, 2), strides=(2, 2))(x)

x = Convolution2D(256, 3, 3, activation='relu', name='conv3_1', border_mode='same')(x)
x = Convolution2D(256, 3, 3, activation='relu', name='conv3_2', border_mode='same')(x)
x = Convolution2D(256, 3, 3, activation='relu', name='conv3_3', border_mode='same')(x)
x = AveragePooling2D((2, 2), strides=(2, 2))(x)

x = Convolution2D(512, 3, 3, activation='relu', name='conv4_1', border_mode='same')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv4_2', border_mode='same')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv4_3', border_mode='same')(x)
x = AveragePooling2D((2, 2), strides=(2, 2))(x)

x = Convolution2D(512, 3, 3, activation='relu', name='conv5_1', border_mode='same')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv5_2', border_mode='same')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv5_3', border_mode='same')(x)
x = AveragePooling2D((2, 2), strides=(2, 2))(x)

model = Model(ip, x)
model.load_weights("vgg16_weights_th.h5")

print('Model loaded.')

# get the symbolic outputs of each "key" layers
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
shape_dict = dict([(layer.name, layer.output_shape) for layer in model.layers])


def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3

    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = 400 * 400
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    channel_dim = 0 if K.image_dim_ordering() == "th" else -1
    channels = K.shape(base)[channel_dim]
    size = 400 * 400

    multiplier = 1 / (2. * channels ** 0.5 * size ** 0.5)
    return multiplier * K.sum(K.square(combination - base))


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :, :400 - 1, :400 - 1] - x[:, :, 1:, :400 - 1])
    b = K.square(x[:, :, :400 - 1, :400 - 1] - x[:, :, :400 - 1, 1:])
    return K.sum(K.pow(a + b, 1.25))


# combine the loss function into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict['conv5_2']  # conv5_2 or conv4_2
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[nb_tensors-1, :, :, :]

loss += content_weight * content_loss(base_image_features, combination_features)

style_masks = [None for _ in range(nb_style_images)]
channel_index = 1

feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    shape = shape_dict[layer_name]
    combination_features = layer_features[nb_tensors - 1, :, :, :]

    style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]
    sl = []
    for j in range(nb_style_images):
        sl.append(style_loss(style_reference_features[j], combination_features))

    for j in range(nb_style_images):
        loss += (style_weight[j] / len(feature_layers)) * sl[j]

loss += total_variation_weight * total_variation_loss(combination_image)

# cal grads
grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, 400, 400))
    else:
        x = x.reshape((1, 400, 400, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# 输入
x = preprocess_image("content.jpg")
pre_min_val = -1

for i in range(num_iter):
    print("Starting iteration %d of %d" % ((i+1), num_iter))
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)

    if pre_min_val == -1:
        pre_min_val = min_val

    improvement = (pre_min_val - min_val) / pre_min_val * 100

    print("Current loss value:", min_val, " Improvement : %0.3f" % improvement, "%")
    pre_min_val = min_val

    img = deprocess_image(x.copy())

    fname = result_prefix + '_at_iteration_%d.png' % (i+1)
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as ', fname)
    print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))

    if min_improvement is not 0.0:
        if improvement < min_improvement and improvement is not 0.0:
            print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." % (
                improvement, min_improvement))
            exit()
