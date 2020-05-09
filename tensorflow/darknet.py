import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D, \
    MaxPool2D, GlobalAveragePooling2D, Softmax
import numpy as np
import cv2
from tensorflow_core.python.keras.layers import AveragePooling2D


def parse_cfg(cfg_file):
    with open(cfg_file, "r") as file:
        lines = [line.rstrip("\n") for line in file if line != '\n' and line[0] != "#"]

    holder = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks


def darknet(cfg_file):
    blocks = parse_cfg(cfg_file)

    model_size = int(blocks[0]['width']), int(blocks[0]['height']), int(blocks[0]['channels'])

    inputs = input_image = Input(shape=model_size)
    # inputs = inputs / 255.
    # print(inputs.shape)
    # inputs = tf.image.resize(inputs, (256, 256))
    # print(inputs.shape)
    # inputs = inputs[:, :, ::-1]
    # print(inputs.shape)

    for i, block in enumerate(blocks[1:]):
        if block['type'] == 'convolutional':
            activation = block['activation']
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            strides = int(block['stride'])

            if strides > 1:
                # downsampling is performed, so we need to adjust the padding
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(filters,
                            kernel_size,
                            strides=strides,
                            padding='valid' if strides > 1 else 'same',
                            name='conv_' + str(i),
                            use_bias=False if "batch_normalize" in block else True)(inputs)
            if "batch_normalize" in block:
                inputs = BatchNormalization(name="batch_normalize_" + str(i))(inputs)
            if activation == 'leaky':
                inputs = LeakyReLU(alpha=0.1, name="leaky_" + str(i))(inputs)
        elif block['type'] == 'maxpool':
            size = int(block['size'])
            stride = int(block['stride'])
            inputs = MaxPool2D(pool_size=(size, size), strides=(stride, stride), padding='same',
                               name='maxpool_' + str(i))(inputs)
        elif block['type'] == 'avgpool':
            inputs = GlobalAveragePooling2D(name='avgpool_' + str(i))(inputs)
        elif block['type'] == 'softmax':
            inputs = Softmax(name='softmax_' + str(i))(inputs)
            pass
    outputs = inputs

    model = Model(input_image, outputs)
    model.summary()
    return model


def load_weights(model, cfg_file, weight_file):
    # Open the weights file
    fp = open(weight_file, 'rb')

    # The first 5 in values are header information
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number
    # 4,5. Images seen by the network (during training)
    header = np.fromfile(fp, dtype=np.int32, count=5)

    blocks = parse_cfg(cfg_file)

    for i, block in enumerate(blocks[1:]):
        if block['type'] == 'convolutional':
            conv_layer = model.get_layer('conv_' + str(i))
            print(f"Layer: {i + 1} {conv_layer}")

            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if "batch_normalize" in block:
                norm_layer = model.get_layer('batch_normalize_' + str(i))
                print(f"Layer: {i + 1} {norm_layer}")
                size = np.prod(norm_layer.get_weights()[0].shape)

                bn_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)
                # tf [gama, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))
                bn_weights = bn_weights[[1, 0, 2, 3]]
            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))

            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
            if "batch_normalize" in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

    assert len(fp.read()) == 0, 'failed to read all data'
    fp.close()
    return model


if __name__ == '__main__':
    config_file_id = "/media/data_it/projects/computer_vision/OCR/national_id/api/core/classify/model/id.cfg"
    weight_file_id = "/media/data_it/projects/computer_vision/OCR/national_id/api/core/classify/model/id_126.weights"
    config_file_doc = "/media/data_it/projects/computer_vision/OCR/national_id/api/core/classify/model/darknet19.cfg"
    weight_file_doc = "/media/data_it/projects/computer_vision/OCR/national_id/api/core/classify/model/darknet19_4000.weights"
    model = darknet(config_file_id)
    model = load_weights(model, config_file_id, weight_file_id)

    img = cv2.imread("/home/thiennt/Desktop/id_images/id_cropped_test1.png")
    # img = cv2.resize(img, (256, 256)) / 255.0
    # cv2.imshow("img", img)
    # inputs_image = np.array([img])
    inputs_image = cv2.dnn.blobFromImage(img, 1./255., (256, 256), mean=[0, 0, 0], swapRB=1, crop=False)
    inputs_image = np.transpose(inputs_image, (0, 2, 3, 1))
    result = model.predict(inputs_image)
    print(result)
    cv2.waitKey(0)

    tf.saved_model.save(model, export_dir="/media/data_it/thiennt/computer_vision/model_serving/models/id_ocr/classifier/flip/2")


