import numpy as np
from yolov3 import yolov3_net
from yolov3 import parse_cfg
import argparse
import tensorflow as tf


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../cfg/yolov3.cfg")
    parser.add_argument("--weight", type=str, default="../weights/yolov3.weights")
    parser.add_argument("--n_classes", type=int, default=80)
    parser.add_argument("--output", type=str, default="../weights/test/")

    args = parser.parse_args()

    # weight_file = "../weights/yolov3.weights"
    # cfg_file = "../cfg/yolov3.cfg"

    # num_classes = 80

    model = yolov3_net(args.config, args.n_classes)
    load_weights(model, args.config, args.weight)

    try:
        model.save_weights(args.output)
        # tf.saved_model.save(model, args.output)
        print("The file yolov3_weights. tf has been saved successfully.")
    except IOError:
        print("Couldn't write the file yolov3_weights.tf .")


if __name__ == '__main__':
    main()
