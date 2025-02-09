import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D


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


def yolov3_net(cfg_file, num_classes):
    """
    Build model yolo from config file
    :param cfg_file:
    :param num_classes:
    :return:
    """
    blocks = parse_cfg(cfg_file)

    model_size = int(blocks[0]['width']), int(blocks[0]['height']), int(blocks[0]['channels'])

    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0

    inputs = input_image = Input(shape=model_size)
    inputs = inputs / 255.0
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

        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            inputs = UpSampling2D(stride)(inputs)

        elif block['type'] == 'route':
            block['layers'] = block['layers'].split(',')
            start = int(block['layers'][0])

            if len(block['layers']) > 1:
                end = int(block['layers'][1]) - i
                filters = output_filters[i + start] + output_filters[end]  # Index negatif :end - index
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        elif block['type'] == 'shortcut':
            from_ = int(block['from'])
            inputs = outputs[i - 1] + outputs[i + from_]

        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = block['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)

            out_shape = inputs.get_shape().as_list()

            inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5: 5 + num_classes]

            # Refile bounding boxes
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)

            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)

            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])

            strides = (input_image.get_shape().as_list()[1] // out_shape[1],
                       input_image.get_shape().as_list()[1] // out_shape[2])

            box_centers = (box_centers + cxy) * strides

            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
                scale += 1
            else:
                out_pred = prediction
                scale = 1
        outputs[i] = inputs
        output_filters.append(filters)
    model = Model(input_image, out_pred)
    # model.summary()
    print(model.outputs)
    return model
