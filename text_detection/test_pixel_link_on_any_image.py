# encoding = utf-8

import numpy as np
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing
import util
import pixel_link
from nets import pixel_link_symbol
import config

tf.flags.DEFINE_string('checkpoint_path', 'conv2_2/model.ckpt-73018',
                       'the path of pretrained model to be used. '
                       'If there are checkpoints in train_dir, this config will be ignored.')
tf.flags.DEFINE_string('dataset_dir', 'custom_image', 'The directory where the dataset files are stored.')
tf.flags.DEFINE_integer('eval_image_width', 1280, 'resized image width for inference')
tf.flags.DEFINE_integer('eval_image_height',  720, 'resized image height for inference')
tf.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')
FLAGS = tf.flags.FLAGS


def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    config.init_config(image_shape, batch_size=1, num_gpus=1, pixel_conf_threshold=0.5, link_conf_threshold=0.5)


def test():
    global_step = tf.train.get_or_create_global_step()
    with tf.name_scope('evaluation_%dx%d' % (FLAGS.eval_image_height, FLAGS.eval_image_width)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            image = tf.placeholder(dtype=tf.int32, shape=[None, None, 3])
            processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None,
                                                                                 out_shape=config.image_shape,
                                                                                 data_format=config.data_format,
                                                                                 is_training=False)
            b_image = tf.expand_dims(processed_image, axis=0)
            # build model and loss
            net = pixel_link_symbol.PixelLinkNet(b_image, is_training=False)
            masks = pixel_link.tf_decode_score_map_to_mask_in_batch(net.pixel_pos_scores, net.link_pos_scores)
            
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    # Variables to restore: moving avg. or normal weights.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore(tf.trainable_variables())
    variables_to_restore[global_step.op.name] = global_step

    saver = tf.train.Saver(var_list=variables_to_restore)
    with tf.Session() as sess:
        saver.restore(sess, util.tf.get_latest_ckpt(FLAGS.checkpoint_path))
        files = util.io.ls(FLAGS.dataset_dir)
        for image_name in files:
            file_path = util.io.join_path(FLAGS.dataset_dir, image_name)
            image_data = util.img.imread(file_path)
            link_scores, pixel_scores, mask_vals = sess.run([net.link_pos_scores, net.pixel_pos_scores, masks],
                                                            feed_dict={image: image_data})
            h, w, _ = image_data.shape

            def get_bboxes(mask):
                return pixel_link.mask_to_bboxes(mask, image_data.shape)
            
            def draw_bboxes(img, bboxes, color):
                for bbox in bboxes:
                    points = np.reshape(bbox, [4, 2])
                    cnts = util.img.points_to_contours(points)
                    util.img.draw_contours(img, contours=cnts, idx=-1, color=color, border_width=1)
            image_idx = 0
            mask = mask_vals[image_idx, ...]
            bboxes_det = get_bboxes(mask)

            draw_bboxes(image_data, bboxes_det, util.img.COLOR_RGB_RED)
            print util.sit(image_data, image_name)


def main(_):
    config_initialization()
    test()


if __name__ == '__main__':
    tf.app.run()
