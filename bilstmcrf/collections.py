import six
import tensorflow as tf

GRADS = "grads"
TRAIN_LOSS = "train_loss"
EVAL_LOSS = "eval_loss"


def get_dict_from_collection(name):
    key = tf.get_collection(name + "_key")
    value = tf.get_collection(name + "_value")
    return dict(zip(key, value))


def add_dict_to_collection(name, dict_):
    for k, v in six.iteritems(dict_):
        tf.add_to_collection(name + "_key", k)
        tf.add_to_collection(name + "_value", v)
