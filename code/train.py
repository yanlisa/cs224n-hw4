from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np
from util import *

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
from datetime import datetime

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%s(levelname)s:%s(message)s', level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.85, "Fraction of units randomly *NOT* dropped on non-recurrent connections.")
tf.app.flags.DEFINE_float("mu", 0.001, "proportion of loss to enforce st < end")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 500, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("perspective_size", 50, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_integer("train_set_size", -1, "size of training set")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_integer("model_type", 3, "basic: 0, multiperspective: 1, mix: 2, cnn: 3")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_boolean("clip_gradients",True, "Clip gradients")
tf.app.flags.DEFINE_float("max_grad_norm", 10., "max grad to clip to")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        # Lisa: word: i, rev_vocab: just words themselves
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

# Lisa functions

# End Lisa functions

def main(_):
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    FLAGS.sessname = "{:%Y%m%d_%H%M%S}".format(datetime.now())
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir,
                        "log{}.txt".format(FLAGS.sessname)))
    logging.getLogger().addHandler(file_handler)

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = None
    train_p, raw_train_p, train_q, train_ans = \
            load_dataset("train", FLAGS.data_dir)
    val_p, raw_val_p, val_q, val_ans = \
            load_dataset("val", FLAGS.data_dir)

    max_len_p = max(max(map(len, train_p)), max(map(len, val_p)))
    max_len_p = FLAGS.output_size # truncate
    max_len_q = max(max(map(len, train_q)), max(map(len, val_q)))
    max_len_ans = max(map(len, train_ans)) # 2

    train_padded_p, train_mask_p, train_padded_q, train_mask_q, train_ans = \
            preprocess_data((train_p, train_q, train_ans), "train",
                max_len_p, max_len_q)
    val_padded_p, val_mask_p, val_padded_q, val_mask_q, val_ans = \
            preprocess_data((val_p, val_q, val_ans), "val",
                max_len_p, max_len_q)

    t_len = FLAGS.train_set_size
    if t_len != -1: # minibatch to check overfitting
        train_dataset = zip(train_padded_p[:t_len], train_mask_p[:t_len],
    	                train_padded_q[:t_len], train_mask_q[:t_len], train_ans[:t_len])
    else: # regular version
        train_dataset = zip(train_padded_p, train_mask_p,
    	                train_padded_q, train_mask_q, train_ans)
    FLAGS.num_iters = len(train_dataset)
    val_dataset = zip(val_padded_p, val_mask_p,
                    val_padded_q, val_mask_q, val_ans)
    raw_dataset = (raw_train_p, raw_val_p)
    dataset = (train_dataset, val_dataset, raw_dataset)
    logger.info("Sanity check on lengths: min %s, max %s" % \
            (lambda x: (min(x), max(x)))(map(len, train_padded_p)))

    logger.info("Loading glove embeddings...")
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    embeddings = np.load(embed_path)#, glove=glove)
    glove = embeddings['glove'] # np array
    logger.info("glove dims {}".format(glove.shape))

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size,
            flags=FLAGS,
            max_len_p=max_len_p, max_len_q=max_len_q)
    decoder = Decoder(output_size=FLAGS.output_size, flags=FLAGS)

    qa = QASystem(encoder, decoder, glove, max_len_p, max_len_q,
            FLAGS)
    # create saver
    qa.saver = tf.train.Saver()


    logger.info("{}".format(vars(FLAGS)))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, save_train_dir)

        #qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)
        f1, em = qa.evaluate_answer(sess, train_dataset, log=True)
        logger.info("final evaluation: F1: {}, EM: {}".format(f1, em))

if __name__ == "__main__":
    tf.app.run()
