from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin
from util import *

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.80, "Fraction of units randomly *NOT* dropped on non-recurrent connections.")
tf.app.flags.DEFINE_float("mu", 0.000, "proportion of loss to enforce st < end")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 500, "The output size of your model.")
tf.app.flags.DEFINE_integer("num_kernels", 8, "The number of kernels to use for cnn.")
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
tf.app.flags.DEFINE_float("max_grad_norm", 5., "max grad to clip to")
tf.app.flags.DEFINE_float("exp_reduce", 5.0, "fraction to reduce lr by per epoch")
tf.app.flags.DEFINE_float("reduce_every", 4, "reduce every x epochs")

# old version
# tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
# tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
# tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
# tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

# ALWAYS KEEP THIS
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")


FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    raw_context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                raw_context_data.append(' '.join(context_tokens))
                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, raw_context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, raw_context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, raw_context_data, question_data, question_uuid_data


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}

    p, raw_p, q, uuid_data = dataset
    max_len_p = min(max(map(len, p)), FLAGS.output_size)
    max_len_q = max(map(len, q))

    # p, mask_p, q, mask_q, _ = \
    #         preprocess_data((p, q),'dev', max_len_p, max_len_q)

    # get the guesses
    #all_samples = zip(p, mask_p, q, mask_q)
    num_iters = len(p)/int(FLAGS.batch_size) + 1
    i_st = 0
    all_guess_st, all_guess_end = [], []
    print("{} samples, processing in {} steps...".format(
        len(p), num_iters))
    for i in range(int(num_iters)):
        if i_st >= len(p): continue
        i_end = (i+1) * FLAGS.batch_size
        if i_end >= len(p):
            p_samples = p[i_st:]
            q_samples = q[i_st:]
        else:
            p_samples = p[i_st:i_end]
            q_samples = q[i_st:i_end]
        samples_set = \
                preprocess_data((p_samples, q_samples),
                        'dev({},{})'.format(i_st, i_end),
                        max_len_p, max_len_q)
        guess_st, guess_end = model.answer(sess, samples_set[:-1])
        all_guess_st += guess_st
        all_guess_end += guess_end
        i_st = i_end # prev
        # # TODO: REMOVE!!!
        #if i == 10: break


    # record wrt uuid
    easy_process = zip(uuid_data, raw_p,
            all_guess_st, all_guess_end)
    for uuid, text, st, end in easy_process:
         answers[uuid] = get_substring(text, st, end)
    #print("answers", answers)

    return answers


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


def main(_):
    global FLAGS
    print("FLAGS:", vars(FLAGS))
    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = np.load(embed_path)#, glove=glove)
    glove = embeddings['glove'] # np array

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    # print(vars(FLAGS))
    # with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
    #     json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, raw_context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)

    # preprocess data by truncating
    p = map(lambda line: map(int, (line.strip()).split(' ')),
            context_data)
    q = map(lambda line: map(int, (line.strip()).split(' ')),
            question_data)
    raw_context_data = map(lambda line: (line.strip()).split(' '),
            raw_context_data)
    max_len_p = min(max(map(len, p)), FLAGS.output_size)
    max_len_q = max(map(len, q))

    dataset = (p, raw_context_data, q, question_uuid_data)
    #dataset = (context_data, raw_context_data, question_data, question_uuid_data)
    train_p, raw_train_p, train_q, train_ans = \
            load_dataset("train", FLAGS.data_dir)
    train_padded_p, train_mask_p, train_padded_q, train_mask_q, train_ans = \
            preprocess_data((train_p, train_q, train_ans), "train",
                max_len_p, max_len_q)
    train_dataset = zip(train_padded_p, train_mask_p,
    	            train_padded_q, train_mask_q, train_ans)


    # Reload flags 
    print("loaded flags", vars(FLAGS))

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size,
            flags=FLAGS,
            max_len_p=max_len_p, max_len_q=max_len_q)
    decoder = Decoder(output_size=FLAGS.output_size,flags=FLAGS)

    qa = QASystem(encoder, decoder, glove, max_len_p, max_len_q, FLAGS)
    # create saver
    qa.saver = tf.train.Saver()

    # train dir
    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        initialize_model(sess, qa, train_dir)
        qa.raw_train = raw_train_p
        f1, em = qa.evaluate_answer(sess, train_dataset)
        logging.info("train total f1 {}, em {}".format(f1, em))

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
