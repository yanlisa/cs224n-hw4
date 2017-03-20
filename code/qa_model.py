from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from util import Progbar, minibatches, ConfusionMatrix, get_substring
from cnn import TDNN
from cnn.ops import highway, batch_norm, conv2d

from evaluate import exact_match_score, f1_score

logger = logging.getLogger("qa_model")
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%s(levelname)s:%s(message)s', level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim, flags=None, max_len_p=-1,max_len_q=-1):
        self.size = size # state size
        self.para_size = size # hidden state size for paragraph
        self.q_size = size # hidden state size for paragraph
        self.vocab_dim = vocab_dim # input size
        self.tf_cache = {}
        if flags is not None:
            self.config = flags
        self.max_len_p = max_len_p
        self.max_len_q = max_len_q
        # Create bidirectional lstm cell here.
        # with tf.variable_scope("encode", reuse=True):
        #     cell = tf.nn.rnn_cell.BasicLSTMCell(self.size,
        #             state_is_tuple=True)

        # initialize variables here?

    def encode(self, inputs, masks, encoder_state_input, reuse=False):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        logging.info(">>>bilstm encode")
        # initial_state_fw, initial_state_bw
        # time_major=False: [batch_size, max_time, cell_fw.output_size]
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.size,
                        state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                input_keep_prob=self.dropout_placeholder,
                output_keep_prob=self.dropout_placeholder)

        with tf.variable_scope("encode"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            (fw, bw), out_tuple = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                    inputs, sequence_length=masks, dtype=tf.float32)

        return [fw, bw], out_tuple

    def cnn_encode(self, inputs, seq_len, encoder_state_input, reuse=False):
        """Calls cnn encoding and does things
        """
        logging.info(">>>cnn encode")
        # [?, seq_len, embedding_size] (all padded)
        embedding_size = inputs.get_shape()[-1]
        logging.info("embed size {}, seq len {}".format(embedding_size, seq_len))

        kernels = [3,4,5,6]
        num_features = self.config.state_size/len(kernels) # typically 100/4=25
        use_highway = [False, False, False, False]
        input_ = inputs
        layers = []
        for idx, kernel_dim in enumerate(kernels):
            input_ = tf.expand_dims(inputs,-1)
            conv = conv2d(input_,num_features, kernel_dim, embedding_size,
                    name="kernellayer%d" % idx,
                    padding="SAME")
            # [?, seq_len, 1, num_features]
            conv_relu = tf.nn.relu(tf.squeeze(conv, [2]))
            logging.info("layer {} conv shape (kernel {}, feature {}): {}".format(
                idx, kernel_dim, num_features, conv_relu.get_shape()))

            if use_highway[idx]:
                cnn_output = tf.reshape(conv_relu, [-1, seq_len * num_features])
                with tf.variable_scope("highway%d" % idx):
                    highway_layers = 1
                    hw_output = highway(cnn_output, cnn_output.get_shape()[1],
                            highway_layers, 0)
                hw_output = tf.reshape(hw_output, [-1,seq_len, num_features])
                logging.info("conv layer {} final {}".format(idx, hw_output.get_shape()))
                conv_output = hw_output
            else:
                conv_output = conv_relu
            layers.append(conv_output)
            input_ = conv_output
        
        # # the normal one from
        # # https://arxiv.org/pdf/1508.06615.pdf
        # # word_cnn = TDNN(inputs,embedding_size)
        # # cnn_output = word_cnn.output
        # cnn_output = input_
        # logging.info("cnn {}".format(cnn_output.get_shape()))

        # # can't use batchnorm with 4d arguments
        # # use_bn = True
        # # if use_bn:
        # #     logging.info("batch norming post CNN")
        # #     bn = batch_norm()
        # #     norm_output = bn(tf.expand_dims(tf.expand_dims(cnn_output, 1), 1))
        # #     cnn_output = tf.squeeze(norm_output, [1,2])

        # # use highway, because why not
        cnn_output = tf.concat(2, layers)
        logging.info("final shape of conv {}".format(cnn_output.get_shape()))
        return cnn_output

    def mp_filter(self, p_context, q_query):
        logging.info(">>>mp_filter")
        """
        Before encoding, do some filtering on the embeddings.
        p_j = r_j * p_j
        r = cosine_sim(q, p)
        r_j = max_i (r_ij)
        """
        # expand to [batch_size, max_len_p, max_len_q, embedding_size]
        p_exp = tf.expand_dims(p_context, 2)
        p_tile = tf.tile(p_exp, [1,1,self.max_len_q,1])
        q_exp = tf.expand_dims(q_query, 1)
        q_tile = tf.tile(q_exp, [1,self.max_len_p,1,1])
        # calculate different cosine similarity...
        p_tile = tf.nn.l2_normalize(p_tile, 3)
        q_tile = tf.nn.l2_normalize(q_tile, 3)
        #logger.info("q_tile", q_tile.get_shape(), "p_tile", p_tile.get_shape())
        r_ij = self.cosine_sim(p_tile, q_tile, dim=3)
        #logger.info("cosine sim shape", r_ij.get_shape())
        # should be [batch_size, max_len_p]
        r = tf.reduce_max(r_ij,2)
        #logger.info("relevancy shape", r.get_shape())
        r = tf.expand_dims(r, 2)
        new_context = tf.mul(p_context, r)
        #logger.info("new context shape", new_context.get_shape())
        return new_context
        

    def basic_attention(self, p_context, q_query, p_size, q_size, output_size):
        logging.info(">>>basic_attention")
        # for each context word, figure out how q_query influences
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("basic_attention"):
            #tf.get_variable_scope().reuse_variables()
            W_a = tf.get_variable("W_a",
                    shape=(q_size, output_size),
                initializer=xavier_initializer)
            W_b = tf.get_variable("W_b",
                    shape=(p_size, output_size),
                initializer=xavier_initializer)
            # q * W_a
            q_reshape = tf.reshape(q_query, [-1, q_size])
            scores = tf.matmul(q_reshape, W_a)
            scores = tf.reshape(scores, [-1, self.max_len_q, output_size])
            # scores = p * (q * W_a)^T
            # [batch_size, p_len, q_len] 
            logging.info("p: {}, q*W_a: {},  output {}".format(
                p_context.get_shape(), scores.get_shape(), output_size))
            scores = tf.matmul(p_context, scores, transpose_b=True)
            scores = tf.nn.softmax(scores) # normalize to 1

            # \sum(scores * q)
            # [batch_size, p_len, q_len, 1] 
            scores_exp = tf.expand_dims(scores, 3)
            # [batch_size, 1, q_len, 2d]
            q_exp = tf.expand_dims(q_query, 1)
            # [batch_size, p_len, q_len, 2d]
            score_q = tf.mul(scores_exp, q_exp)
            # [batch_size, p_len, 2d]
            context_attn = tf.reduce_sum(score_q, 2)
            
            # [batch_size, p_len, 4d]
            p_c = tf.concat(2, [context_attn, p_context])
            # [batch_size, p_len, 2d]
            # p_c_attn * W_b
            p_c_reshape = tf.reshape(p_c, [-1, p_size])
            p_attn = tf.matmul(p_c_reshape, W_b)
            p_attn = tf.reshape(p_attn, [-1, self.max_len_p, 2*output_size])
        return p_attn

    def mix_attention(self, p_context, q_query, p_size=-1):
        if p_size == -1:
            p_size = 2*self.size
        logging.info(">>>mix_attention")
        # p_context and q_query are simple concats of fw and bw, so
        # they are 2*self.size in dimension
        # p q^t = [?,p_len,200] * [?, 200, q_len] = [?, p_len, q_len]
        a = tf.nn.softmax(tf.batch_matmul(p_context,
                                    tf.transpose(q_query, [0, 2, 1])))
        # a q = [?, p_len, 200]
        p_mix = tf.batch_matmul(a, q_query)

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("mix_attention"):
            W_1 = tf.get_variable("W_1",
                shape=(2*p_size, self.size),
                initializer=xavier_initializer)
            b_1 = tf.get_variable("b_1",
                    shape=(self.size,),
                    initializer=tf.constant_initializer(0))
            cc_rs = tf.reshape(tf.concat(2, [p_context, p_mix]),
                                [-1, 2*p_size])
            cc_rs_W = tf.matmul(cc_rs, W_1)
            p_attn = tf.reshape(cc_rs_W, [-1, self.max_len_p, self.size]) + b_1
        return p_attn

    def bidaf_attention(self, p_context, q_query, p_size=-1, q_size=-1,
            p_mask=None):
        if p_size == -1:
            p_size = 2*self.size
        if q_size == -1:
            q_size = 2*self.size
        # note if both are using cnn, then they are the same size (self.config.state_size)
        logging.info(">>>bidaf_attention")
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        # S = [?,p_len,q_len]
        # U: [?,q_len,q_dim], H: [?,p_len,p_dim]
        # S: [?,p_len,p_dim] x [p_dim, q_dim] x [?,q_len,q_dim]
        with tf.variable_scope("bidaf_flow"):
            W_pq = tf.get_variable("W_pq",
                    shape=(q_size, p_size),
                    initializer=xavier_initializer)
            q_reshape = tf.reshape(q_query, [-1, q_size])
            # [?, q_len, q_dim] --> [?, q_len, p_dim]
            wq = tf.matmul(q_reshape, W_pq)
            # [?, q_len, p_dim] --> [?, p_dim, q_len]
            wq = tf.transpose(tf.reshape(wq, [-1, self.max_len_q, p_size]), [0,2,1])
            # [?, p_len, q_len]
            S = tf.batch_matmul(p_context, wq)
            logging.info("S shape {}".format(S.get_shape()))

        # context to query
        a = tf.nn.softmax(S) # softmax across queries for given word
        # [?, p_len, q_dim]
        U_tilde = tf.batch_matmul(a, q_query)
        logging.info("U tilde {}".format(U_tilde.get_shape()))

        # query to context
        b = tf.nn.softmax(tf.reduce_max(S, 2)) # [?,p_len]
        logging.info("b {}".format(b.get_shape()))
        b = tf.expand_dims(b, 2) # [?,p_len,1]
        # [?,p_dim]
        h_tilde = tf.reduce_sum(tf.mul(p_context, b), 1)
        logging.info("h_tilde {}".format(h_tilde.get_shape()))
        # [?,p_len,p_dim]
        h_tilde = tf.expand_dims(h_tilde, 1)
        H_tilde = tf.tile(h_tilde, [1,self.max_len_p,1])
        logging.info("H_tilde {}".format(H_tilde.get_shape()))

        # [?,p_len,4*p_dim]
        G = tf.concat(2, [p_context, # h
                          U_tilde, # u_tilde
                          tf.mul(p_context, U_tilde), # h * u_tilde
                          tf.mul(p_context, H_tilde)])
        logging.info("G {}".format(G.get_shape()))

        # modeling
        with tf.variable_scope("bidaf_modeling"):
            M = self.cnn_encode(G, self.max_len_p, None)
        # cell = tf.nn.rnn_cell.BasicLSTMCell(self.size,
        #                 state_is_tuple=True)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        #         input_keep_prob=self.dropout_placeholder,
        #         output_keep_prob=self.dropout_placeholder)
        # with tf.variable_scope("bidaf_modeling"):
        #     (fw, bw), _ = tf.nn.bidirectional_dynamic_rnn(cell, cell,
        #             G, sequence_length=p_mask, dtype=tf.float32)

        # M = tf.concat(2, [fw, bw])
        logging.info("M {}".format(M.get_shape()))
        return M

    def mp_attention(self, p_context, q_query, q_out):
        logging.info(">>>mp_attention")
        # for each context word, figure out how q_query influences
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        p_fw, p_bw = p_context
        q_fw, q_bw = q_query
        # reshape and tile to begin with
        # [batch_size, max_len_p, max_len_q,state_size] 
        #p_fw_exp = tf.reshape(tf.tile(tf.expand_dims(p_fw, 
        # u = tf.reshape(tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1]), [N * M, JQ, 2 * d])
        logger.info("p_fw shape before attn {}".format(p_fw.get_shape()))
        logger.info("q_fw shape before attn {}".format(q_fw.get_shape()))
        out_q_fw, out_q_bw = q_out[0][1], q_out[1][1]
        with tf.variable_scope("mp_fullcontext"):
            W_1 = tf.get_variable("W_1", # forward
                    shape=(self.size, self.config.perspective_size),
                    initializer=xavier_initializer)
            W_2 = tf.get_variable("W_2", # backward
                    shape=(self.size, self.config.perspective_size),
                    initializer=xavier_initializer)
            # q * W [batch_size,state_size,perspective] 
            # tile [batch_size,max_len_p,state_size,perspective] 
            q_fw_m = tf.mul(tf.expand_dims(out_q_fw, 2), W_1)
            q_bw_m = tf.mul(tf.expand_dims(out_q_bw, 2), W_2)
            # q_fw_m = tf.expand_dims(q_fw_m, 1)
            # q_fw_m = tf.tile(q_fw_m, [1,self.config.output_size,1,1])
            q_fw_m = self.expand_and_tile(q_fw_m, 1, self.config.output_size)
            q_bw_m = self.expand_and_tile(q_bw_m, 1, self.config.output_size)

            # p * W
            p_fw_m = tf.mul(tf.expand_dims(p_fw, 3), W_1)
            p_bw_m = tf.mul(tf.expand_dims(p_bw, 3), W_2)
            
            # cosine sims
            m_fw = self.cosine_sim(p_fw_m, q_fw_m, dim=2)
            m_bw = self.cosine_sim(p_bw_m, q_bw_m, dim=2)
            # [batch_size, max_len_p, perspective]
            full_m_fw = m_fw
            full_m_bw = m_bw
        # with tf.variable_scope("mp_maxcontext"):
        #     # max pooling
        #     W_3 = tf.get_variable("W_3", # forward
        #             shape=(self.size, self.config.perspective_size),
        #             initializer=xavier_initializer)
        #     W_4 = tf.get_variable("W_4", # backward
        #             shape=(self.size, self.config.perspective_size),
        #             initializer=xavier_initializer)
        #     q_fw_m = tf.mul(tf.expand_dims(q_fw, 3), W_3)
        #     q_bw_m = tf.mul(tf.expand_dims(q_bw, 3), W_4)
        #     p_fw_m = tf.mul(tf.expand_dims(p_fw, 3), W_3)
        #     p_bw_m = tf.mul(tf.expand_dims(p_bw, 3), W_4)
        #     # [batch_size, max_len_p, max_len_q, state_size, perspective]
        #     q_fw_m = self.expand_and_tile(q_fw_m, 1, self.max_len_p)
        #     q_bw_m = self.expand_and_tile(q_bw_m, 1, self.max_len_p)
        #     p_fw_m = self.expand_and_tile(p_fw_m, 2, self.max_len_q)
        #     p_bw_m = self.expand_and_tile(p_bw_m, 2, self.max_len_q)

        #     # cosine sims on state_size
        #     m_fw = self.cosine_sim(p_fw_m, q_fw_m, dim=3)
        #     # m_fw = self.cosine_sim_dense(p_fw_m, q_fw_m)#,
        #     #           #[0, 1, 3, 2], [0, 1, 3, 2])
        #    
        #     m_bw = self.cosine_sim(p_bw_m, q_bw_m, dim=3)
        #     # reduce max on max_q_len
        #     m_fw = tf.reduce_max(m_fw, 2)
        #     m_bw = tf.reduce_max(m_bw, 2)
        #     # [batch_size, max_len_p, perspective]
        #     max_m_fw = m_fw
        #     max_m_bw = m_bw
        # with tf.variable_scope("dumb"):
        #     W_2 = tf.get_variable("W_5", # forward
        #             shape=(self.max_len_q, self.size),
        #             initializer=xavier_initializer)
        #     g = tf.mul(q_fw, W_2)
        #     g = tf.reduce_sum(g, 2)
        #     g = tf.tile(tf.expand_dims(g, 1), [1, self.max_len_p, 1])
        #     q_fw_exp = g
        #     g = tf.mul(q_bw, W_2)
        #     g = tf.reduce_sum(g, 2)
        #     g = tf.tile(tf.expand_dims(g, 1), [1, self.max_len_p, 1])
        #     q_bw_exp = g
        # full_m_fw = tf.concat(2, [p_fw, q_fw_exp])
        # full_m_bw = tf.concat(2, [p_bw, q_bw_exp])
        

        # logger.info("full context size fw", full_m_fw.get_shape())
        # logger.info("maxpool context fw  ", max_m_fw.get_shape())

        return tf.concat(2, [full_m_fw, full_m_bw])
        #return tf.concat(2, [full_m_fw, full_m_bw, max_m_fw, max_m_bw])

    def cosine_sim_dense(self, p, q, transpose_p, transpose_q):
        p_tr = tf.transpose(p, transpose_p)
        p_tr = tf.nn.l2_normalize(p_tr, dim=len(transpose_p)-1) # assumes last thing
        q_tr = tf.transpose(q, transpose_q)
        q_tr = tf.nn.l2_normalize(q_tr, dim=len(transpose_q)-1) # assumes last thing
        cosine_similarity = tf.matmul(p_tr, q_tr)
        logger.info("dense sim shape {}".format(cosine_similarity.get_shape()))
        return cosine_similarity


        

    def expand_and_tile(self, vector, axis, tile_num):
        vector = tf.expand_dims(vector, axis)
        # new_dims = [1] * len(vector.get_shape())
        # new_dims[axis] = tile_num
        # vector = tf.tile(vector, new_dims)
        return vector

    def cosine_sim(self, p, q, dim=2):
        # normalize l2 wrt dim
        p = tf.nn.l2_normalize(p, dim)
        q = tf.nn.l2_normalize(q, dim)
        m = tf.mul(p, q)
        m = tf.reduce_sum(m,axis=dim)
        logger.info("shape of cosine {}".format(m.get_shape()))
        return m

    def cnn_attention(self, p_context, p_masks, cnn_q):
        # cnn_q: [?,q_len]
        logging.info(">>>cnn attention")
        q_size = cnn_q.get_shape().as_list()[1]
        # p_context: [?,max_p_len,self.size]
        # for each context word, figure out how q_query influences
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("cnn_attention"):
            #tf.get_variable_scope().reuse_variables()
            W_q = tf.get_variable("W_q",
                shape=(q_size, self.size),
                initializer=xavier_initializer)
            W_q_q = tf.expand_dims(tf.matmul(cnn_q, W_q), 1)
            # [?,1,self.size]

            W_p = tf.get_variable("W_p",
                shape=(2*self.size, self.size),
                initializer=xavier_initializer)
            b_p = tf.get_variable("b_p",
                shape=(self.size,),
                initializer=tf.constant_initializer(0))
            p_reshape = tf.reshape(p_context, [-1, 2*self.size])
            W_p_p = tf.matmul(p_reshape, W_p)
            W_p_p = tf.reshape(W_p_p, [-1, self.max_len_p, self.size]) + b_p
            # [?,max_p_len,self.size]
            G_p = tf.tanh(W_q_q + W_p_p)
            W_scores = tf.get_variable("W_scores",
                shape=(self.size, 2*self.size),
                initializer=xavier_initializer)
            b_scores = tf.get_variable("b_scores",
                shape=(2*self.size,),
                initializer=tf.constant_initializer(0))
            # [?,max_p_len,2*self.size]
            G_p_W = tf.matmul(tf.reshape(G_p, [-1, self.size]), W_scores)
            G_p_W = tf.reshape(G_p_W, [-1, self.max_len_p, 2*self.size])
            scores = tf.nn.softmax(G_p_W + b_scores, 1)
            logging.info("scores{}".format(scores.get_shape()))
            scores = tf.Print(scores, [tf.reduce_sum(scores, 1)], message="norm on dim 1?")

        # [?,max_p_len,4*self.size]
        p_scores = tf.concat(2, [tf.mul(p_context, scores), p_context])

        # another lstm
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                input_keep_prob=self.dropout_placeholder,
                output_keep_prob=self.dropout_placeholder)
        with tf.variable_scope("cnn_attn"):
            p_attn, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                    p_scores,
                    sequence_length=p_masks,
                    dtype=tf.float32)
        # [?,max_p_len,2*self.size]
        p_attn = tf.concat(2, p_attn)
        print("p_attn", p_attn.get_shape())
        return p_attn

class Decoder(object):
    def __init__(self, output_size, flags=None):
        self.output_size = output_size
        if flags is not None:
            self.config = flags
        self.tf_cache = {}

    def basic_decode(self, knowledge_rep,masks=None):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        logging.info(">>>basic decode")
        # 2 for either st or end
        # time_major=False: [batch_size, max_time, 2]
        cell = tf.nn.rnn_cell.BasicLSTMCell(2, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                input_keep_prob=self.dropout_placeholder,
                output_keep_prob=self.dropout_placeholder)
        with tf.variable_scope("basic_decode"):
            xavier_initializer = tf.contrib.layers.xavier_initializer()
            a_st_end, _ = tf.nn.dynamic_rnn(cell, knowledge_rep,
                    sequence_length=masks,
                    dtype=tf.float32)
            # each of dim [batch_size, max_time(, 1)]
            a_st, a_end = tf.unpack(a_st_end, axis=2)
            print ("a_st", a_st.get_shape())

            W_st = tf.get_variable("W_st",
                (self.output_size,self.output_size),
                initializer=xavier_initializer)
            b_st = tf.get_variable("b_st",
                    shape=(self.output_size,),
                    initializer=tf.constant_initializer(0))

            W_end = tf.get_variable("W_end",
                (self.output_size,self.output_size),
                initializer=xavier_initializer)
            b_end = tf.get_variable("b_end",
                    shape=(self.output_size,),
                    initializer=tf.constant_initializer(0))
            logger.info("W_end {}, a_end {}".format(
                W_end.get_shape(), a_end.get_shape()))
            self.y_st = tf.matmul(a_st, W_st) + b_st
            self.y_end = tf.matmul(a_end, W_end) + b_end
            # self.y_st = tf.Print(self.y_st, [self.y_st],
            #         message="y_st pre softmax")
            # self.y_end = tf.Print(self.y_end, [self.y_end],
            #         message="y_end pre softmax")

        return (self.y_st, self.y_end)

    def mix_decode(self, knowledge_rep,masks=None,input_size=-1):
        if input_size == -1:
            input_size = self.config.state_size
        logging.info(">>>mix decode")
        # 2 for either st or end
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("mix_decode_st"):
            W_st = tf.get_variable("W_st",
                (input_size,),
                initializer=xavier_initializer)
            logger.info("W_st {}, p {}".format(
                W_st.get_shape(), knowledge_rep.get_shape()))
            self.y_st = tf.reduce_sum(tf.mul(knowledge_rep, W_st), 2)

        cell = tf.nn.rnn_cell.BasicLSTMCell(input_size,
                state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                input_keep_prob=self.dropout_placeholder,
                output_keep_prob=self.dropout_placeholder)
        with tf.variable_scope("mix_decode_end"):
            # a_end, _ = tf.nn.dynamic_rnn(cell, knowledge_rep,
            #         sequence_length=masks,
            #         dtype=tf.float32)
            a_end = self.cnn_encode(knowledge_rep, self.config.output_size, None)
            # each of dim [batch_size, max_time, output_size)]
            W_end = tf.get_variable("W_end",
                (input_size,),
                initializer=xavier_initializer)
            logger.info("W_end {}, a_end{}".format(
                W_end.get_shape(), a_end.get_shape()))
            self.y_end = tf.reduce_sum(tf.mul(a_end, W_end), 2)

        return (self.y_st, self.y_end)

    def mp_decode(self, knowledge_rep, masks=None):
        logging.info(">>>mp decode")
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                input_keep_prob=self.dropout_placeholder,
                output_keep_prob=self.dropout_placeholder)
        with tf.variable_scope("mp_aggregation"):
            aggr, _ = tf.nn.dynamic_rnn(cell, knowledge_rep,
                    sequence_length=masks,
                    dtype=tf.float32)

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("mp_decode"):
            W_st = tf.get_variable("W_st",
                (self.config.state_size,),
                initializer=xavier_initializer)

            W_end = tf.get_variable("W_end",
                (self.config.state_size,),
                initializer=xavier_initializer)
            self.y_st = tf.reduce_sum(tf.mul(aggr, W_st), 2)
            self.y_end = tf.reduce_sum(tf.mul(aggr, W_end), 2)
            logger.info("y_st shape {}".format(self.y_st.get_shape()))
        return (self.y_st, self.y_end)
        

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        self.saver = None
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.cnn_encode = self.encoder.cnn_encode
        self.pretrained_embeddings = args[0]
        self.max_len_p = args[1]
        self.max_len_q = args[2]
        self.config = args[3] # FLAGS in train.py
        # ==== set up placeholder tokens ========
        self.p_placeholder = tf.placeholder(tf.int32,
                shape=(None,self.max_len_p))
        self.mask_p_placeholder = tf.placeholder(tf.int32,
                shape=(None,))
        self.q_placeholder = tf.placeholder(tf.int32,
                shape=(None,self.max_len_q))
        self.mask_q_placeholder = tf.placeholder(tf.int32,
                shape=(None,))
        self.st_placeholder = tf.placeholder(tf.int32,
                shape=(None,)) # dim 2
        self.end_placeholder = tf.placeholder(tf.int32,
                shape=(None,)) # dim 2
        self.dropout_placeholder = tf.placeholder(tf.float32,
                shape=())
        self.mu_placeholder = tf.placeholder(tf.float32,
                shape=())
        self.lr_placeholder = tf.placeholder(tf.float32,
                shape=())
        # create boolean masks for softmaxing at end
        self.mask_p_seq = tf.sequence_mask(self.mask_p_placeholder,
                maxlen=self.max_len_p, dtype=tf.bool)
        self.mask_q_seq = tf.sequence_mask(self.mask_q_placeholder,
                maxlen=self.max_len_p, dtype=tf.bool)
        
        # let dropout be accessible from encoder and decoder.
        self.encoder.dropout_placeholder = self.dropout_placeholder
        self.decoder.dropout_placeholder = self.dropout_placeholder
        
        self.use_basic = self.config.model_type == 0
        self.use_mp = self.config.model_type == 1
        self.use_mix = self.config.model_type == 2
        self.use_cnn = self.config.model_type == 3

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_training()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        # (None, max_len_p, hidden_size)
        # returns tuple[fw, bw], not concatenated
        if self.use_mp:
            encode_p = self.encoder.mp_filter(self.p_embeddings, self.q_embeddings)
        if not self.use_cnn:
            encode_p, encode_out_p = self.encoder.encode(self.p_embeddings,
                    self.mask_p_placeholder, None)
            encode_q, encode_out_q = self.encoder.encode(self.q_embeddings,
                self.mask_q_placeholder, None, reuse=True)
        if self.use_cnn:
            logging.info("cnn encode for p")
            with tf.variable_scope("cnn_p"):
                cnn_p = self.encoder.cnn_encode(self.p_embeddings,
                    self.max_len_p, None)
            logging.info("cnn encode for q")
            with tf.variable_scope("cnn_q"):
                cnn_q = self.encoder.cnn_encode(self.q_embeddings,
                    self.max_len_q, None)
        if self.use_basic or self.use_mix:
            encode_p = tf.concat(2, encode_p)
        if self.use_basic or self.use_mix:
            encode_q = tf.concat(2, encode_q)
        if self.use_cnn:
            encode_p = cnn_p
            encode_q = cnn_q

        if self.use_basic:
            attn_p = self.encoder.basic_attention(encode_p, encode_q,
                    4*self.config.state_size, 2*self.config.state_size,
                    2*self.config.state_size)
        elif self.use_mp:
            attn_p = self.encoder.mp_attention(encode_p, encode_q, encode_out_q)
        elif self.use_mix:
            attn_p = self.encoder.mix_attention(encode_p, encode_q)
        elif self.use_cnn:
            # attn_p = self.encoder.mix_attention(encode_p, encode_q,
            #         p_size=self.config.state_size)
            attn_p = self.encoder.bidaf_attention(encode_p, encode_q,
                    p_size=self.config.state_size, q_size=self.config.state_size,
                    p_mask=self.mask_p_placeholder)
            # attn_p = self.encoder.cnn_attention(encode_p, 
            #         self.mask_p_placeholder, cnn_q)
        logger.info("attn_p {}".format(attn_p.get_shape()))

        # encoded p, attention p, and elt-wise mult of the two
        if self.use_basic:
            encode_context = tf.concat(2, [attn_p, encode_p, attn_p * encode_p])
        else:
            encode_context = attn_p
        logger.info("encode_context {}".format(encode_context.get_shape()))
        logger.info("mask_p_placeholder {}".format(self.mask_p_placeholder.get_shape()))

        # [None,], [None,] (length is length of batch)
        if self.use_basic:
            self.yp, self.yp2 = self.decoder.basic_decode(encode_context,
                    masks=self.mask_p_placeholder) # start, end
        elif self.use_mp:
            self.yp, self.yp2 = self.decoder.mp_decode(encode_context,
                    masks=self.mask_p_placeholder) # start, end
        elif self.use_mix:
            self.yp, self.yp2 = self.decoder.mix_decode(encode_context,
                    masks=self.mask_p_placeholder)
        elif self.use_cnn:
            self.yp, self.yp2 = self.decoder.mix_decode(encode_context,
                    masks=self.mask_p_placeholder,
                    input_size=self.config.state_size) # using bidaf modeling: lstm
        logger.info("self.yp {}, yp2 {}".format(self.yp, self.yp2))

    def exp_mask(self, val, mask):
        VERY_NEGATIVE_NUMBER = -1e30
        logging.info("cast mask shape {}".format(tf.cast(mask, 'float').get_shape()))
        return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            logger.info("placeholder size {}".format(
                self.st_placeholder.get_shape()))
            logger.info("yp size {}".format(self.yp.get_shape()))
            self.st_inds = tf.Print(self.st_placeholder,
                    [self.st_placeholder],
                    message="expected starts",
                    summarize=self.config.batch_size)
            self.end_inds = tf.Print(self.end_placeholder,
                    [self.end_placeholder],
                    message="expected ends",
                    summarize=self.config.batch_size)
            self.yp2 = tf.Print(self.yp2, [self.mask_p_placeholder],
                    message="mask lens",
                    summarize=self.config.batch_size)

            # mask
            yp_mask = self.exp_mask(self.yp, self.mask_p_seq)
            yp2_mask = self.exp_mask(self.yp2, self.mask_p_seq)
            logging.info("yp shape{}".format(yp_mask.get_shape()))
            # TODO:ignore mask
            # yp_mask = self.yp
            # yp2_mask = self.yp2
            yp_mask = tf.Print(yp_mask, [tf.argmax(yp_mask, 1)],
                    message="guessed starts",
                    summarize=self.config.batch_size)

            yp2_mask = tf.Print(yp2_mask, [tf.argmax(yp2_mask, 1)],
                    message="guessed ends",
                    summarize=self.config.batch_size)

            batch_softmax_st = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.st_inds, logits=yp_mask)
            batch_softmax_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.end_inds, logits=yp2_mask)
            # logger.info("len st and end", len(st_batch), len(end_batch), len(ans_batch))
            # Lisa: changed 3/17 to sum of average losses from each of these

            # constrain start < end
            ans_st = tf.argmax(yp_mask,1)
            ans_end = tf.argmax(yp2_mask,1)
            pos_range = (ans_st - ans_end)
            pos_range = tf.Print(pos_range, [pos_range],
                    message="diff",
                    summarize=self.config.batch_size)
            self.loss = tf.reduce_mean(batch_softmax_st + batch_softmax_end) + \
                    self.mu_placeholder * tf.reduce_mean(tf.cast(pos_range, tf.float32))

    def setup_training(self):
        # self.train_op = \
        #     tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        # with tf.variable_scope("minimize_step"):
        #     self.global_step = tf.get_variable("global_step",
        #             0, trainable=False)
        #     epoch_size = self.config.num_iters / float(self.config.batch_size)
        #     self.lr_exp = tf.train.exponential_decay(
        #       self.config.learning_rate,                # Base learning rate.
		# 	    self.global_step,  # Current index into the dataset.
		# 	    2 * epoch_size,          # Decay step. (every two epochs)
		# 	    0.95,                # Decay rate,
        #         staircase=True)
        optimizer_type = \
                get_optimizer(self.config.optimizer)
        # TODO: NOT DECAYING LEARNING RATE BC ADAM OPTIMIZER
        #optimizer = optimizer_type(self.lr_exp)
        optimizer = optimizer_type(self.lr_placeholder)
        grads_and_vars = optimizer.compute_gradients(self.loss)

        if self.config.clip_gradients:
            grads, and_vars = zip(*grads_and_vars)
            grads, _ = \
                    tf.clip_by_global_norm(grads,
                            self.config.max_grad_norm)
            grads_and_vars = zip(grads, and_vars)

        self.grad_norm = tf.global_norm(grads_and_vars)
        self.grad_norm = tf.Print(self.grad_norm,
                [self.grad_norm], message="grad norm:")
        self.train_op = optimizer.apply_gradients(grads_and_vars)
                #global_step=self.global_step)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = tf.Variable(self.pretrained_embeddings,
                    dtype=tf.float32,
                    trainable=False)
            self.p_embeddings = tf.nn.embedding_lookup(embeddings,
                    self.p_placeholder)
            # (None, max_len_p, embed.size)
            logger.info("p shape {}".format(self.p_placeholder.get_shape()))
            logger.info("p embeddings shape {}".format(
                self.p_embeddings.get_shape()))

            self.q_embeddings = tf.nn.embedding_lookup(embeddings,
                    self.q_placeholder)
            # (None, max_len_p, embed.size)
            logger.info("q shape {}".format(self.q_placeholder.get_shape()))
            logger.info("q embeddings shape {}".format(self.q_embeddings.get_shape()))

    def create_feed_dict(self, p_batch, mask_p_batch, q_batch, mask_q_batch, ans_batch=None, dropout=1, mu=0.001):
        input_feed = {}
        input_feed[self.p_placeholder] = p_batch
        input_feed[self.mask_p_placeholder] = mask_p_batch
        input_feed[self.q_placeholder] = q_batch
        input_feed[self.mask_q_placeholder] = mask_q_batch
        if ans_batch is not None:
            st_batch, end_batch = zip(*ans_batch)
            input_feed[self.st_placeholder] = st_batch
            input_feed[self.end_placeholder] = end_batch
        #dropout = 1 # TODO: change and move to train.py
        input_feed[self.dropout_placeholder] = dropout
        input_feed[self.mu_placeholder] = mu
        input_feed[self.lr_placeholder] = self.config.learning_rate
        # logger.info("input_feed_dict {}, {}".format(
        #     len(input_feed), len(input_feed[self.p_placeholder])))
        return input_feed

    def optimize(self, sess, data):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        p_batch, mask_p_batch, q_batch, mask_q_batch, ans_batch = data
        input_feed = self.create_feed_dict(p_batch, mask_p_batch,
                q_batch, mask_q_batch, ans_batch=ans_batch,
                dropout=self.config.dropout,
                mu=self.config.mu)

        #run_metadata = tf.Print(run_metadata, [run_metadata])
        output_feed = [self.train_op, self.loss, self.grad_norm]

        outputs = sess.run(output_feed, input_feed,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                    output_partition_graphs=True))
        _, loss, norms = outputs

        return loss, norms

    def decode(self, sess, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        p_batch, mask_p_batch, q_batch, mask_q_batch = test_x
        input_feed = self.create_feed_dict(p_batch, mask_p_batch,
                q_batch, mask_q_batch)

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        # output_feed = [self.train_op, self.loss]
        # _, inds = sess.run(output_feed, input_feed)

        output_feed = [self.decoder.y_st, self.decoder.y_end]
        y_st, y_end = sess.run(output_feed, input_feed)

        return y_st, y_end

    def answer(self, sess, test_x):

        yp, yp2 = self.decode(sess, test_x)

        yp_list = (yp).tolist()
        yp2_list = (yp2).tolist()
        batch_size = len(yp_list)

        spans = []
        for i in range(len(yp_list)):
            ypif, yp2if = yp_list[i], yp2_list[i]
            best_st = 0
            max_jp = 0 # max joint probability
            best_span = (0, 1)
            for j in range(len(ypif)):
                p_st = ypif[best_st] # the default best prob
                if ypif[j] > ypif[best_st]:
                    best_st = j
                    p_st = ypif[j]
                p_end = yp2if[j]
                if p_st * p_end > max_jp:
                    best_span = (best_st, j+1)
                    max_jp = p_st * p_end
            spans.append(best_span)
        return zip(*spans)


        # greedy, has empty spans
        # yp, yp2 = self.decode(sess, test_x)
        # a_s = np.argmax(yp, axis=1)
        # a_e = np.argmax(yp2, axis=1)
        # return (a_s, a_e)

    def test(self, sess, val_data):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        y_st, y_end = self.decode(sess, data[:-1])
        st_batch, end_batch = zip(*data[-1])
        # p_batch, mask_p_batch, q_batch, mask_q_batch, ans_batch = data
        # input_feed = self.create_feed_dict(p_batch, mask_p_batch,
        #         q_batch, mask_q_batch, ans_batch=ans_batch)
        # output_feed = [self.decoder.y_st, self.decoder.y_end]
        # y_st, y_end = sess.run(output_feed, input_feed)

        return outputs

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, sess, dataset, train_set=True, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param sess: sess should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        # tokens predicted
        # correctly predicted
        # tokens in actual
        # precision: # true positives/(all predicted)
        # recall: # true positives/(all correct)
        # all correct = tp + fn
        # all predicted = tp + fp
        # tn we don't care about that much
        f1 = 0.
        em = 0.
        # just want one "minibatch"
        raw_dataset = self.raw_train
        if not train_set:
            raw_dataset = self.raw_val
        sample = min(sample, len(dataset)) # in case it's too small

        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        indices = indices[:sample]
        samples = zip(*[dataset[i] for i in indices])
        actual_st, actual_end = zip(*samples[-1])
        text_samples = [raw_dataset[i] for i in indices]


        # # sanity check with argmax
        # yp, yp2 = self.decode(sess, samples[:-1])
        # a_s = np.argmax(yp, axis=1)
        # a_e = np.argmax(yp2, axis=1)

        # does runs
        guess_st, guess_end = self.answer(sess, samples[:-1])
        for i in range(sample):
            raw_par = text_samples[i]
            prediction = get_substring(text_samples[i],
                    guess_st[i], guess_end[i])
            actual = get_substring(text_samples[i],
                    actual_st[i], actual_end[i])
            # # logger.info("prediction argmax:({},{})".format(
            # #     a_s[i], a_e[i], get_substring(text_samples[i],
            # #         a_s[i], a_e[i])))
            # logger.info("prediction:({},{}){}\nactual:({},{}){}".format(
            #     guess_st[i], guess_end[i],prediction,
            #     actual_st[i], actual_end[i], actual))
            f1 += f1_score(prediction, actual)
            em += exact_match_score(prediction, actual)

        avg_f1, avg_em = f1/float(sample), em/float(sample)
        logger.info("f1 {}, em {}".format(avg_f1, avg_em))
        return avg_f1, avg_em
            

        # TODO: use f1_score, exact_match_score from evaluate.py!!

        par_lens = map(len, samples[0])
        macro = np.array([0., 0., 0., 0.])
        micro = np.array([0., 0., 0., 0., 0.])
        default = np.array([0., 0., 0., 0., 0.])
        data = []
        for i in range(sample):
            # logger.info("actual: %s, %s" % (actual_st[i], actual_end[i]))
            # logger.info("guess: %s, %s" % (guess_st[i], guess_end[i]))
            act_range = np.arange(actual_st[i], actual_end[i]+1)
            guess_range = np.arange(guess_st[i], guess_end[i]+1)
            fp = len(np.setdiff1d(guess_range, act_range).tolist())
            tp = max(len(guess_range) - fp, 0)
            fn = len(np.setdiff1d(act_range, guess_range).tolist())
            tn = max(par_lens[i] - len(guess_range), 0)
            exact = len(guess_range) == len(act_range) and \
                    fp == 0 and fn == 0 and \
                    tp == len(act_range)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
            # logger.info("for this sample: acc %s, prec %s, rec %s, f1 %s, exact %s" % \
            #         (acc, prec, rec, f1, int(exact)))

            # update micro/macro averages
            micro += np.array([tp, fp, tn, fn, int(exact)])
            macro += np.array([acc, prec, rec, f1])
            default += np.array([tp, fp, tn, fn, int(exact)])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn, exact = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        em = exact / float(sample)
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / float(sample))
        tp, fp, tn, fn, exact = default
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        em = exact / float(sample)
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        #return to_table(data, self.labels + ["micro","macro","not-O"], ["label", "acc", "prec", "rec", "f1"])
        logger.info("total exact: %s" % exact)

        #if log:
        #logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
        logger.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, sess, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param sess: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        self.saver = tf.train.Saver()

        tic = time.time()
        params = tf.trainable_variables()
        logger.info("trainable variables {}".format('\n'.join(
            map(lambda var: '\t{}:{}'.format(var.name, var.get_shape()),
            params))))
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # Lisa
        self.best_score = 0.
        train_set, val_set, raw_set = dataset
        logger.info("train_set{}".format(len(train_set)))
        logger.info("val_set {}".format(len(val_set)))
        self.raw_train, self.raw_val = raw_set
        for epoch in range(self.config.epochs):
            logger.info("Epoch %d out of %d", epoch+1, self.config.epochs)
            g = tf.get_default_graph()
            if epoch % 2 == 0 and epoch != 0:
                # decrease every other epoch?
                self.config.learning_rate /= self.config.exp_reduce
                logger.info("Reducing learning rate: {}".format(self.config.learning_rate))
            def dim_calc(tfshape):
                is_none = False
                prod = 1
                try:
                    for s in tfshape.as_list():
                        if s is None: is_none = True; continue
                        prod *= s
                    return str("(%s): %s" % (prod, tfshape.as_list()))
                except:
                    return str(tfshape)
            if epoch == 0: # only for first epoch
                for op in g.get_operations():
                    logger.info("{}, {}".format(op.name,
                        map(lambda v: dim_calc(v.get_shape()),
                                op.outputs)))
            score = self.run_epoch(sess, train_set, val_set)
            if score > self.best_score:
                self.best_score = score
                logger.info("New best score!")
                if self.saver:
                    logger.info("Saving model in {}".format(
                        self.config.train_dir))
                    self.saver.save(sess,
                            os.path.join(self.config.train_dir,
                                'model{}.weights'.format(self.config.sessname)),
                            latest_filename='checkpoint{}'.format(
                                self.config.sessname))
    # Lisa
    # from assignment3/ner_model.py
    def run_epoch(self, sess, train_set, dev_set):
        prog = Progbar(target=1 + int(len(train_set) / self.config.batch_size))
        print_every = 50 
        for i, batch in enumerate(minibatches(train_set, self.config.batch_size)):
            loss, norms = self.optimize(sess, batch)
            prog.update(i + 1, [("train loss", loss)])
            if i % print_every == 1:
                logger.info("Current batch:{}/{}, loss: {}, grad norm: {}".format(
                    i, int(len(train_set)/self.config.batch_size), loss, norms))
                f1, em = self.evaluate_answer(sess, train_set, log=True)
                f1_val, em_val = self.evaluate_answer(sess, dev_set, log=True,
                        train_set=False)

                logger.info("Train F1: {}, EM: {}, for {} samples".format(f1, em, 100))
                logger.info("Val F1: {}, EM: {}, for {} samples".format(f1_val, em_val, 100))
                if f1 > self.best_score:
                    logger.info("New best score!")
                    self.best_score = f1
                    if self.saver:
                        logger.info("Saving model in {}".format(
                            self.config.train_dir))
                        self.saver.save(sess,
                                os.path.join(self.config.train_dir,
                                    'model{}.weights'.format(self.config.sessname)))
        logger.info("")
        f1, em = self.evaluate_answer(sess, train_set, log=True)
        f1_val, em_val = self.evaluate_answer(sess, dev_set, log=True,
                train_set=False)
        logger.info("After epoch: Train F1: {}, EM: {}, for {} samples".format(f1, em, 100))
        logger.info("After epoch: Val F1: {}, EM: {}, for {} samples".format(f1_val, em_val, 100))

        # TODO: implement validation on dev set.
        # from ner_model.py: self.evaluate(sess, dev_set)
        # here: test() or validate()
        return f1
