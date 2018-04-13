import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction

from common import tf_util, rnn_util


class rnn_model(tf_util.BaseModel):
    def __init__(self,
                 create_embedding_layer_fn,
                 use_chacacter_embedding,
                 rnn_layer_num,
                 hidden_state_size,
                 learning_rate,
                 dropout,
                 keep_prob,
                 l2,
                 l2_decay,
                 output_dense_num,
                 output_dense_size,
                 decay_rate,
                 decay_steps,
                 sample_num,
                 ):
        """
        :param create_embedding_layer_fn: the function used to create the embedding layer
        :param use_chacacter_embedding: boolean indicates whether use character embedding
        :param rnn_layer_num: the rnn layer number
        :param hidden_state_size: the rnn hidden state size
        :param learning_rate: the learning_rate
        :param dropout: boolean, indicate whether to use dropout
        :param keep_prob: the dropout keep rate
        :param l2: boolean indicate whether use l2
        :param l2_decay: l2 parameter
        :param output_dense_num: the layer number of the output network
        :param output_dense_size: the output network hidden layer size
        :param decay_rate: the exponential learning rate decay rate
        :param decay_steps: the exponential learning rate decay steps
        """
        super().__init__(learning_rate=learning_rate)
        self.embedding_layer_fn = create_embedding_layer_fn()
        self.vocabulary_size = len(create_embedding_layer_fn)
        self.use_chacacter_embedding = use_chacacter_embedding
        self.input_seq = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None),
                                        name="input_seq")
        self.input_seq_length = tf.placeholder(dtype=tf.int32,
                                               shape=(None, ),
                                               name="input_seq_length")
        input_placeholder = [self.input_seq, self.input_seq_length]
        if self.use_chacacter_embedding:
            self.character_input_seq = tf.placeholder(dtype=tf.int32,
                                                      shape=(None, None, None),
                                                      name="character_input_seq")
            self.character_input_seq_length = tf.placeholder(dtype=tf.int32,
                                                             shape=(None, None),
                                                             name="character_input_seq_length")
            input_placeholder += [self.character_input_seq, self.character_input_seq_length]

        self._input_placeholder = input_placeholder
        self.rnn_layer_num = rnn_layer_num
        self.hidden_state_size = hidden_state_size
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.l2 = l2
        self.l2_decay = l2_decay
        self.output_dense_num = output_dense_num
        self.output_dense_size = output_dense_size
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.sample_num = sample_num

        # tf_util.init_all_op(self)

        tf_util.add_summary_scalar("loss", self.loss_op)
        self._summary_op = tf_util.merge_op()

        sess = tf_util.get_session()
        init = tf.global_variables_initializer()
        sess.run(init)

    @property
    def input_placeholders(self):
        return self._input_placeholder

    @tf_util.define_scope("word_embedding",)
    def embedding_op(self):
        if not self.use_chacacter_embedding:
            return self.embedding_layer_fn(self.input_seq)
        else:
            return self.embedding_layer_fn(self.input_seq, self.character_input_seq, self.character_input_seq_length)

    def _gru_cell(self):
        if self.dropout:
            return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.hidden_state_size),
                                                 output_keep_prob=self.keep_prob)
        else:
            return tf.nn.rnn_cell.GRUCell(self.hidden_state_size)

    def _mutligru_cell(self):
        return tf.nn.rnn_cell.MultiRNNCell([self._gru_cell() for _ in range(self.rnn_layer_num)])

    @tf_util.define_scope("rnn", initializer=tf.orthogonal_initializer())
    def rnn_op(self):
        embedding = self.embedding_op
        return rnn_util.rnn(self._mutligru_cell,
                            embedding,
                            self.input_seq_length)

    @tf_util.define_scope("forward_label")
    def forward_label_op(self):
        return self.input_seq[:, 1:]

    @tf_util.define_scope("forward_predict")
    def forward_logit_op(self):
        forward_output = self.rnn_op[0][:, :-1, :]
        forward_output = tf.contrib.layers.fully_connected(forward_output, tf_util.get_shape(self.embedding_op)[-1],
                                                           None)
        return forward_output

    @tf_util.define_scope("lm")
    def lm_loss_op(self):
        sample_num = self.sample_num
        vocabulary_size = self.vocabulary_size
        embedding_size = tf_util.get_shape(self.embedding_op)[-1]
        def nce_loss_fn(labels, logits):
            labels = tf.reshape(labels, (-1, 1))
            logits = tf.reshape(logits, (-1, embedding_size))
            nce_weights = tf.get_variable(
                name="nce_weight",
                shape=(vocabulary_size, embedding_size)
            )
            nce_bias = tf.get_variable(
                name="nce_bias",
                shape=(vocabulary_size, )
            )
            return tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_bias,
                    labels=labels,
                    inputs=logits,
                    num_sampled=sample_num,
                    num_classes=vocabulary_size,
                )
            )
        with tf.variable_scope("forward_loss"):
            forward_output = self.forward_logit_op
            forward_label = self.forward_label_op
            nce_loss = nce_loss_fn(forward_label, forward_output)

        return nce_loss

    @tf_util.define_scope("loss")
    def loss_op(self):
        loss = tf.constant(0.0, dtype=tf.float32)
        if not self.l2:
            pass
        else:
            loss += tf_util.regularizer_loss()

        loss += self.lm_loss_op

        return loss

    @tf_util.define_scope("perplexity")
    def perplexity_op(self):
        return tf.reduce_mean(tf_util.perplexity(self.forward_logit_op, self.forward_label_op, self.input_seq_length-1))

    @property
    def summary_op(self):
        return self._summary_op
