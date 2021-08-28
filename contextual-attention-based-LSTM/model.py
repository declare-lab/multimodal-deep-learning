import tensorflow as tf

from tensorflow.python.layers.core import Dense


class LSTM_Model():
    def __init__(self, input_shape, lr, a_dim, v_dim, t_dim, emotions, attn_fusion=True, unimodal=False,
                 enable_attn_2=False, seed=1234):
        if unimodal:
            self.input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], input_shape[1]))
        else:
            self.a_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], a_dim))
            self.v_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], v_dim))
            self.t_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], t_dim))
        self.emotions = emotions
        self.mask = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0]))
        self.seq_len = tf.placeholder(tf.int32, [None, ], name="seq_len")
        self.y = tf.placeholder(tf.int32, [None, input_shape[0], self.emotions], name="y")
        self.lr = lr
        self.seed = seed
        self.attn_fusion = attn_fusion
        self.unimodal = unimodal
        self.lstm_dropout = tf.placeholder(tf.float32, name="lstm_dropout")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.lstm_inp_dropout = tf.placeholder(tf.float32, name="lstm_inp_dropout")
        self.dropout_lstm_out = tf.placeholder(tf.float32, name="dropout_lstm_out")
        self.attn_2 = enable_attn_2

        # Build the model
        self._build_model_op()
        self._initialize_optimizer()

    def GRU(self, inputs, output_size, name, dropout_keep_rate):
        with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):
            kernel_init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)
            bias_init = tf.zeros_initializer()

            cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,
                                          kernel_initializer=kernel_init, bias_initializer=bias_init)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_rate)

            output, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)

            return output

    def GRU2(self, inputs, output_size, name, dropout_keep_rate):
        with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):
            kernel_init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)
            bias_init = tf.zeros_initializer()

            fw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,
                                             kernel_initializer=kernel_init, bias_initializer=bias_init)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_rate)

            bw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,
                                             kernel_initializer=kernel_init, bias_initializer=bias_init)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_rate)

            output_fw, _ = tf.nn.dynamic_rnn(fw_cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)
            output_bw, _ = tf.nn.dynamic_rnn(bw_cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            return output

    def BiGRU(self, inputs, output_size, name, dropout_keep_rate):
        with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):
            kernel_init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)
            bias_init = tf.zeros_initializer()

            fw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,
                                             kernel_initializer=kernel_init, bias_initializer=bias_init)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_rate)

            # bw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,
            #                                 kernel_initializer=kernel_init, bias_initializer=bias_init)
            # bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_rate)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=fw_cell, inputs=inputs,
                                                         sequence_length=self.seq_len, dtype=tf.float32)

            output_fw, output_bw = outputs
            output = tf.concat([output_fw, output_bw], axis=-1)
            return output

    def self_attention(self, inputs_a, inputs_v, inputs_t, name):
        """

        :param inputs_a: audio input (B, T, dim)
        :param inputs_v: video input (B, T, dim)
        :param inputs_t: text input (B, T, dim)
        :param name: scope name
        :return:
        """

        inputs_a = tf.expand_dims(inputs_a, axis=1)
        inputs_v = tf.expand_dims(inputs_v, axis=1)
        inputs_t = tf.expand_dims(inputs_t, axis=1)
        # inputs = (B, 3, T, dim)
        inputs = tf.concat([inputs_a, inputs_v, inputs_t], axis=1)
        t = inputs.get_shape()[2].value
        share_param = True
        hidden_size = inputs.shape[-1].value  # D value - hidden size of the RNN layer
        kernel_init1 = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)
        # kernel_init2 = tf.random_normal_initializer(seed=self.seed, dtype=tf.float32,stddev=0.01)
        # bias_init = tf.zeros_initializer()
        dense = Dense(hidden_size, kernel_initializer=kernel_init1)
        if share_param:
            scope_name = 'self_attn'
        else:
            scope_name = 'self_attn' + name
        # print(scope_name)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        with tf.variable_scope(scope_name):
            outputs = []
            for x in range(t):
                t_x = inputs[x, :, :, :]
                # t_x => B, 3, dim
                den = True
                if den:
                    x_proj = dense(t_x)
                    x_proj = tf.nn.tanh(x_proj)
                else:
                    x_proj = t_x
                u_w = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.01, seed=1234))
                x = tf.tensordot(x_proj, u_w, axes=1)
                alphas = tf.nn.softmax(x, axis=-1)
                output = tf.matmul(tf.transpose(t_x, [0, 2, 1]), alphas)
                output = tf.squeeze(output, -1)
                outputs.append(output)

            final_output = tf.stack(outputs, axis=1)
            # print('final_output', final_output.get_shape())
            return final_output

    def attention(self, inputs_a, inputs_b, attention_size, params, mask=None, return_alphas=False):
        """
        inputs_a = (b, 18, 100)
        inputs_b = (b, 100)
        :param inputs_a:
        :param inputs_b:
        :param attention_size:
        :param time_major:
        :param return_alphas:
        :return:
        """
        if mask is not None:
            mask = tf.cast(self.mask, tf.bool)
        shared = True
        if shared:
            scope_name = 'attn'
        else:
            scope_name = 'attn_'
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            hidden_size = inputs_a.shape[2].value  # D value - hidden size of the RNN layer
            den = False
            x_proj = inputs_a
            y_proj = inputs_b
            # print('x_proj', x_proj.get_shape())
            # print('y_proj', y_proj.get_shape())

            # Trainable parameters
            w_omega = params['w_omega']
            b_omega = params['b_omega']
            # dense_attention_2 = params['dense']
            with tf.variable_scope('v', reuse=tf.AUTO_REUSE):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size

                v = tf.tensordot(x_proj, w_omega, axes=1) + b_omega
                # v  = dense_attention_2(x_proj)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tanh(tf.matmul(v, tf.expand_dims(y_proj, -1), name='vu'))  # (B,T) shape (B T A) * (B A 1) = (B T)
            vu = tf.squeeze(vu, -1)
            # print('vu', vu.get_shape())
            # masking
            # mask = None
            if mask is not None:
                vu = tf.where(mask, vu, tf.zeros(tf.shape(vu), dtype=tf.float32))

            alphas = tf.nn.softmax(vu, 1, name='alphas')  # (B,T) shape
            if mask is not None:
                alphas = tf.where(mask, alphas, tf.zeros(tf.shape(alphas), dtype=tf.float32))
                a = tf.reduce_sum(tf.expand_dims(alphas, -1), axis=1)
                condition = tf.equal(a, 0.0)
                case_true = tf.ones(tf.shape(a), tf.float32)
                a_m = tf.where(condition, case_true, a)
                alphas = tf.divide(alphas, a_m)

            # print('alphas', alphas.get_shape())

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            output = tf.matmul(tf.transpose(inputs_a, [0, 2, 1]), tf.expand_dims(alphas, -1))
            output = tf.squeeze(output, -1)
            # print('r', output.get_shape())
            # output = tf.reduce_sum(r, 1)

            if not return_alphas:
                return tf.expand_dims(output, 1)
            else:
                return tf.expand_dims(output, 1), alphas

    def self_attention_2(self, inputs, name):
        """

        :param inputs_a: audio input (B, T, dim)
        :param inputs_v: video input (B, T, dim)
        :param inputs_t: text input (B, T, dim)
        :param name: scope name
        :return:
        """

        t = inputs.get_shape()[1].value
        share_param = True
        hidden_size = inputs.shape[-1].value  # D value - hidden size of the RNN layer
        if share_param:
            scope_name = 'self_attn_2'
        else:
            scope_name = 'self_attn_2' + name
        # print(scope_name)
        # inputs = tf.transpose(inputs, [2, 0, 1, 3])
        # dense = Dense(hidden_size)
        # init1 = tf.random_normal_initializer(seed=self.seed, dtype=tf.float32,stddev=0.01)
        attention_size = hidden_size
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.01, seed=self.seed))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.01, seed=self.seed))
        # dense_attention_2 = Dense(attention_size, activation=None,kernel_initializer=init1,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        params = {'w_omega': w_omega,
                  'b_omega': b_omega,
                  # 'dense': dense_attention_2
                  }
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            outputs = []
            for x in range(t):
                t_x = inputs[:, x, :]

                output = self.attention(inputs, t_x, hidden_size, params, self.mask)  # (b, d)
                outputs.append(output)

            final_output = tf.concat(outputs, axis=1)
            return final_output

    def _build_model_op(self):
        # self attention
        if self.unimodal:
            input = self.input
        else:
            if self.attn_fusion:
                input = self.self_attention(self.a_input, self.v_input, self.t_input, '')
                input = input * tf.expand_dims(self.mask, axis=-1)
            else:
                input = tf.concat([self.a_input, self.v_input, self.t_input], axis=-1)

        # input = tf.nn.dropout(input, 1-self.lstm_inp_dropout)
        self.gru_output = self.BiGRU(input, 100, 'gru', 1 - self.lstm_dropout)
        self.inter = tf.nn.dropout(self.gru_output, 1 - self.dropout_lstm_out)
        # self.inter = self.gru_output
        if self.attn_2:
            self.inter = self.self_attention_2(self.inter, '')
        init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)
        if self.unimodal:
            self.inter1 = Dense(100, activation=tf.nn.tanh,
                                kernel_initializer=init, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(
                self.inter)
        else:
            self.inter1 = Dense(200, activation=tf.nn.relu,
                                kernel_initializer=init, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(
                self.inter)
            self.inter1 = self.inter1 * tf.expand_dims(self.mask, axis=-1)
            self.inter1 = Dense(200, activation=tf.nn.relu,
                                kernel_initializer=init, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(
                self.inter1)
            self.inter1 = self.inter1 * tf.expand_dims(self.mask, axis=-1)
            self.inter1 = Dense(200, activation=tf.nn.relu,
                                kernel_initializer=init, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(
                self.inter1)
        self.inter1 = self.inter1 * tf.expand_dims(self.mask, axis=-1)
        self.inter1 = tf.nn.dropout(self.inter1, 1 - self.dropout)
        self.output = Dense(self.emotions, kernel_initializer=init,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(self.inter1)
        # print('self.output', self.output.get_shape())
        self.preds = tf.nn.softmax(self.output)
        # To calculate the number correct, we want to count padded steps as incorrect
        correct = tf.cast(
            tf.equal(tf.argmax(self.preds, -1, output_type=tf.int32), tf.argmax(self.y, -1, output_type=tf.int32)),
            tf.int32) * tf.cast(self.mask, tf.int32)

        # To calculate accuracy we want to divide by the number of non-padded time-steps,
        # rather than taking the mean
        self.accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(self.seq_len, tf.float32))
        # y = tf.argmax(self.y, -1)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.y)
        loss = loss * self.mask

        self.loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)

    def _initialize_optimizer(self):
        train_vars = tf.trainable_variables()
        reg_loss = []
        total_parameters = 0
        for train_var in train_vars:
            # print(train_var.name)
            reg_loss.append(tf.nn.l2_loss(train_var))

            shape = train_var.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        # print(total_parameters)
        print('Trainable parameters:', total_parameters)

        self.loss = self.loss + 0.00001 * tf.reduce_mean(reg_loss)
        self.global_step = tf.get_variable(shape=[], initializer=tf.constant_initializer(0), dtype=tf.int32,
                                           name='global_step')
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
        # self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08)

        self.train_op = self._optimizer.minimize(self.loss, global_step=self.global_step)
