import tensorflow as tf
import numpy as np

'''
Tuan Anh's tensorflow book
'''

# data I/O
input_file = "shakespeare.txt"
data = open(input_file, 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
data_in_int = [char_to_ix[c] for c in data]
data_in_onehot = np.zeros([len(data),len(chars)])
data_in_onehot[np.arange(len(data)),data_in_int] = 1
print ('data has %d characters, %d unique.' % (data_size, vocab_size))

#hyper parameters
n_inputs = len(chars)
n_outputs = len(chars)
seq_length = 50
n_neurons_1 = 50
n_neurons_2 = 50
learning_rate = 0.001
train_size = int(1e6)
batch_size = int(train_size/seq_length)
epochs = int(1)

#model
X = tf.placeholder(tf.float32,[None,seq_length,n_inputs])

cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons_1,use_peepholes=True,name="cell_1")
cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons_2,use_peepholes=True,name="cell_2")

multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell_1,cell_2])
h_states, fin_state = tf.nn.dynamic_rnn(multi_layer_cell,X,dtype=tf.float32) #h_states: None*seq_length*n_neurons_2, fin_state: tupple of last c and h states[None*n_neurons_2, None*n_neurons_2]
outputs = tf.layers.dense(h_states,n_outputs,name="dense") #None*seq_length*n_outputs
#softmax_outputs = tf.nn.softmax(outputs,axis=2) #None*seq_length*n_outputs

#loss
Y = tf.placeholder(tf.int32, [None, seq_length, n_outputs]) #None*seq_length*n_outputs

loss = tf.losses.softmax_cross_entropy(Y,outputs)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

#predict
predict = tf.argmax(outputs,axis=2) #None*seq_length
predict_onehot = tf.one_hot(predict,depth=n_outputs) #None*seq_length*n_outputs

#save
saver = tf.train.Saver()

#train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        start_i = np.random.randint(0,seq_length)
        X_train = np.reshape(data_in_onehot[start_i:start_i+train_size],[-1,seq_length,n_inputs])
        Y_train = np.reshape(data_in_onehot[start_i+1:start_i+1+train_size],[-1,seq_length,n_outputs])

        _train, _loss = sess.run([training_op,loss], feed_dict={X:X_train,Y:Y_train})

        print("epoch: %d CEL: %f"%(epoch, _loss))
        if(epoch%100==0):
            # save
            save_path = saver.save(sess, "./checkpoints_LSTM/model.ckpt")
            print("Model saved in path: %s" % save_path)
            # generate sequence
            seq_onehot = np.zeros([seq_length, n_inputs])
            story_length = 500
            for i_char in range(story_length):
                X_batch = np.reshape(seq_onehot[-seq_length:], [1, seq_length, n_inputs])
                gen_onehot = sess.run(predict_onehot, feed_dict={X: X_batch})
                seq_onehot = np.append(seq_onehot, gen_onehot[0, -1, :].reshape([1, -1]), axis=0)

            # print result:
            story = ""
            for i in range(seq_onehot.shape[0]):
                story += ix_to_char[np.argmax(seq_onehot[i])]
            print(story)




