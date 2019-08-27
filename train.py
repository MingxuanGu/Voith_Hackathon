import tensorflow as tf
import model
from data import data
from trainer import Trainer
import numpy as np
LEARNING_RATE = 2.0
ADAM_LEARNING_RATE   = 1
SGD_LEARNINGRATE = 2
Adadelta_LEARNING_RATE = 1
BATCH_NUM = -1
EPOCHS = 3000
ITERATION = 500
DECAY_RATE = 0.95
def main():
    dataset = data()
    data_train = dataset.get_data()
    train_iter = 0
    if (BATCH_NUM==-1):
        train_iter = 1
    else:
        train_iter = int(np.floor(data_train.shape[0]/BATCH_NUM))
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        x = tf.placeholder(tf.float32, [None] + [21], 'inputs')
        labels = tf.placeholder(tf.float32, [None] + [2], 'labels')
        prediction_logits = model.create(x, 2)
        prediction = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = prediction_logits)
        loss = tf.reduce_mean(prediction)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=ITERATION,
                                                   decay_rate=DECAY_RATE,
                                                   staircase=True)
        optimizer = tf.train.AdadeltaOptimizer(Adadelta_LEARNING_RATE)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=SGD_LEARNINGRATE)
        # optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_LEARNING_RATE)
        # print("train_iter")
        # print(train_iter)
        # input()
        trainer = Trainer(loss, prediction_logits, optimizer, x, labels, train_iter, dataset)

    with tf.Session() as sess:

        writer = tf.summary.FileWriter('./graph', sess.graph)
        sess.run(tf.initializers.global_variables())


        # train
        print('Run training loop')
        trainer.run(sess, num_epochs=EPOCHS)

        #save
        # print('Save model')
        # sh.simple_save(sess, str(SAVE_DIR), inputs = {'x': x}, outputs = {'y':prediction})


        #create zip file for submission
        # print('Create zip file for submission')
        # sh.zip_dir(SAVE_DIR, SAVE_DIR / 'model.zip')
    print("finish")


if __name__=='__main__':
    main()
