import tensorflow as tf
from matplotlib import pyplot as plt

FALSE = tf.Variable(initial_value=False, trainable=False)
TRUE = tf.Variable(initial_value=True, trainable=False)
class Trainer:

    def __init__(self, loss, prediction, optimizer, inputs, labels, train_iter, ds_train):
        '''
            Initialize the trainer

            Args:
                loss        	an operation that computes the loss
                predictions     an operation that computes the predictions for the current
                optimizer       optimizer to use
                ds_train        instance of Dataset that holds the training data
                ds_validation   instance of Dataset that holds the validation data
                stop_patience   the training stops if the validation loss does not decrease for this number of epochs
                evaluation      instance of Evaluation
                inputs          placeholder for model inputs
                labels          placeholder for model labels
        '''
        self._train_op = optimizer.minimize(loss)
        self.iter = train_iter
        self._loss = loss
        self._predictions = prediction
        self._ds_train = ds_train
        # self._ds_validation = ds_validation
        # self._stop_patience = stop_patience
        # self._evaluation = evaluation
        # self._validation_losses = []
        self._model_inputs = inputs
        self._model_labels = labels
        self._train_loss = []
        # self._validation_f1 = []

        with tf.variable_scope('model', reuse = True):
            self._model_is_training = tf.get_variable('is_training', dtype = tf.bool)


    def _train_epoch(self, sess):
        '''
            trains for one epoch and prints the mean training loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''


        # TODO
        self._model_is_training.assign(True)
        mean_loss = 0
        for i in range(self.iter):
            print("iter", i)
            data,labels = self._ds_train.next()
            _,loss_value, prediction = sess.run([self._train_op,self._loss, self._predictions],feed_dict={self._model_inputs:data,self._model_labels:labels})
            mean_loss = mean_loss + loss_value
            # print(prediction)
            # input()
        mean_loss = mean_loss/self.iter
        self._train_loss.append(mean_loss)
        print("mean loss: ",mean_loss)

        pass

    def run(self, sess, num_epochs=-1):
        '''
            run the training until num_epochs exceeds or the validation loss did not decrease
            for stop_patience epochs

            args:
                sess        the tensorflow session that should be used
                num_epochs  limit to the number of epochs, -1 means not limit
        '''

        # initial validation step
        # self._valid_step(sess)

        i = 0

        # training loop
        while i < num_epochs or num_epochs == -1:
            print("epochs:{:d}".format(i))
            self._train_epoch(sess)
            # self._valid_step(sess)
            i += 1
            # if i % 30 == 0:
            #     plot = np.round(self._train_loss.copy(), decimals=5)
            #     plt.plot(plot[1:], color="blue")
            #     plot = np.round(self._validation_losses.copy(), decimals=5)
            #     plt.plot(plot[1:], color="red")
            #     plot = np.round(self._validation_f1.copy(), decimals=5)
            #     plt.plot(plot[1:], color="green")
            #     plt.legend(['training', 'validation','validation_f1'], loc='upper left')
            #     plt.show()
            #
            # if self._should_stop():
            #     break

        print("end of run")
        print("start to plot")
        plt.plot(self._train_loss[:])
        plt.show()
