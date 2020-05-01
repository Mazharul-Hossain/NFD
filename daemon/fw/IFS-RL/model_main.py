import logging
import os

from datetime import datetime

import tensorflow as tf

import logging
import os
import random
from collections import deque
from datetime import datetime
from typing import List

import numpy as np
import tensorflow as tf

import DeepQNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

flags = tf.app.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 50000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 5000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 64, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_integer('frame_size', 1, 'Frame size. (Stack env\'s observation T-n ~ T)')
flags.DEFINE_string('model_name', 'OFS_RL_ConvNetv1', 'DeepLearning Network Model name (MLPv1, ConvNetv1)')
flags.DEFINE_float('learning_rate', 0.0001, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_string('gym_result_dir', 'gym-results/', 'Directory to put the gym results.')
flags.DEFINE_string('gym_env', 'CartPole-v0',
                    'Name of Open Gym\'s enviroment name. (CartPole-v0, CartPole-v1, MountainCar-v0)')
flags.DEFINE_boolean('step_verbose', False, 'verbose every step count')
flags.DEFINE_integer('step_verbose_count', 100, 'verbose step count')
flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')
tf.flags.DEFINE_string('log_dir', '.', 'Root directory to raw image dataset.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)


class ModelMain:
    def __init__(self):
        logging.basicConfig(filename=os.path.join(FLAGS.log_dir, 'log_tf_IFS_RL_{}.txt'.format(os.getpid())),
                            level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')
        self.log_file = open(os.path.join(FLAGS.log_dir, 'log_my_IFS_RL_{}.txt'.format(os.getpid())), 'a')

        self.log_to_file("Class created successfully")

    def log_to_file(self, message):
        TimeFormat = "%d-%b-%Y (%H:%M:%S)"
        timestamp_str = datetime.now().strftime(TimeFormat)

        print("{} Process#{}: {}".format(
            timestamp_str, os.getpid(), message, file=self.log_file, flush=True))

    def face_info_for_ranking(self, face, last_rtt, s_rtt, cost, is_passing_null):
        message = type(face) + face + type(last_rtt) + last_rtt + type(s_rtt) + s_rtt + type(
            cost) + cost + is_passing_null
        self.log_to_file(message)

    def replay_train(self, mainDQN: DeepQNetwork, targetDQN: DeepQNetwork, train_batch: list) -> float:
        """Trains `mainDQN` with target Q values given by `targetDQN`
        Args:
            mainDQN (DeepQNetwork``): Main DQN that will be trained
            targetDQN (DeepQNetwork): Target DQN that will predict Q_target
            train_batch (list): Minibatch of replay memory
                Each element is (s, a, r, s', done)
                [(state, action, reward, next_state, done), ...]
        Returns:
            float: After updating `mainDQN`, it returns a `loss`
        """
        states = np.vstack([x[0] for x in train_batch])
        actions = np.array([x[1] for x in train_batch[:FLAGS.batch_size]])
        rewards = np.array([x[2] for x in train_batch[:FLAGS.batch_size]])
        next_states = np.vstack([x[3] for x in train_batch])
        done = np.array([x[4] for x in train_batch[:FLAGS.batch_size]])

        predict_result = targetDQN.predict(next_states)
        Q_target = rewards + FLAGS.discount_rate * np.max(predict_result, axis=1) * (1 - done)

        X = states
        y = mainDQN.predict(states)
        y[np.arange(len(X)), actions] = Q_target

        # Train our network using target and predicted Q values on each episode
        return mainDQN.update(X, y)

    def get_copy_var_ops(self, *, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
        """Creates TF operations that copy weights from `src_scope` to `dest_scope`
        Args:
            dest_scope_name (str): Destination weights (copy to)
            src_scope_name (str): Source weight (copy from)
        Returns:
            List[tf.Operation]: Update operations are created and returned
        """
        # Copy variables src_scope to dest_scope
        op_holder = []

        src_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder
