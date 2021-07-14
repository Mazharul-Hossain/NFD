import logging
import multiprocessing
import os
import random
import time
import copy
from collections import deque, defaultdict
from datetime import datetime
from statistics import mean
from typing import List

import numpy as np
import tensorflow as tf
from DeepQNetwork import DeepQNetwork

logger = None

flags = tf.app.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 50000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 5000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 32, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_integer('training_interval', 60, 'Training Interval. (after how long training will happen)')
flags.DEFINE_string('model_name', 'IFS_RL_ConvNetv1', 'DeepLearning Network Model name (IFS_RL_ConvNetv1)')
flags.DEFINE_float('learning_rate', 0.0001, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_boolean('step_verbose', False, 'verbose every step count')
flags.DEFINE_integer('step_verbose_count', 100, 'verbose step count')
flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')
tf.flags.DEFINE_string('log_dir', '/tmp/minindn/', 'Root directory to raw image dataset.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

NO_INFORMATION = "NO_INFORMATION"
READY_FOR_CALCULATION = "READY_FOR_CALCULATION"
RESULT_READY = "RESULT_READY"

consecutive_len = 100  # training input length
exploration_rate = 0.2  # probability of choosing to explore


class ModelMain:
    def __init__(self):
        logging.basicConfig(filename=os.path.join(FLAGS.log_dir, 'log_tf_IFS_RL_{}.txt'.format(os.getpid())),
                            level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
        global logger
        logger = logging.getLogger()

        # logger.info("FLAGS configure.")
        # logger.info(FLAGS.__flags)

        # store the previous observations in replay memory
        self.replay_buffer = deque(maxlen=FLAGS.replay_memory_length)
        self.last_n_game_reward = deque(maxlen=consecutive_len)

        # ========================================================
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.main_DQN = DeepQNetwork(
            self.sess, FLAGS.model_name, 128, 48,
            learning_rate=FLAGS.learning_rate, name="main")

        self.target_DQN = DeepQNetwork(
            self.sess, FLAGS.model_name, 128, 48, name="target")

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

        self.training_process = multiprocessing.Process(target=self.process_episode_training, args=())
        self.global_step = 0
        # ========================================================

        self.all_rtt_reward = defaultdict(list)
        self.status_list = defaultdict(lambda: NO_INFORMATION)
        self.best_prefix_face = {}

        # state = [states, actions, rewards]
        self.state = {}
        self.next_state = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

        self.ordered_face_list = []

    def get_prefix_face_status(self, name_prefix):
        return self.status_list[name_prefix]

    def get_prefix_face_result(self, name_prefix):
        if name_prefix not in self.best_prefix_face:
            return None
        return self.best_prefix_face[name_prefix]

    def face_info_for_ranking(self, name_prefix, face, last_rtt, s_rtt, cost, is_passing_null):
        message = "[{}: {} {}, ".format('name_prefix', type(name_prefix), str(name_prefix))
        message += "{}: {} {}, ".format('face', type(face), str(face))
        message += "{}: {} {}, ".format('last_rtt', type(last_rtt), str(last_rtt))
        message += "{}: {} {}, ".format('s_rtt', type(s_rtt), s_rtt)
        message += "{}: {} {} {}]".format('cost', type(cost), str(cost), is_passing_null)
        logger.info(message)

        self.next_state[name_prefix][face][0] = last_rtt - s_rtt

    def calculate_prefix_face_result(self, name_prefix):
        all_faces = self.next_state[name_prefix].keys()

        if np.random.rand() < exploration_rate:
            action = random.choice(all_faces)
        else:
            state = []
            for face in all_faces:
                state.append(self.next_state[name_prefix][face])

            # Choose an action by greedily from the Q-network
            action = np.argmax(self.main_DQN.predict(state))
            action = all_faces[action]

        self.state[name_prefix] = copy.deepcopy(self.next_state[name_prefix])
        for face in all_faces:
            self.next_state[name_prefix][face] = [0.0, 0]

        self.best_prefix_face[name_prefix] = action
        self.status_list[name_prefix] = RESULT_READY

        return action

    def replay_train(self, main_dqn, target_dqn, train_batch):
        """Trains `mainDQN` with target Q values given by `targetDQN`
        Args:
            main_dqn (DeepQNetwork``): Main DQN that will be trained
            target_dqn (DeepQNetwork): Target DQN that will predict Q_target
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

        predict_result = target_dqn.predict(next_states)
        Q_target = rewards + FLAGS.discount_rate * np.max(predict_result, axis=1)

        X = states
        y = main_dqn.predict(states)
        y[np.arange(len(X)), actions] = Q_target

        # Train our network using target and predicted Q values on each episode
        return main_dqn.update(X, y)

    def get_copy_var_ops(self, dest_scope_name, src_scope_name):
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

    def process_episode_training(self):
        # initial copy q_net -> target_net
        copy_ops = self.get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        self.sess.run(copy_ops)
        step_count = 0
        while True:
            time.sleep(FLAGS.training_interval)

            # =================== Data Creation ======================
            # state, action, reward, next_state
            all_name_prefix = self.state.keys()
            for name_prefix in all_name_prefix:
                all_face_state = self.state[name_prefix]
                state, next_state = [], []

                for face in self.ordered_face_list:
                    state.append(self.state[name_prefix][face])
                    next_state.append(self.next_state[name_prefix][face])

                ordered_face_set = set(self.ordered_face_list)
                all_faces = all_face_state.keys()
                for face in all_faces:
                    if face in ordered_face_set:
                        continue
                    self.ordered_face_list.append(face)
                    ordered_face_set.add(face)

                    state.append(self.state[name_prefix][face])
                    next_state.append(self.next_state[name_prefix][face])
                del self.state[name_prefix]
                action = self.best_prefix_face[name_prefix]
                reward = self.calculate_reward(name_prefix)

                self.replay_buffer.append((state, action, reward, next_state))

            # =================== training section ===================
            if len(self.replay_buffer) > FLAGS.batch_size:
                minibatch = random.sample(self.replay_buffer, (FLAGS.batch_size))
                loss, _ = self.replay_train(self.main_DQN, self.target_DQN, minibatch)
                model_loss = loss

                if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
                    logger.info(" - step_count : {}, loss: {}".format(step_count, loss))

            if step_count % FLAGS.target_update_count == 0:
                self.sess.run(copy_ops)

            step_count += 1
            all_name_prefix = self.status_list.keys()
            all_name_prefix_next = self.next_state.keys()

            for key in all_name_prefix:
                self.status_list[key] = READY_FOR_CALCULATION
                if key in all_name_prefix_next:
                    all_name_prefix_next.remove(key)

            for key in all_name_prefix_next:
                self.status_list[key] = READY_FOR_CALCULATION

            # save model checkpoint
            if self.global_step % FLAGS.save_step_count == 0:
                checkpoint_path = FLAGS.gym_env + "_f" + str(
                    FLAGS.frame_size) + "_" + FLAGS.checkpoint_path + "global_step"
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)

                self.saver.save(self.sess, checkpoint_path, global_step=self.global_step)
                logger.info("save model for global_step: {}".format(self.global_step))

            self.global_step += 1

    def calculate_reward(self, name_prefix):
        reward = -1 * mean(self.all_rtt_reward[name_prefix])
        self.all_rtt_reward[name_prefix] = []
        return reward

    def send_face_forwarding_metrics(self, name_prefix, face, last_rtt):
        self.next_state[name_prefix][face][1] += 1
        self.all_rtt_reward[name_prefix].append(last_rtt)
