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

FLAGS = flags.FLAGS


def replay_train(mainDQN: DeepQNetwork, targetDQN: DeepQNetwork, train_batch: list) -> float:
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


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
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


def main(unused_argv):
    logger.info("FLAGS configure.")
    logger.info(FLAGS.__flags)

    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=FLAGS.replay_memory_length)

    consecutive_len = 100  # default value
    # if config.solving_criteria:
    #     consecutive_len = criteria.solving_criteria[0]
    last_n_game_reward = deque(maxlen=consecutive_len)

    tf.reset_default_graph()
    with tf.Session() as sess:
        mainDQN = DeepQNetwork(
            sess, FLAGS.model_name, config.input_size, config.output_size,
            learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main")

        targetDQN = DeepQNetwork(
            sess, FLAGS.model_name, config.input_size, config.output_size,
            frame_size=FLAGS.frame_size, name="target")

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        global_step = 1
        for episode in range(FLAGS.max_episode_count):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            state = env.reset()

            e_reward = 0
            model_loss = 0
            avg_reward = np.mean(last_n_game_reward)

            if FLAGS.frame_size > 1:
                state_with_frame = deque(maxlen=FLAGS.frame_size)

                for _ in range(FLAGS.frame_size):
                    state_with_frame.append(state)

                state = np.array(state_with_frame)
                state = np.reshape(state, (1, config.RAM_FIXED_LENGTH, FLAGS.frame_size))

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                if done:  # Penalty
                    reward = -1

                if FLAGS.frame_size > 1:
                    state_with_frame.append(next_state)

                    next_state = np.array(state_with_frame)
                    next_state = np.reshape(next_state, (1, config.RAM_FIXED_LENGTH, FLAGS.frame_size))

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > FLAGS.batch_size:
                    minibatch = random.sample(replay_buffer, (FLAGS.batch_size))
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    model_loss = loss

                    if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
                        logger.info(f" - step_count : {step_count}, reward: {e_reward} loss: {loss}")

                if step_count % FLAGS.target_update_count == 0:
                    sess.run(copy_ops)

                state = next_state
                e_reward += reward
                step_count += 1

                # save model checkpoint
                if global_step % FLAGS.save_step_count == 0:
                    checkpoint_path = FLAGS.gym_env + "_f" + str(
                        FLAGS.frame_size) + "_" + FLAGS.checkpoint_path + "global_step"
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)

                    saver.save(sess, checkpoint_path, global_step=global_step)
                    logger.info(f"save model for global_step: {global_step} ")

                global_step += 1

            logger.info(
                "Episode: {episode}  reward: {e_reward}  loss: {model_loss}  consecutive_{"
                "consecutive_len}_avg_reward: {avg_reward}")

            # CartPole-v0 Game Clear Checking Logic
            last_n_game_reward.append(e_reward)

            if len(last_n_game_reward) == last_n_game_reward.maxlen:
                avg_reward = np.mean(last_n_game_reward)

                if config.solving_criteria and avg_reward > (config.solving_criteria[1]):
                    logger.info(f"Game Cleared in {episode} episodes with avg reward {avg_reward}")
                    break





if __name__ == '__main__':
    tf.app.run()
