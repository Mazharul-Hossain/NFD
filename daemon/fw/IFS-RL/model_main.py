import logging
import os

from datetime import datetime

import tensorflow as tf

flags = tf.app.flags
tf.flags.DEFINE_string('log_dir', '.', 'Root directory to raw image dataset.')
FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)


class ModelMain():
    def __init__(self, ):
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
