import tensorflow as tf
import numpy as np

from IFS_RL_ConvNet import IFS_RL_ConvNetv1


class DeepQNetwork:

    def __init__(self, session: tf.Session, model_name: str, input_size: int = 128, num_classes: int = 48,
                 learning_rate: float = 0.0001, name: str = "main") -> None:
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            num_classes (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.num_classes = num_classes

        self.net_name = name
        self.learning_rate = learning_rate

        self._build_network(model_name=model_name)

    def _build_network(self, model_name) -> None:
        with tf.variable_scope(self.net_name):
            X_shape = [None, self.input_size]
            self._X = tf.placeholder(tf.float32, X_shape, name="input_x")

            models = {
                "IFS_RL_ConvNetv1": IFS_RL_ConvNetv1,
            }

            model = models[model_name](self._X, self.num_classes, learning_rate=self.learning_rate)
            model.build_network()

            self._Qpred = model.inference
            self._Y = model.Y
            self._loss = model.loss
            self._train = model.optimizer

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        x_shape = [-1, self.input_size]
        x = np.reshape(state, x_shape)
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)
