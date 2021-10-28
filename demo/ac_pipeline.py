"""
@创建日期 ：2021/10/27
@修改日期 ：2021/10/27
@作者 ：jzj
@功能 ：ac 的一个TrainPipeline版本
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
from typing import Tuple
import collections
import tqdm
import statistics

SEED = 37
env = gym.make("CartPole-v0")
env.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


eps = np.finfo(np.float).eps


def env_step(action):
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action):
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


# def tf_env_step_v2(env, action):
#     state, reward, done, _ = env.step(action.numpy())


class ActorCritic(tf.keras.Model):
    def __init__(self, num_action: int, num_hidden_units: int):
        super(ActorCritic, self).__init__()
        self.common = layers.Dense(num_hidden_units, activation=None)
        self.activation = layers.ReLU()
        self.actor = layers.Dense(num_action)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        x = self.activation(x)
        return self.actor(x), self.critic(x)

    def build(self, input_shape):
        super(ActorCritic, self).build(input_shape)


class TrainPipeline():
    def __init__(self):
        self.env = env
        self.model = ActorCritic(num_action=self.env.action_space.n, num_hidden_units=128)
        # self.model.build(input_shape=(1, 4))
        self.optimizer = optimizers.Adam(learning_rate=1e-2)
        self.max_steps = 1000
        self.max_episodes = 10000
        self.reward_threshold = 195
        self.min_episode_criterion = 100

    def run_episode(self, init_state):
        # init_state = tf.constant(self.env.reset(), dtype=tf.float32)

        action_probs = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        values = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        rewards = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True)

        init_state_shape = init_state.shape
        state = init_state
        for t in tf.range(self.max_steps):
            state = tf.expand_dims(state, 0)
            action_logit_t, value = self.model(state)

            action = tf.random.categorical(action_logit_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logit_t)

            values = values.write(t, tf.squeeze(value))
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # 执行
            state, reward, done = tf_env_step(action)
            state.set_shape(init_state_shape)

            rewards = rewards.write(t, reward)
            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expect_returns(self, rewards, gamma=0.99, standardize=True):
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = tf.cast(rewards[::-1], tf.float32)

        discounted_sum = tf.constant(0.)
        discounted_sum_shape = discounted_sum.shape
        for i in range(n):
            reward = rewards[i]
            discounted_sum = reward * gamma + discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1]

        if standardize:
            # tf.print("tensor: ", tf.math.reduce_std(returns))
            returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)

        return returns

    def compute_loss(self, action_probs, values, returns):
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)

        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        huber_loss = losses.Huber(reduction=losses.Reduction.SUM)
        critic_loss = huber_loss(values, returns)
        return actor_loss + critic_loss

    @tf.function
    def train_step(self, init_state):
        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.run_episode(init_state)

            returns = self.get_expect_returns(rewards)

            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            loss = self.compute_loss(action_probs, values, returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward

    def train(self):
        episodes_reward = collections.deque(maxlen=self.min_episode_criterion)

        with tqdm.trange(self.max_episodes) as t:
            for i in t:
                init_state = tf.constant(env.reset(), tf.float32)
                episode_reward = int(self.train_step(init_state))
                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                t.set_description(f"Episode {i}")
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                if running_reward > self.reward_threshold and i > self.min_episode_criterion:
                    break
        print("finish")


def demo():
    pipeline = TrainPipeline()
    pipeline.train()


if __name__ == '__main__':
    demo()
