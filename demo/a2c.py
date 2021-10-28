"""
@创建日期 ：2021/9/23
@修改日期 ：2021/9/23
@作者 ：jzj
@功能 ：https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
    1、Run the agent on the environment to collect training data per episode.
    2、Compute expected return at each time step.
    3、Compute the loss for the combined actor-critic model.
    4、Compute gradients and update network parameters.
    5、Repeat 1-4 until either success criterion or max episodes has been reached.
"""

import gym
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from typing import Tuple
import collections
import tqdm
import statistics

SEED = 42

env = gym.make("CartPole-v0")
env.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

eps = np.finfo(np.float32).eps
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


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


def env_step(action):
    """
    采集数据
    """
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action):
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


def run_episode(init_state, model, max_steps):
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = init_state.shape
    state = init_state
    for t in tf.range(max_steps):
        state = tf.expand_dims(state, 0)
        action_logits_t, value = model(state)

        # 基于action_logits softmax结果的的概率采样
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # 存储critic values
        values = values.write(t, tf.squeeze(value))

        # 存储action的log概率
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # 执行行动
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # 存储reward
        rewards = rewards.write(t, reward)
        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    return action_probs, values, rewards


def get_expected_return(rewards, gamma, standardize=True):
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape

    for i in range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)

    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

    return returns


def compute_loss(action_probs, values, returns):
    """计算actor-critic loss"""
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    critic_loss = huber_loss(values, returns)
    return (actor_loss + critic_loss)/10


@tf.function
def train_step(init_state, model, optimizer, gamma, max_steps_per_episode):
    with tf.GradientTape() as tape:

        action_probs, values, rewards = run_episode(init_state, model, max_steps_per_episode)

        returns = get_expected_return(rewards, gamma)

        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        loss = compute_loss(action_probs, values, returns)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)
    return episode_reward


def demo():
    min_episodes_criterion = 100
    max_episodes = 10000
    max_steps_per_episode = 1000

    reward_threshold = 180
    running_reward = 0

    gamma = 0.99

    episodes_reward = collections.deque(maxlen=min_episodes_criterion)
    model = ActorCritic(num_action=env.action_space.n, num_hidden_units=128)

    with tqdm.trange(max_episodes) as t:
        for i in t:
            init_state = tf.constant(env.reset(), dtype=tf.float32)
            temp_reward = train_step(init_state, model, optimizer, gamma, max_steps_per_episode)

            episode_reward = int(temp_reward)

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

            if i % 10 == 0:
                pass
                # print(f"Episode {i} : average reward: {av}")
            if running_reward > reward_threshold and i >= min_episodes_criterion:
                break
    print("finish")


if __name__ == '__main__':
    demo()
