"""
@创建日期 ：2021/10/26
@修改日期 ：2021/10/26
@作者 ：jzj
@功能 ：
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, losses, optimizers
from typing import Tuple
import collections
import tqdm
import statistics



def softmax(x):
    probs = np.exp(x - np.max(x))
    probs = probs / np.sum(probs)
    return probs


def huber_loss(values, returns, delta=1.):
    advantage = returns - values
    res = []
    for i in advantage:
        if -delta < i < delta:
            res.append(1/2 * i ** 2)
        else:
            res.append(abs(i))
    return np.sum(res, dtype=np.float32)


def get_expected_returns(rewards, gamma, standardize=True, eps=1e-7):
    """
    计算期望收益
    """
    n = rewards.shape[0]
    returns = []
    discounted_sum = 0
    for i in range(n):
        temp_reward = rewards[n-(i+1)]
        discounted_sum = temp_reward + gamma * discounted_sum
        returns.append(discounted_sum)
    returns = np.stack(returns[::-1])

    if standardize:
        returns = (returns - np.mean(returns)) / (np.sqrt(returns) + eps)

    return returns.astype(np.float32)


# @tf.function
# def compute_loss(action_probs, values, returns):
#     advantage = returns - values
#
#     action_log_probs = np.log(action_probs)
#     actor_loss = -np.sum(action_log_probs * advantage)
#     critic_loss = huber_loss(values, returns)
#     return actor_loss + critic_loss


def compute_loss(action_probs, values, returns):
    """计算actor-critic loss"""
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    critic_loss = huber_loss(values, returns)
    # print("loss:  ", actor_loss + critic_loss)
    # tf.print(actor_loss, critic_loss)
    return 0.5 * (actor_loss + critic_loss)


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


class Pipeline:
    def __init__(
            self,
            env,
            max_steps,
            gamma=0.99,
            eps=1e-7):
        self.env = env
        self.max_steps = max_steps
        self.gamma = gamma
        self.eps = eps
        self.policy_value_fn = ActorCritic(env.action_space.n, 128)
        self.optimizer = optimizers.Adam(1e-3)

    def run_episode(self):
        action_probs = []
        values = []
        rewards = []

        init_state = self.env.reset()
        state = init_state
        for i in range(self.max_steps):
            state = np.expand_dims(state, 0)
            action_logits_t, value = self.policy_value_fn(state)

            # 使用action_logits_t的重要度采样
            action_probs_t = softmax(action_logits_t)
            action = np.random.choice(range(self.env.action_space.n), p=action_probs_t[0])
            state, reward, done, _ = self.env.step(action)
            state, reward, done = state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)

            # 存储critic value， action_prob, reward
            values.append(np.squeeze(value))
            action_probs.append(action_probs_t[0, action])
            rewards.append(reward)
            if np.bool(done):
                break

        action_probs = np.stack(action_probs)
        values = np.stack(values)
        rewards = np.stack(rewards)
        return action_probs, values, rewards

    def train_step(self):
        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.run_episode()
            returns = get_expected_returns(rewards, self.gamma, standardize=True, eps=self.eps)
            loss = compute_loss(action_probs, values, returns)

        grads = tape.gradient(loss, self.policy_value_fn.trainable_variables)
        print(grads)
        self.optimizer.apply_gradients(zip(grads, self.policy_value_fn.trainable_variables))
        return np.sum(rewards)


def demo():
    SEED = 42
    env = gym.make("CartPole-v0")
    env.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    eps = np.finfo(np.float32).eps
    max_step_per_episode = 1000
    max_episodes = 10000
    min_episodes_criterion = 100
    running_threshold = 195

    episodes_reward = collections.deque(maxlen=min_episodes_criterion)

    pipeline = Pipeline(env, max_step_per_episode, eps=eps)

    with tqdm.trange(max_episodes) as t:

        for i in t:
            episode_reward = pipeline.train_step()
            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f"Episode {i}")
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

            if running_reward > running_threshold and i >= min_episodes_criterion:
                break
    print("finish")


def function_test():
    env = gym.make("CartPole-v0")
    model = ActorCritic(2, 128)
    state = env.reset()
    state = np.expand_dims(state, 0)

    action_logits_t, value = model(state)
    print(action_logits_t, value)


if __name__ == '__main__':
    demo()
    # function_test()
