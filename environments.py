"""Environment creation independent of learning algorithms."""
import gymnasium as gym


def make_env(name, *, slippery=True, max_steps=100):
    if name not in ('FrozenLake-v1', 'CliffWalking-v1'):
        raise ValueError('Unsupported environment')
    kwargs = {'is_slippery': slippery} if name == 'FrozenLake-v1' else {}
    return gym.make(name, max_episode_steps=max_steps, **kwargs)

