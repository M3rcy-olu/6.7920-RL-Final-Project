"""Tabular online accumulating TD(lambda), independent of Gymnasium."""


def td_update(values, traces, state, reward, next_state, terminated,
              *, alpha, gamma, lam):
    """Online accumulating traces. Bootstrap at truncation, not termination."""
    target = reward + (0.0 if terminated else gamma * values[next_state])
    delta = target - values[state]
    traces[state] += 1.0
    values += alpha * delta * traces
    traces *= gamma * lam

