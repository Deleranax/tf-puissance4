import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Print iterations progress
from docutils.io import InputError


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='=', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + ">" + '.' * (length - filled_length)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def grid_reverse(grid):
    output = []
    for i in grid:
        if i == 1:
            output.append(2)
        elif i == 2:
            output.append(1)
        else:
            output.append(0)
    return output

def test_trained_model(model):
    env.reset()
    while True:
        player_input = int(input("Action (0-6) : "))
        grille, reward, done, infos = env.step(player_input)
        env.render()
        if done:
            break
        prediction = np.argmax(model.predict(state)[0])
        grille, reward, done, infos = env.step(prediction)
        env.render()
        if done:
            break

# Q-Learning settings
DISCOUNT = 0.95
EPISODES = 5000
STATS_EVERY = 10
SHOW_EVERY = 1000

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 10
END_EPSILON_DECAYING = EPISODES//1.2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# For stats
ep_rewards1 = []
ep_rewards2 = []
aggr_ep_rewards = {'ep': [], 'epl': [], 'avg1': [], 'max1': [], 'min1': [], 'avg2': [], 'max2': [], 'min2': []}

env = gym.make('gym_puissance4:puissance4-v0')

model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(batch_input_shape=(1, 42)),
    tf.keras.layers.Dense(168, activation="sigmoid"),
    tf.keras.layers.Dense(7, activation="linear")
])

model1.compile(loss='mse', optimizer='adam', metrics=['mae'])

model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(batch_input_shape=(1, 42)),
    tf.keras.layers.Dense(168, activation="sigmoid"),
    tf.keras.layers.Dense(7, activation="linear")
])

model2.compile(loss='mse', optimizer='adam', metrics=['mae'])

for episode in range(EPISODES):
    print_progress_bar(episode, EPISODES)
    episode_reward1 = 0
    episode_reward2 = 0
    episode_length = 0
    state = np.array(env.reset())
    state = state.reshape((1, 42))
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    done = False
    logic = True

    while not done:
        episode_length += 1

        model = None
        if logic:
            logic = False
            model = model1
        else:
            logic = True
            model = model2

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(model.predict(state)[0])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        new_state = np.array(new_state)
        new_state = new_state.reshape((1, 42))

        if logic:
            episode_reward1 += reward
        else:
            episode_reward2 += reward

        if render:
            env.render()
            time.sleep(0.2)

        if not done:
            # Update the TARGET value
            target = reward + DISCOUNT * np.max(model.predict(new_state))

            # Current TARGET value (for current state and performed action)
            current_target = model.predict(state)

            # Alter current TARGET value
            current_target[0][action] = target

            # Update the model with the TARGET values
            model.fit(state, current_target, verbose=0)

        state = new_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards1.append(episode_reward1)
    ep_rewards2.append(episode_reward2)

    if not episode % STATS_EVERY:

        average_reward1 = sum(ep_rewards1[-STATS_EVERY:])/STATS_EVERY
        average_reward2 = sum(ep_rewards2[-STATS_EVERY:]) / STATS_EVERY

        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['epl'].append(episode_length)

        aggr_ep_rewards['avg1'].append(average_reward1)
        aggr_ep_rewards['max1'].append(max(ep_rewards1[-STATS_EVERY:]))
        aggr_ep_rewards['min1'].append(min(ep_rewards1[-STATS_EVERY:]))

        aggr_ep_rewards['avg2'].append(average_reward2)
        aggr_ep_rewards['max2'].append(max(ep_rewards2[-STATS_EVERY:]))
        aggr_ep_rewards['min2'].append(min(ep_rewards2[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, avg1: {average_reward1:>4.1f}, avg2: {average_reward2:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close()  # this was already here, no need to add it again. Just here so you know where we are :)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['epl'], label="episode length")

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg1'], label="average rewards (model 1)")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max1'], label="max rewards (model 1)")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min1'], label="min rewards (model 1)")

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg2'], label="average rewards (model 2)")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max2'], label="max rewards (model 2)")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min2'], label="min rewards (model 2)")

plt.legend(loc=4)
plt.show()
