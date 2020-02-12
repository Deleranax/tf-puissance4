import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='', printEnd="\r"):
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



# Q-Learning settings
DISCOUNT = 0.95
EPISODES = 5000
STATS_EVERY = 100
MAX_VALUE = 2
MIN_VALUE = 0
SHOW_EVERY = 1000

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 5
END_EPSILON_DECAYING = EPISODES//1
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

env = gym.make('gym_puissance4:puissance4-v0')

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(32, 6, 7)),
    tf.keras.layers.Dense(168, activation="sigmoid"),
    tf.keras.layers.Dense(7, activation="linear")
])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

for episode in range(EPISODES):
    episode_reward = 0
    state = np.array(env.reset())
    state = state.reshape((1, 6, 7))
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    done = False
    logic = False
    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(model.predict(state))
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        new_state = np.array(new_state)
        new_state = new_state.reshape((1, 6, 7))

        episode_reward += reward

        if logic:
            new_state = grid_reverse(new_state)
            logic = not logic

        if render:
            env.render()
            time.sleep(0.2)

        if not done:
            # Update the TARGET value
            target = reward + DISCOUNT * np.max(model.predict(new_state))

            # Current TARGET value (for current state and performed action)
            current_target = model.predict(state)

            # Alter current TARGET value
            current_target[action] = target

            # Update the model with the TARGET values
            model.fit(state, current_target)

        state = new_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close()  # this was already here, no need to add it again. Just here so you know where we are :)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()

def test_trained_model():
    env.reset()
    while True:
        player_input = int(input("Action (0-6) : "))
        grille, reward, done, infos = env.step(player_input)
        env.render()
        if done:
            break
        prediction = np.argmax(model.predict(state))
        grille, reward, done, infos = env.step(prediction)
        env.render()
        if done:
            break
