import gym
import time
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

def grid_reverse(grid):
    output = []
    for y in grid:
        list = []
        for x in y:
            if x == 1:
                list.append(2)
            elif x == 2:
                list.append(1)
            else:
                list.append(0)
        output.append(list)
    return output


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='', printEnd="\r"):
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
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ">" + '.' * (length - filledLength)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def test_trained_model():
    env.reset()
    while True:
        player_input = int(input("Action (0-6) : "))
        grille, reward, done, infos = env.step(player_input)
        env.render()
        if done:
            break
        prediction = np.argmax(q_table[state]) % 6
        grille, reward, done, infos = env.step(prediction)
        env.render()
        if done:
            break


def lts(s):
    # initialize an empty string
    str1 = " "

    # return string
    return str1.join(str(elem) for elem in s)

env = gym.make('gym_puissance4:puissance4-v0')

# Q-Learning settings
LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 5000
STATS_EVERY = 100
DISCRETE_SIZE = [3] * 6 * 7
TABLE_SHAPE = DISCRETE_SIZE + [env.action_space.n]
TABLE_SIZE = (3 ^ (6 * 7)) * 7
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

q_table = {}
for i in range(3 ** (6 * 7)):
    printProgressBar(i, 3 ** (6 * 7))
    print(lts([(i // 3**x) % 3 for x in range(42)]))
    q_table[lts([(i // 3**x) % 3 for x in range(42)])] = np.random.uniform(low=0, high=6, size=3)

#q_table = np.random.uniform(low=0, high=6, size=TABLE_SIZE)

for episode in range(EPISODES):
    episode_reward = 0
    state = np.reshape(env.reset()[1], [-1])
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    done = False
    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[lts(state)])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if new_state[0] == 2:
            grille = grid_reverse(new_state[1])
        else:
            new_state = new_state[1]

        if render:
            env.render()
            time.sleep(0.2)

        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[lts(np.reshape(new_state, [-1]))])

            # Current Q value (for current state and performed action)
            current_q = q_table[lts(np.reshape(state, [-1]))][action]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[lts(np.reshape(state, [-1]))][action] = new_q

        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        #elif reward >= 1:
        #   q_table[state + (action,)] = reward*6

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

    # AT THE END
    if episode % 50 == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)

env.close()  # this was already here, no need to add it again. Just here so you know where we are :)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()