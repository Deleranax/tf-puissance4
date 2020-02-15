import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os


# Print iterations progress
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
    for i in grid[0]:
        if i == 1:
            output.append(2)
        elif i == 2:
            output.append(1)
        else:
            output.append(0)
    return np.array([output])


def test_trained_model(model, reverse=False):
    env.reset()
    while True:
        player_input = int(input("Action (0-6) : "))
        state, reward, done, infos = env.step(player_input)
        env.render()
        if done:
            break

        state = np.array(state)
        state = state.reshape((1, 42))

        if reverse:
            state = grid_reverse(state)

        prediction = np.argmax(model.predict(state))
        state, reward, done, infos = env.step(prediction)
        env.render()
        if done:
            break


def load_trained_model(path):
    return tf.keras.models.load_model(path)


# Q-Learning settings
DISCOUNT = 0.95
EPISODES = 10
STATS_EVERY = 10
SHOW_EVERY = 100
BACKUP_EVERY = 1000

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 10
END_EPSILON_DECAYING = 1000
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# For stats
ep_rewards1 = []
ep_rewards2 = []
aggr_ep_rewards = {'ep': [], 'epl': [], 'avg1': [], 'avg2': []}

env = gym.make('gym_puissance4:puissance4-v0')


def create_model1():
    model1 = tf.keras.Sequential([
        tf.keras.layers.Flatten(batch_input_shape=(1, 42)),
        tf.keras.layers.Dense(168, activation="sigmoid"),
        tf.keras.layers.Dense(7, activation="linear")
    ])

    model1.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model1


def create_model2():
    model2 = tf.keras.Sequential([
        tf.keras.layers.Flatten(batch_input_shape=(1, 42)),
        tf.keras.layers.Dense(168, activation="sigmoid"),
        tf.keras.layers.Dense(7, activation="linear")
    ])

    model2.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model2


# model1 = load_trained_model("model1/")
model1 = create_model1()

# model2 = load_trained_model("model1/")
model2 = create_model2()


d1 = datetime.datetime.today()

for episode in range(EPISODES):
    d2 = datetime.datetime.today() - d1
    eta = datetime.timedelta(seconds=((d2.total_seconds() / (episode + 1)) * EPISODES)) - d2
    print_progress_bar(episode, EPISODES, suffix="in " + str(d2)[:-7] + " ETA: " + str(eta)[:-7])
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

    last_model1_action = None
    last_model2_action = None

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


        # Update the TARGET value
        target = reward + DISCOUNT * np.max(model.predict(new_state))

        # Current TARGET value (for current state and performed action)
        current_target = model.predict(state)

        # Alter current TARGET value
        current_target[0][action] = target

        # Update the model with the TARGET values
        model.fit(state, current_target, verbose=0)

        if done:
            if reward == 1:
                reward = -1
            else:
                reward = 0.5

            if not logic:
                model = model1
                # Update the TARGET value
                target = reward + DISCOUNT * np.max(model.predict(new_state))

                # Current TARGET value (for current state and performed action)
                current_target = model.predict(state)

                # Alter current TARGET value
                current_target[0][last_model1_action] = target

                # Update the model with the TARGET values
                model.fit(state, current_target, verbose=0)

                episode_reward1 += reward
            else:
                model = model2
                # Update the TARGET value
                target = reward + DISCOUNT * np.max(model.predict(new_state))

                # Current TARGET value (for current state and performed action)
                current_target = model.predict(state)

                # Alter current TARGET value
                current_target[0][last_model2_action] = target

                # Update the model with the TARGET values
                model.fit(state, current_target, verbose=0)

                episode_reward2 += reward

        if logic:
            last_model1_action = action
        else:
            last_model2_action = action

        state = new_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards1.append(episode_reward1)
    ep_rewards2.append(episode_reward2)

    if not episode % BACKUP_EVERY:
        if not os.path.exists('model1'):
            os.makedirs('model1')

        if not os.path.exists('model2'):
            os.makedirs('model2')

        if not os.path.exists('model1/checkpoints'):
            os.makedirs('model1/checkpoints')

        if not os.path.exists('model2/checkpoints'):
            os.makedirs('model2/checkpoints')

        model1.save("model1/checkpoints/{}.h5".format(episode))
        model2.save("model2/checkpoints/{}.h5".format(episode))

    if not episode % STATS_EVERY:
        average_reward1 = sum(ep_rewards1[-STATS_EVERY:]) / STATS_EVERY
        average_reward2 = sum(ep_rewards2[-STATS_EVERY:]) / STATS_EVERY

        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['epl'].append(episode_length)

        aggr_ep_rewards['avg1'].append(average_reward1)
        aggr_ep_rewards['avg2'].append(average_reward2)

        print(f'Episode: {episode:>5d}, average of model 1: {average_reward1:>4.1f}, average of model 2: {average_reward2:>4.1f}, episode length: {episode_length:>3d}, current epsilon: {epsilon:>1.2f}')

d2 = datetime.datetime.today() - d1
print("Finished in {}.".format(str(d2)[:-7]))

# Saving models
if not os.path.exists('model1'):
    os.makedirs('model1')

if not os.path.exists('model2'):
        os.makedirs('model2')


model1.save("model1/{}.h5".format(datetime.datetime.today().strftime("%d-%m-%Y-%H-%M-%S")))
model2.save("model2/{}.h5".format(datetime.datetime.today().strftime("%d-%m-%Y-%H-%M-%S")))


env.close()  # this was already here, no need to add it again. Just here so you know where we are :)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['epl'], label="episode length")

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg1'], label="average rewards (model 1)")

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg2'], label="average rewards (model 2)")

plt.legend(loc=4)
plt.show()
