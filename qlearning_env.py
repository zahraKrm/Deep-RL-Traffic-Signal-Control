import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


class Blobup:

    def __init__(self):
        self.x = int(SIZE / 2 + 1)
        self.y = int(SIZE - 1)

    def __str__(self):
        return f'{self.x}, {self.y}'


class Blobdown:

    def __init__(self):
        self.x = int(SIZE / 2 - 1)
        self.y = int(1)

    def __str__(self):
        return f'{self.x}, {self.y}'


class Blobright:

    def __init__(self):
        self.x = int(1)
        self.y = int(SIZE / 2 + 1)

    def __str__(self):
        return f'{self.x}, {self.y}'


class Blobleft:

    def __init__(self):
        self.x = int(SIZE - 1)
        self.y = int(SIZE / 2 - 1)

    def __str__(self):
        return f'{self.x}, {self.y}'


SIZE = 15
carsup = 1
carsdown = 2
lightup = 3
lightdown = 3
LIGHT_GREEN = (0, 255, 0)
LIGHT_RED = (0, 0, 255)
light_blob_up = Blobup()
light_blob_up.x = int(SIZE / 2 + 2)
light_blob_up.y = int(SIZE / 2 + 2)
light_blob_down = Blobdown()
light_blob_down.x = int(SIZE / 2 - 2)
light_blob_down.y = int(SIZE / 2 - 2)
light_blob_right = Blobright()
light_blob_right.x = int(SIZE / 2 - 2)
light_blob_right.y = int(SIZE / 2 + 2)
light_blob_left = Blobleft()
light_blob_left.x = int(SIZE / 2 + 2)
light_blob_left.y = int(SIZE / 2 - 2)
d = {1: (255, 0, 255), 2: (255, 255, 0), 3: LIGHT_GREEN, 4: LIGHT_RED}
vertical_is_red = 1
horizontal_is_red = 0
ACTION_SPACE = 2
HM_EPISODES = 100
count = 200
epsilon = 1.0
EPS_DECAY = 0.995
DISCOUNT = 0.95
ALPHA = 0.1
SHOW_EVERY = 10
AGGREGATE_STATS_EVERY = 10
show = False

with open('x1.npy', 'rb') as f:
    x1 = np.load(f)
with open('x2.npy', 'rb') as f:
    x2 = np.load(f)
with open('x3.npy', 'rb') as f:
    x3 = np.load(f)
with open('x4.npy', 'rb') as f:
    x4 = np.load(f)


def get_q_values(state):
    if state not in q_table:
        q_table[state] = np.zeros(ACTION_SPACE)
    return q_table[state]


def actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, range_Q):
    full_flag_counter = 0
    if x1[range_Q] > 0.9 and UP_Queue <= SIZE / 2 - 2:
        car_up.append(Blobup())
    UP_Queue = 0
    for i in car_up:
        full_flag = 0
        if i.y > light_blob_up.y:
            UP_Queue += 1
        for j in car_up:
            if i != j:
                if i.y - 1 == j.y:
                    full_flag = 1
                if vertical_is_red == 1 and i.y - 1 == light_blob_up.y:
                    full_flag = 1
            if i == j:
                if vertical_is_red == 1 and i.y - 1 == light_blob_up.y:
                    full_flag = 1
        if full_flag == 1:
            full_flag_counter += 1
        if full_flag == 0:
            if i.y != 0:
                i.y -= 1
        if i.y == 0:
            car_up.remove(i)
    if show:
        for mycar in car_up:
            env[mycar.y][mycar.x] = d[carsup]
        env[light_blob_up.y][light_blob_up.x] = d[lightup]
    if x2[range_Q] > 0.9 and DOWN_Queue <= SIZE / 2 - 2:
        car_down.append(Blobdown())
    DOWN_Queue = 0
    for i in car_down:
        full_flag = 0
        if i.y < light_blob_down.y:
            DOWN_Queue += 1
        for j in car_down:
            if i != j:
                if i.y + 1 == j.y:
                    full_flag = 1
                if vertical_is_red == 1 and i.y + 1 == light_blob_down.y:
                    full_flag = 1
            if i == j:
                if vertical_is_red == 1 and i.y + 1 == light_blob_down.y:
                    full_flag = 1
        if full_flag == 1:
            full_flag_counter += 1
        if full_flag == 0:
            if i.y != SIZE - 1:
                i.y += 1
        if i.y == SIZE - 1:
            car_down.remove(i)
    if show:
        for mycar in car_down:
            env[mycar.y][mycar.x] = d[carsup]
        env[light_blob_down.y][light_blob_down.x] = d[lightup]
    if x3[range_Q] > 0.9 and RIGHT_Queue <= SIZE / 2 - 2:
        car_right.append(Blobright())
    RIGHT_Queue = 0
    for i in car_right:
        full_flag = 0
        if i.x < light_blob_right.x:
            RIGHT_Queue += 1
        for j in car_right:
            if i != j:
                if i.x + 1 == j.x:
                    full_flag = 1
                if horizontal_is_red == 1 and i.x + 1 == light_blob_right.x:
                    full_flag = 1
            if i == j:
                if horizontal_is_red == 1 and i.x + 1 == light_blob_right.x:
                    full_flag = 1
        if full_flag == 1:
            full_flag_counter += 1
        if full_flag == 0:
            if i.x != SIZE - 1:
                i.x += 1
        if i.x == SIZE - 1:
            car_right.remove(i)
    if show:
        for mycar in car_right:
            env[mycar.y][mycar.x] = d[carsdown]
        env[light_blob_right.y][light_blob_right.x] = d[lightdown]
    if x4[range_Q] > 0.9 and LEFT_Queue <= SIZE / 2 - 2:
        car_left.append(Blobleft())
    LEFT_Queue = 0
    for i in car_left:
        full_flag = 0
        if i.x > light_blob_left.x:
            LEFT_Queue += 1
        for j in car_left:
            if i != j:
                if i.x - 1 == j.x:
                    full_flag = 1
                if horizontal_is_red == 1 and i.x - 1 == light_blob_left.x:
                    full_flag = 1
            if i == j:
                if horizontal_is_red == 1 and i.x - 1 == light_blob_left.x:
                    full_flag = 1
        if full_flag == 1:
            full_flag_counter += 1
        if full_flag == 0:
            if i.x != 0:
                i.x -= 1
        if i.x == 0:
            car_left.remove(i)
    if show:
        for mycar in car_left:
            env[mycar.y][mycar.x] = d[carsdown]
        env[light_blob_left.y][light_blob_left.x] = d[lightdown]
    return UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, full_flag_counter


def reset_episode():
    global vertical_is_red, horizontal_is_red, lightup, lightdown
    global car_up, car_down, car_right, car_left
    vertical_is_red = 0
    horizontal_is_red = 1
    lightup = 3
    lightdown = 3
    car_up = []
    car_down = []
    car_right = []
    car_left = []
    return 0, 0, 0, 0, -5, 0


car_up = []
car_down = []
car_right = []
car_left = []
q_table = {}
episode_rewards = []
env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

for episode in range(HM_EPISODES):
    print(episode)
    if episode % SHOW_EVERY == 0:
        print(f'on #{episode}, epsilon is {epsilon}')
        if episode_rewards:
            print(f'{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}')
    UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, last_action, last_situation = reset_episode()
    episode_reward = 0
    for i in range(count):
        range_Q = episode * count + i
        if range_Q >= len(x1):
            break
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        obs = (UP_Queue + DOWN_Queue, RIGHT_Queue + LEFT_Queue)
        if np.random.random() > epsilon:
            action = int(np.argmax(get_q_values(obs)))
        else:
            action = random.randint(0, ACTION_SPACE - 1)
        if action == 0:
            horizontal_is_red = 1
            if last_action != action:
                for index1 in range(3):
                    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                    UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, full_flag_counter = actionfunc(
                        UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, range_Q
                    )
            vertical_is_red = 0
            lightup = 3
            lightdown = 4
        elif action == 1:
            vertical_is_red = 1
            if last_action != action:
                for index1 in range(3):
                    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                    UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, full_flag_counter = actionfunc(
                        UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, range_Q
                    )
            horizontal_is_red = 0
            lightup = 4
            lightdown = 3
        last_action = action
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, full_flag_counter = actionfunc(
            UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, range_Q
        )
        current_situation = full_flag_counter
        if current_situation <= last_situation:
            if UP_Queue >= SIZE / 4:
                reward = -1
            elif DOWN_Queue >= SIZE / 4:
                reward = -1
            elif RIGHT_Queue >= SIZE / 4:
                reward = -1
            elif LEFT_Queue >= SIZE / 4:
                reward = -1
            else:
                reward = 0
        else:
            reward = -2
        new_obs = (UP_Queue + DOWN_Queue, RIGHT_Queue + LEFT_Queue)
        done = i == count - 1
        current_q = get_q_values(obs)
        if done:
            target = reward
        else:
            target = reward + DISCOUNT * np.max(get_q_values(new_obs))
        current_q[action] = (1 - ALPHA) * current_q[action] + ALPHA * target
        last_situation = current_situation
        episode_reward += reward
    episode_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = np.mean(episode_rewards[-AGGREGATE_STATS_EVERY:])
        print(f'reward avg: {average_reward:.2f}, epsilon: {epsilon:.4f}, states: {len(q_table)}')
    epsilon *= EPS_DECAY

with open('qtable-same-0.9.pickle', 'wb') as f:
    pickle.dump(q_table, f)
with open('QLearning_env_episode_rewards.npy', 'wb') as f:
    np.save(f, episode_rewards)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Reward {SHOW_EVERY}ma')
plt.xlabel('episode #')
plt.show()
plt.plot(episode_rewards)
plt.xlabel('episode #')
plt.ylabel('Reward')
plt.show()
print(f'Saved Q-table with {len(q_table)} states to qtable-same-0.9.pickle')
