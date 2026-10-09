import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras.optimizers import Adam

logging.getLogger('tensorflow').setLevel(logging.ERROR)
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
SIZE = 20
CAR_SIZE = 35
GAP_SIZE = 2
CAR_PIC = cv2.imread('Car.JPG')
CAR_PIC = cv2.resize(CAR_PIC, dsize=(CAR_SIZE, CAR_SIZE), interpolation=cv2.INTER_CUBIC)
CAR = {
    'left': cv2.rotate(CAR_PIC, cv2.ROTATE_90_CLOCKWISE),
    'up': cv2.rotate(CAR_PIC, cv2.ROTATE_180),
    'right': cv2.rotate(CAR_PIC, cv2.ROTATE_90_COUNTERCLOCKWISE),
    'down': CAR_PIC,
}
CAR_SPACE = CAR_SIZE + 2 * GAP_SIZE
GRAPHIC_SIZE = SIZE * CAR_SPACE
DISPLAY_SIZE = 15 * CAR_SPACE  # same window size as test.py (SIZE=15)
DISPLAY_DELAY_MS = 100  # ms per frame (30=fast, 100=slower)
RENDER_SIMULATION = True
SAVE_FIGURES = True
FIGURES_DIR = 'figures'
ACTION_SPACE = 2
# OpenCV uses BGR order. (0, 0, 255) displays as red, (0, 255, 0) as green.
LIGHT_GREEN = (0, 255, 0)
LIGHT_RED = (0, 0, 255)


def env_to_ghraphic(env):
    output = np.zeros((GRAPHIC_SIZE, GRAPHIC_SIZE, 3), dtype=np.uint8)
    output += np.ones((GRAPHIC_SIZE, GRAPHIC_SIZE, 3), dtype=np.uint8) * 255
    for i in range(SIZE):
        for j in range(SIZE):
            loc_i = i * CAR_SPACE + GAP_SIZE
            loc_j = j * CAR_SPACE + GAP_SIZE
            if j < SIZE / 2 - 2:
                if i < SIZE / 2 - 2:
                    output[loc_i - 1:loc_i + CAR_SIZE + 1, loc_j - 1:loc_j + CAR_SIZE + 1] = (0, 0, 0)
            if j > SIZE / 2 + 1:
                if i > SIZE / 2 + 1:
                    output[loc_i - 1:loc_i + CAR_SIZE + 1, loc_j - 1:loc_j + CAR_SIZE + 1] = (0, 0, 0)
            if j < SIZE / 2 - 2:
                if i > SIZE / 2 + 1:
                    output[loc_i - 1:loc_i + CAR_SIZE + 1, loc_j - 1:loc_j + CAR_SIZE + 1] = (0, 0, 0)
            if j > SIZE / 2 + 1:
                if i < SIZE / 2 - 2:
                    output[loc_i - 1:loc_i + CAR_SIZE + 1, loc_j - 1:loc_j + CAR_SIZE + 1] = (0, 0, 0)
            if j < SIZE / 2 - 3:
                if i < SIZE / 2 - 3:
                    output[loc_i - 1:loc_i + CAR_SIZE + 1, loc_j - 1:loc_j + CAR_SIZE + 1] = (120, 120, 120)
            if j > SIZE / 2 + 2:
                if i > SIZE / 2 + 2:
                    output[loc_i - 1:loc_i + CAR_SIZE + 1, loc_j - 1:loc_j + CAR_SIZE + 1] = (120, 120, 120)
            if j < SIZE / 2 - 3:
                if i > SIZE / 2 + 2:
                    output[loc_i - 1:loc_i + CAR_SIZE + 1, loc_j - 1:loc_j + CAR_SIZE + 1] = (120, 120, 120)
            if j > SIZE / 2 + 2:
                if i < SIZE / 2 - 3:
                    output[loc_i - 1:loc_i + CAR_SIZE + 1, loc_j - 1:loc_j + CAR_SIZE + 1] = (120, 120, 120)
            if (env[i][j] == np.array([255, 0, 255])).all():
                if j > SIZE / 2:
                    output[loc_i:loc_i + CAR_SIZE, loc_j:loc_j + CAR_SIZE] = CAR['up']
                else:
                    output[loc_i:loc_i + CAR_SIZE, loc_j:loc_j + CAR_SIZE] = CAR['down']
            elif (env[i][j] == np.array((255, 255, 0))).all():
                if i > SIZE / 2:
                    output[loc_i:loc_i + CAR_SIZE, loc_j:loc_j + CAR_SIZE] = CAR['right']
                else:
                    output[loc_i:loc_i + CAR_SIZE, loc_j:loc_j + CAR_SIZE] = CAR['left']
            if (env[i][j] == np.array(LIGHT_GREEN)).all():
                light_color = LIGHT_GREEN
            elif (env[i][j] == np.array(LIGHT_RED)).all():
                light_color = LIGHT_RED
            else:
                light_color = None
            if light_color is not None:
                center_coordinates = (loc_j + CAR_SIZE // 2, loc_i + CAR_SIZE // 2)
                cv2.circle(output, center_coordinates, CAR_SIZE // 5, light_color, CAR_SIZE)
    return output


def create_dqn_model():
    model = Sequential()
    model.add(Input(shape=(SIZE, SIZE, 3)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(ACTION_SPACE, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model


def load_dqn_model(path='deep1.h5'):
    model = create_dqn_model()
    model.load_weights(path)
    return model


def show_env_frame(env):
    img = env_to_ghraphic(env)
    for qq in range(SIZE):
        cv2.line(img, (int(qq * CAR_SPACE), 0), (int(qq * CAR_SPACE), img.shape[0]), (0, 0, 0), 1, 1)
        cv2.line(img, (0, int(qq * CAR_SPACE)), (img.shape[0], int(qq * CAR_SPACE)), (0, 0, 0), 1, 1)
        if qq == SIZE - 1:
            cv2.line(img, (int((qq + 1) * CAR_SPACE) - 1, 0), (int((qq + 1) * CAR_SPACE) - 1, img.shape[0]), (0, 0, 0), 1, 1)
            cv2.line(img, (0, int((qq + 1) * CAR_SPACE) - 1), (img.shape[0], int((qq + 1) * CAR_SPACE) - 1), (0, 0, 0), 1, 1)
    img = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imshow('image', np.array(img))
    cv2.waitKey(DISPLAY_DELAY_MS)


carsup = 1
carsdown = 2
lightup = 3
lightdown = 3
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
car_up = []
car_down = []
car_right = []
car_left = []
index = 0
vertical_is_red = 0
horizontal_is_red = 1
SHOW_EVERY = 100
UP_Queue = 0
DOWN_Queue = 0
RIGHT_Queue = 0
LEFT_Queue = 0

def actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, range_Q):
    full_flag_counter = 0
    new_car = 0
    if x1[range_Q] > 0.9 and UP_Queue <= SIZE / 2 - 2:
        car_up.append(Blobup())
        new_car += 1
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
        new_car += 1
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
        new_car += 1
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
        new_car += 1
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
    return (UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, full_flag_counter, new_car)


modelll = load_dqn_model('deep1.h5')
with open('x1.npy', 'rb') as f:
    x1 = np.load(f)
with open('x2.npy', 'rb') as f:
    x2 = np.load(f)
with open('x3.npy', 'rb') as f:
    x3 = np.load(f)
with open('x4.npy', 'rb') as f:
    x4 = np.load(f)
show = RENDER_SIMULATION
last_action = -5
UP_Queue = 0
DOWN_Queue = 0
RIGHT_Queue = 0
LEFT_Queue = 0
car_up = []
car_down = []
car_right = []
car_left = []
step = 0
current_state = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
episode_reward = 0
all_waiting_time = 0
all_cars = 0
last_situation = 0
episode_rewards = []
all_waiting_time_arr = []
for range_Q in range(len(x1)):
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    if UP_Queue >= SIZE - 1:
        UP_Queue = SIZE - 1
    if DOWN_Queue >= SIZE - 1:
        DOWN_Queue = SIZE - 1
    if RIGHT_Queue >= SIZE - 1:
        RIGHT_Queue = SIZE - 1
    if LEFT_Queue >= SIZE - 1:
        LEFT_Queue = SIZE - 1
    action = np.argmax(modelll.predict(np.array(current_state).reshape(-1, *current_state.shape) / 255, verbose=0)[0])
    if action == 0:
        horizontal_is_red = 1
        lightdown = 4
        if last_action != action:
            for index1 in range(3):
                env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, full_flag_counter, new_car = actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, range_Q)
                show_env_frame(env)
        vertical_is_red = 0
        lightup = 3
    elif action == 1:
        vertical_is_red = 1
        lightup = 4
        if last_action != action:
            for index1 in range(3):
                env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, full_flag_counter, new_car = actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, range_Q)
                show_env_frame(env)
        horizontal_is_red = 0
        lightdown = 3
    last_action = action
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, full_flag_counter, new_car = actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue, range_Q)
    current_state = env
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
    last_situation = current_situation
    episode_reward += reward
    all_waiting_time += full_flag_counter
    all_cars += new_car
    if (range_Q + 1) % 200 == 0:
        print(episode_reward)
        episode_rewards.append(episode_reward)
        all_waiting_time_arr.append(all_waiting_time / all_cars)
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        index = 0
        vertical_is_red = 0
        horizontal_is_red = 1
        last_action = -5
        UP_Queue = 0
        DOWN_Queue = 0
        RIGHT_Queue = 0
        LEFT_Queue = 0
        car_up = []
        car_down = []
        car_right = []
        car_left = []
        episode_reward = 0
        all_waiting_time = 0
        all_cars = 0
        last_situation = 0
        current_state = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    show_env_frame(env)
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.figure(figsize=(8, 4))
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Reward {SHOW_EVERY}ma')
plt.xlabel('episode #')
plt.title('Deep RL Test Reward (Moving Average)')
plt.grid(True)
if SAVE_FIGURES:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'deeprl_test_reward_ma.png'), dpi=200)
if RENDER_SIMULATION:
    plt.show()
plt.close()
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.ylabel('Reward')
plt.xlabel('episode #')
plt.title('Deep RL Test Reward')
plt.grid(True)
if SAVE_FIGURES:
    plt.savefig(os.path.join(FIGURES_DIR, 'deeprl_test_reward.png'), dpi=200)
if RENDER_SIMULATION:
    plt.show()
plt.close()
with open('DeepRL_Test_same_0.9_episode_rewards.npy', 'wb') as f:
    np.save(f, episode_rewards)
moving_avg = np.convolve(all_waiting_time_arr, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.figure(figsize=(8, 4))
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Waiting time {SHOW_EVERY}ma')
plt.xlabel('episode #')
plt.title('Deep RL Test Waiting Time (Moving Average)')
plt.grid(True)
if SAVE_FIGURES:
    plt.savefig(os.path.join(FIGURES_DIR, 'deeprl_test_waiting_time_ma.png'), dpi=200)
if RENDER_SIMULATION:
    plt.show()
plt.close()
plt.figure(figsize=(8, 4))
plt.plot(all_waiting_time_arr)
plt.ylabel('Mean wait per car')
plt.xlabel('episode #')
plt.title('Deep RL Test Waiting Time')
plt.grid(True)
if SAVE_FIGURES:
    plt.savefig(os.path.join(FIGURES_DIR, 'deeprl_test_waiting_time.png'), dpi=200)
if RENDER_SIMULATION:
    plt.show()
plt.close()
with open('DeepRL_Test_same_0.9_waiting_time.npy', 'wb') as f:
    np.save(f, all_waiting_time_arr)
print(f'Saved {len(episode_rewards)} episode rewards.')
print(f'Mean reward: {np.mean(episode_rewards):.2f}')
print(f'Mean waiting time: {np.mean(all_waiting_time_arr):.2f}')
if SAVE_FIGURES:
    print(f"Figures saved to '{FIGURES_DIR}/'")
