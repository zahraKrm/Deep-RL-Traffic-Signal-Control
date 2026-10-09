import time
import random
from collections import deque
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
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
# OpenCV uses BGR order. (0, 0, 255) displays as red, (0, 255, 0) as green.
LIGHT_GREEN = (0, 255, 0)
LIGHT_RED = (0, 0, 255)
d = {1: (255, 0, 255), 2: (255, 255, 0), 3: LIGHT_GREEN, 4: LIGHT_RED}
index = 0
vertical_is_red = 0
horizontal_is_red = 1
ACTION_SPACE = 2
HM_EPISODES = 10
count = 1000
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000
DISCOUNT = 0.95

def actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue):
    if np.random.random() > 0.9 and UP_Queue <= SIZE - 2:
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
        if full_flag == 0:
            if i.y != 0:
                i.y -= 1
        if i.y == 0:
            car_up.remove(i)
    if show:
        for mycar in car_up:
            env[mycar.y][mycar.x] = d[carsup]
        env[light_blob_up.y][light_blob_up.x] = d[lightup]
    if np.random.random() > 0.9 and DOWN_Queue <= SIZE - 2:
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
        if full_flag == 0:
            if i.y != SIZE - 1:
                i.y += 1
        if i.y == SIZE - 1:
            car_down.remove(i)
    if show:
        for mycar in car_down:
            env[mycar.y][mycar.x] = d[carsup]
        env[light_blob_down.y][light_blob_down.x] = d[lightup]
    if np.random.random() > 0.9 and RIGHT_Queue <= SIZE - 2:
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
        if full_flag == 0:
            if i.x != SIZE - 1:
                i.x += 1
        if i.x == SIZE - 1:
            car_right.remove(i)
    if show:
        for mycar in car_right:
            env[mycar.y][mycar.x] = d[carsdown]
        env[light_blob_right.y][light_blob_right.x] = d[lightdown]
    if np.random.random() > 0.9 and LEFT_Queue <= SIZE - 2:
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
        if full_flag == 0:
            if i.x != 0:
                i.x -= 1
        if i.x == 0:
            car_left.remove(i)
    if show:
        for mycar in car_left:
            env[mycar.y][mycar.x] = d[carsdown]
        env[light_blob_left.y][light_blob_left.x] = d[lightdown]
    return (UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue)
episode_rewards = []
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 2
MODEL_NAME = '2x256'
AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = True
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self._scalar_writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self._scalar_writer.as_default():
            for name, value in stats.items():
                tf.summary.scalar(name, float(value), step=self.step)
        self._scalar_writer.flush()

class DQNAgent:

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir='logs/{}-{}'.format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=(SIZE, SIZE, 3)))
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
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([t[0] for t in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([t[3] for t in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []
        for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + DISCOUNT * max_future_q
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
agent = DQNAgent()
step = 0
ep_rewards = [-200]
for episode in range(HM_EPISODES):
    print(episode)
    last_action = -5
    UP_Queue = 0
    DOWN_Queue = 0
    RIGHT_Queue = 0
    LEFT_Queue = 0
    car_up = []
    car_down = []
    car_right = []
    car_left = []
    agent.tensorboard.step = episode
    current_state = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    if episode % SHOW_EVERY == 0:
        print(f'on #{episode}, epsilon is {epsilon}')
        print(f'{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}')
        show = True
        show1 = True
    else:
        show1 = False
        show = True
    episode_reward = 0
    step = 1
    done = False
    for i in range(count):
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        if UP_Queue >= SIZE - 1:
            UP_Queue = SIZE - 1
        if DOWN_Queue >= SIZE - 1:
            DOWN_Queue = SIZE - 1
        if RIGHT_Queue >= SIZE - 1:
            RIGHT_Queue = SIZE - 1
        if LEFT_Queue >= SIZE - 1:
            LEFT_Queue = SIZE - 1
        obs = (UP_Queue + DOWN_Queue, RIGHT_Queue + LEFT_Queue)
        last_situation = UP_Queue + DOWN_Queue + RIGHT_Queue + LEFT_Queue
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, ACTION_SPACE)
        if action == 0:
            horizontal_is_red = 1
            if last_action != action:
                for index1 in range(3):
                    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                    UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue = actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue)
            vertical_is_red = 0
            lightup = 3
            lightdown = 4
        elif action == 1:
            vertical_is_red = 1
            if last_action != action:
                for index1 in range(3):
                    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                    UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue = actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue)
            horizontal_is_red = 0
            lightup = 4
            lightdown = 3
        last_action = action
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue = actionfunc(UP_Queue, DOWN_Queue, RIGHT_Queue, LEFT_Queue)
        new_state = env
        if UP_Queue >= SIZE - 1:
            UP_Queue = SIZE - 1
        if DOWN_Queue >= SIZE - 1:
            DOWN_Queue = SIZE - 1
        if RIGHT_Queue >= SIZE - 1:
            RIGHT_Queue = SIZE - 1
        if LEFT_Queue >= SIZE - 1:
            LEFT_Queue = SIZE - 1
        new_obs = (UP_Queue + DOWN_Queue, RIGHT_Queue + LEFT_Queue)
        current_situation = UP_Queue + DOWN_Queue + RIGHT_Queue + LEFT_Queue
        if current_situation < last_situation:
            if UP_Queue > SIZE / 4:
                reward = -1
            elif DOWN_Queue > SIZE / 4:
                reward = -1
            elif RIGHT_Queue > SIZE / 4:
                reward = -1
            elif LEFT_Queue > SIZE / 4:
                reward = -1
            else:
                reward = 0
        else:
            reward = -2
        agent.update_replay_memory((current_state, action, reward, new_state))
        if i == count - 1:
            done = True
            print(done)
        agent.train(done, step)
        current_state = new_state
        step += 1
        if show1:
            img = Image.fromarray(env, 'RGB')
            img = img.resize((600, 600))
            cv2.imshow('image', np.array(img))
            cv2.waitKey(1)
        episode_reward += reward
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
with open('DeepRL_env_episode_rewards.npy', 'wb') as f:
    np.save(f, episode_rewards)
agent.model.save('deep1.h5')
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Reward {SHOW_EVERY}ma')
plt.xlabel('episode #')
plt.show()
plt.plot(episode_rewards)
plt.show()
