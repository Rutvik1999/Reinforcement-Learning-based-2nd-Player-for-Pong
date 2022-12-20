import pygame
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import cv2
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() #to use custom loss function
np.random.seed(seed=0)
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Concatenate, Input
from tensorflow.keras.losses import categorical_crossentropy
from keras.utils.np_utils import to_categorical  

training_mode = False
inp = input("Press T for training\n")
if inp == "T" or inp == "t": 
    training_mode = True

pygame.init()
WINDOW_W, WINDOW_H = 600, 600
CLK = 10
ENABLE_CLK = False
clk = False
last_3_hit_l = 0
last_3_hit_r = 0
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PADDLE_H, PADDLE_W = 120, 10
BALL_R = 8
MAX_BOUNCE_ANGLE = (PADDLE_H/2)*(np.pi/12)
WIN_SCORE = 10
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Pong")
folder_path = '/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final'
frame_skip = 0
w_crop_ratio = 0.8

class Paddle:
    def __init__(self, x, y, w, h, PADDLE_VEL = 120):
        self.PADDLE_VEL = PADDLE_VEL
        self.x, self.y, self.w, self.h = x, y, w, h
        self.initial_x, self.initial_y = x, y

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.w, self.h))

    def move(self, up=True):
        if up: self.y -= self.PADDLE_VEL
        else:  self.y += self.PADDLE_VEL

    def reset(self):
        self.x, self.y = self.initial_x, self.initial_y

class Ball:
    def __init__(self, x, y, r, BALL_VEL = 20):
        self.BALL_VEL = BALL_VEL
        self.x, self.y, self.r, self.x_vel, self.y_vel = x, y, r, self.BALL_VEL, np.random.randint(-1 * self.BALL_VEL, self.BALL_VEL)
        self.initial_x, self.initial_y = self.x, self.y

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (self.x, self.y), self.r)

    def reset(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.y_vel = np.random.randint(-1 * self.BALL_VEL, self.BALL_VEL)
        self.x_vel *= -1

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset_ref(self):
        self.x_vel *= -1

def draw(screen, objs):
    screen.fill(BLACK)
    for o in objs:
        o.draw(screen)
    pygame.display.update()

def movement_handler(key, l_paddle, r_paddle):
    if key[pygame.K_w] and l_paddle.y >= 0:                         l_paddle.move(up=True)
    if key[pygame.K_s] and l_paddle.y + l_paddle.h <= WINDOW_H:     l_paddle.move(up=False)
    if key[pygame.K_UP] and r_paddle.y >=0:                         r_paddle.move(up=True)
    if key[pygame.K_DOWN] and r_paddle.y + r_paddle.h <= WINDOW_H:  r_paddle.move(up=False)

    if l_paddle.y < 0:                      l_paddle.y = 0
    if l_paddle.y + l_paddle.h > WINDOW_H:  l_paddle.y = WINDOW_H - l_paddle.h
    if r_paddle.y < 0:                      r_paddle.y = 0
    if r_paddle.y + r_paddle.h > WINDOW_H:  r_paddle.y = WINDOW_H - r_paddle.h

def check_paddle_collision(ball, l_paddle, r_paddle):
    is_paddle_colission = False
    is_l_paddle = False
    relative_intersect_y = 0
        
    if (ball.x <= (l_paddle.x + l_paddle.w)) and ((ball.y >= l_paddle.y) and (ball.y <= (l_paddle.y + l_paddle.h))) and ball.x_vel < 0:
        relative_intersect_y = (l_paddle.y  + (l_paddle.h/2) - ball.y )
        is_l_paddle = True
        is_paddle_colission = True

    elif (ball.x >= r_paddle.x) and ((ball.y >= r_paddle.y) and (ball.y <= (r_paddle.y + r_paddle.h))):
        relative_intersect_y = (r_paddle.y + (r_paddle.h/2) - ball.y )
        is_l_paddle = False
        is_paddle_colission = True
        
    return is_paddle_colission, is_l_paddle, relative_intersect_y

def collision_handler(ball, l_paddle, r_paddle):
    is_paddle_colission, is_l_paddle, relative_intersect_y = check_paddle_collision(ball, l_paddle, r_paddle)
    if ball.y >= WINDOW_H :
        ball.y_vel *= -1
    elif ball.y <= 0:
        ball.y_vel *= -1
    else:
        if is_paddle_colission:
            if is_l_paddle:
                reduction_factor = (l_paddle.h / 2) / ball.BALL_VEL
                ball.y_vel = relative_intersect_y / reduction_factor #angle of reflection
            if not is_l_paddle:
                reduction_factor = (r_paddle.h / 2) / ball.BALL_VEL
                ball.y_vel = relative_intersect_y / reduction_factor
            ball.x_vel = ball.x_vel * -1
    ball.y_vel = np.sign(ball.y_vel) * np.max([np.abs(ball.y_vel), 1]) #set min |y_vel| = 1
    if ball.y_vel == 0: ball.reset() #avoid mid-point reflection loop
    return is_paddle_colission, is_l_paddle

def new_game_action(l_paddle_params={"x": 0, "y": WINDOW_H//2 - PADDLE_H//2, "w": PADDLE_W, "h": PADDLE_H}, 
                    r_paddle_params={"x": WINDOW_W - 0 - PADDLE_W, "y": WINDOW_H//2 - PADDLE_H//2, "w": PADDLE_W, "h": PADDLE_H}, 
                    ball_params={"x": WINDOW_W//2, "y": WINDOW_H//2, "r":BALL_R}):
    l_paddle = Paddle(x=l_paddle_params["x"], y=l_paddle_params["y"], w=l_paddle_params["w"], h=l_paddle_params["h"])
    r_paddle = Paddle(x=r_paddle_params["x"], y=r_paddle_params["y"], w=r_paddle_params["w"], h=r_paddle_params["h"])
    ball = Ball(x=ball_params["x"], y=ball_params["y"], r=ball_params["r"])

    return ball, l_paddle, r_paddle

def paddle_ai(x_paddle, ball):
    if ball.x < WINDOW_W * w_crop_ratio: #limit hard-coede AI view
        if x_paddle.y + x_paddle.h/2 < ball.y:
            x_paddle.move(up=False)

        if x_paddle.y + x_paddle.h/2 > ball.y:
            x_paddle.move(up=True)
    #do not move out of window
    if x_paddle.y < 0: x_paddle.y = 0
    if x_paddle.y + x_paddle.h > WINDOW_H: x_paddle.y = WINDOW_H - x_paddle.h

def paddle_control_agent(x_paddle, action):
    if action == 0: #up
        x_paddle.move(up=True)
    elif action == 2: #down
        x_paddle.move(up=False)
    #do not move out of window
    if x_paddle.y <= 0: x_paddle.y = 0
    if x_paddle.y + x_paddle.h >= WINDOW_H: x_paddle.y = WINDOW_H - x_paddle.h

class PongGame:
    global clk, ENABLE_CLK
    total_score1 = 0
    total_score2 = 0
    if ENABLE_CLK == True: clk = pygame.time.Clock()
    def __init__(self):
        self.ball, self.l_paddle, self.r_paddle = new_game_action()
        self.objs = [self.ball, self.l_paddle, self.r_paddle]

    def reset(self):
        for o in self.objs:
            o.reset()
        self.total_score_1, self.total_score_2 = 0, 0

    def get_frame_init(self):
        draw(screen, self.objs)
        image_data1 = pygame.surfarray.array3d(pygame.display.get_surface())
        paddle_ai(self.l_paddle, self.ball)
        self.ball.move()
        _, _ = collision_handler(self.ball, self.l_paddle, self.r_paddle)
        # paddle_ai(self.l_paddle, self.ball)
        return image_data1

    def get_frame(self, action, game_over=False):
        draw(screen, self.objs)
        self.ball.move()
        is_paddle_colission, _ = collision_handler(self.ball, self.l_paddle, self.r_paddle)
        paddle_control_agent(self.r_paddle, action)
        paddle_ai(self.l_paddle, self.ball)
        score = 0
        if self.ball.x < 0 and not is_paddle_colission:
            score = 1
            self.total_score_1 += 1
            self.ball.reset_ref()

        elif self.ball.x >= self.r_paddle.x and not is_paddle_colission:
            self.total_score_2 += 1
            score = -1
            self.ball.reset_ref()

        if score != 0: 
            print("Score = ", score)
            print("Total Score 1 (RL) =", self.total_score_1)
            print("Total Score 2 (TR) =", self.total_score_2)

        image_data2 = pygame.surfarray.array3d(pygame.display.get_surface())

        win = None
        self.total_score1 = self.total_score_1
        self.total_score2 = self.total_score_2

        if self.total_score_2 >= 20:
            win = 0
            game_over = True
            self.ball.reset_ref()
            self.total_score_1 = 0
            self.total_score_2 = 0
            self.r_paddle.reset()
            self.l_paddle.reset()

        elif self.total_score_1 >= 20:
            win = 1
            game_over = True
            self.ball.reset_ref()
            self.total_score_1 = 0
            self.total_score_2 = 0
            self.r_paddle.reset()
            self.l_paddle.reset()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()

        return self.total_score1, self.total_score2, score, image_data2, game_over, win

def create_model(input_img, prev_action):
    model = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_img)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Flatten()(model)
    model = Concatenate()([prev_action, model])
    model = BatchNormalization()(model)
    model = Dense(units=8192, activation='relu')(model)
    model = Dense(units=2048, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(units=128, activation='relu')(model)
    output = Dense(units=3, activation='softmax')(model)
    return output


class Custom_CE_Loss(tf.keras.losses.Loss):
    def __init__(self, reward_list):
        super().__init__()
        self.reward_list = reward_list

    def call(self, y_true, y_pred):

        #fn3
        x1 = -1 * tf.cast(tf.math.greater_equal(self.reward_list, 1), tf.float32) * categorical_crossentropy(y_pred=y_pred, y_true=y_true) * self.reward_list
        x2 = -1 * tf.cast(tf.math.less_equal(self.reward_list, -1), tf.float32) * categorical_crossentropy(y_pred=y_pred, y_true=y_true) * self.reward_list
        x3 = tf.cast(tf.math.greater(self.reward_list, -1), tf.float32) * tf.cast(tf.math.less(self.reward_list, 1), tf.float32) * categorical_crossentropy(y_pred=y_pred, y_true=y_true)
        return x1 + x2 + x3


        #fn2
        # x1 = -1 * tf.cast(tf.math.greater_equal(self.reward_list, 1), tf.float32) * categorical_crossentropy(y_pred=y_pred, y_true=y_true) * self.reward_list
        # x2 = -1 * tf.cast(tf.math.less_equal(self.reward_list, -1), tf.float32) * categorical_crossentropy(y_pred=(1 - y_pred), y_true=y_true) * self.reward_list
        # x3_1 = tf.cast(tf.math.greater(self.reward_list, -1), tf.float32) * categorical_crossentropy(y_pred=y_pred, y_true=y_true)
        # x3_2 = tf.cast(tf.math.less(self.reward_list, 1), tf.float32) * categorical_crossentropy(y_pred=y_pred, y_true=y_true)
        # return (x1 + x2 + (x3_1 * x3_2))

        #fn1
        # x = np.array(self.reward_list == 0, dtype=np.float32) * categorical_crossentropy(y_pred=y_pred, y_true=y_true)
        # return (-1 * categorical_crossentropy(y_pred=y_pred, y_true=y_true) * self.reward_list)

        # return mean_squared_error(y_true, y_pred) * self.reward_list
        # mse = tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=1)
        # return tf.math.reduce_sum(mse * (1 + tf.math.square((np.array(tf.argmax(y_pred, axis=1) == tf.argmax(y_true, axis=1), dtype=np.int64) - self.reward_list))))
        # if y_pred (action_predicted) == y_true (action_taken) and reward = -1 or 0 ===> action taken was incorrect ===> Increase loss by subtracting reward list
        # if argmax(y_pred, axis=1) = [0 2 1] and argmax(y_true, axis=1) = [1 1 1] and rewards = [-1 0 1], 
        # ([0 2 1] == [1 1 1]) = [0 0 1], [0 0 1] - [-1 0 1] = [1 0 0] ===> [1 0 0]^2 ===> [1 0 0] + 1 ===> [2 1 1]^2 ===> mse * [4 1 1]
        # Increase loss where reward = -1

if training_mode == True:

    input_img = Input(shape=(100,100,1), name = 'screen_diff') #state ===> difference between current and previous frame
    episode_reward = Input(shape=(1,),name = 'episode_reward') #reward from action
    prev_action = Input(shape=(3,),name = 'prev_action')    #previous action taken
    target_net = create_model(input_img, prev_action)
    target_net = keras.models.Model(inputs=[input_img, prev_action], outputs=target_net)
    target_net.summary()
    policy_network_train = keras.models.Model(inputs=[input_img, prev_action, episode_reward],outputs=target_net.output)
    policy_network_train.compile(optimizer='Adam',loss=Custom_CE_Loss(episode_reward))

first_run = False

def process_frame(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (100, 100))
    screen = np.expand_dims(screen, axis=0)
    screen = screen / 255
    return screen

def normalize_arr(arr):
    arr = np.nan_to_num(arr)
    arr -= np.mean(arr) # subtract by average
    arr /= np.std(arr)
    return arr

def process_rewards(reward_list, reward_decay = 0.9):
    temp, new_reward_list = 0, np.zeros_like(reward_list, dtype=np.float16)
    for i in range(len(reward_list) - 1, -1, -1):
        if reward_list[i] == 0: temp *= reward_decay
        else: temp = reward_list[i]
        new_reward_list[i] = temp
        new_reward_list = np.clip(new_reward_list, -10.0, 10.0)
    return normalize_arr(new_reward_list)

EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 500
steps_done = 0

def explore():
    return to_categorical(np.random.randint(3), num_classes=3)

def exploit(state, model_target, prev_action):
    pred = model_target.predict([state.reshape(1,100,100,1), prev_action], batch_size=1, verbose=2)
    if np.isnan(pred.reshape(-1)).any():
        pred = np.clip(pred[0], 0.00001, 0.99999)
        pred = pred/np.sum(pred)
    return to_categorical(np.random.choice(3, 1, p = pred.reshape(-1))[0], num_classes=3)

def select_action(state, model_target, prev_action):
    global steps_done
    sample = np.random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # pred = np.clip(pred[0], 0.01, 0.99)
        # return to_categorical(np.random.choice(3, 1, p = pred/np.sum(pred))[0], num_classes=3)
        return exploit(state, model_target, prev_action)
    else:
        return explore()

from PIL import Image
pong = PongGame()
def generate_episode(model_target):
    game_state_info = {"state_list":[], "action_list":[], "reward_list":[]}
    episode_durations=[]
    pong.reset()
    last_screen = pong.get_frame_init()
    init_action = np.random.randint(3)
    _, _, reward, current_screen, done, _ = pong.get_frame(init_action)
    state = process_frame(current_screen - last_screen).reshape(100,100,1)
    game_state_info["state_list"].append(state)
    game_state_info["action_list"].append(to_categorical(init_action, num_classes=3))
    game_state_info["reward_list"].append(reward)

    for t in count():
        # plt.imshow(current_screen.transpose(1,0,2))
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.savefig("/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final/images/frames/" + str(t) + ".jpeg")
        state = process_frame(current_screen - last_screen)
        # plt.imshow(state.reshape(100,100).T)
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.savefig("/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final/images/states/" + str(t) + ".jpeg")
        # cv2.imwrite("/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final/images/states/__" + str(t) + ".jpeg", (state.reshape(100,100).T)*255)
        prev_action = np.array(game_state_info["action_list"][-1:])
        action = select_action(state, model_target, prev_action)
        last_screen = current_screen
        total_score1, total_score2, reward, current_screen, done, _ = pong.get_frame(np.argmax(action))
        print("prev_action =", prev_action, "   reward =", reward)
        game_state_info["state_list"].append(state.reshape(100,100,1))
        game_state_info["action_list"].append(action)
        game_state_info["reward_list"].append(reward)
        if done or (t > 5000): #avoid infinite loops 
            episode_durations.append(t)
            # np.save("/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final/images/rewards_ex.npy", game_state_info["reward_list"])
            break

    return game_state_info, episode_durations, total_score1, total_score2



# def plot_durations_loss(episode_durations, loss_log):
#     plt.figure(2)
#     plt.clf()
#     plt.subplot(1,2,1)
#     plt.title('Training Episode Durations')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(episode_durations)
#     plt.subplot(1,2,2)
#     plt.title('Loss')
#     plt.xlabel('Episode')
#     plt.ylabel('Loss')
#     # loss_log[0][0] /= 10 #0th value very large, further plot not visible clearly
#     plt.plot(loss_log)
#     plt.pause(0.001)  # pause a bit so that plots are updated

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    plt.title('Training Episode Durations')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_loss(loss_log):
    plt.figure(3)
    plt.clf()    
    plt.title('Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    # loss_log[0][0] /= 10 #0th value very large, further plot not visible clearly
    plt.plot(loss_log)
    plt.pause(0.001)

# def plot_reward_list(reward_list):
#     plt.figure(3)
#     plt.clf()
#     plt.title('Rewards in latest episode')
#     plt.xlabel('Frame')
#     plt.ylabel('Reward')
#     plt.plot(reward_list)
#     plt.pause(0.001)  # pause a bit so that plots are updated 

def plot_agent_score(score_list):
    plt.figure(4)
    plt.clf()
    plt.title('Learning Agent Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(score_list)
    plt.pause(0.001)  # pause a bit so that plots are updated 

# def plot_tr_score(score_list):
#     plt.figure(5)
#     plt.clf()
#     plt.title('Hard-coded AI Score')
#     plt.xlabel('Episode')
#     plt.ylabel('Score')
#     plt.plot(score_list)
#     plt.pause(0.001)  # pause a bit so that plots are updated 


episode_durations = []
total_score_1_list = []
total_score_2_list = []
loss_log = []
def train_network(n_episodes, model_train, model_target):
    global episode_durations
    global total_score_1_list
    global total_score_2_list
    global loss_log
    for e in range(n_episodes):
        game_state_info, episode_duration, total_score_1, total_score_2 = generate_episode(model_target)
        episode_durations.append(episode_duration)
        total_score_1_list.append(total_score_1)
        total_score_2_list.append(total_score_2)
        reward_list_processed = process_rewards(game_state_info["reward_list"])
        state_list = np.array(game_state_info["state_list"])
        action_list = np.array(game_state_info["action_list"])
        # action_list = to_categorical(action_list.reshape(-1,1), num_classes=3)
        history = model_train.fit(x = [state_list, np.roll(action_list, axis=0, shift=1), reward_list_processed], y = action_list, batch_size=64, verbose=2)
        loss_log.append(history.history["loss"])
        # plot_reward_list(reward_list_processed)
        plot_durations(episode_durations)
        plot_loss(loss_log)
        plot_agent_score(total_score_1_list)
        # plot_tr_score(total_score_2_list)

    
    return model_train, model_target, reward_list_processed

import time
if training_mode == True:
    total_episodes = 0
    n_episodes = int(input("Train For n_episodes: "))
    while n_episodes > 0:
        policy_network_train, target_net, reward_list_processed = train_network(n_episodes, policy_network_train, target_net)
        total_episodes += n_episodes
        target_net.save(folder_path + "/target_net_keras_cnn_" + str(total_episodes) + ".h5")
        policy_network_train.save(folder_path + "/policy_network_train_" + str(total_episodes) + ".h5")
        n_episodes = int(input("Train for n_episodes more (enter 0 to stop training): "))
    # np.save("/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final/images/episode_durations_fn1_2.npy", np.array(episode_durations))
    # np.save("/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final/images/total_score_1_list_fn1_2.npy", np.array(total_score_1_list))
    # np.save("/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final/images/total_score_2_list_fn1_2.npy", np.array(total_score_2_list))
    # np.save("/Users/rutvik1999/Library/CloudStorage/OneDrive-Personal/MS Study Final/images/loss_log_fn1_2.npy", np.array(loss_log))
        

first_run = False
screen_current, screen_prev, state = 0, 0, 0
target_net = tf.keras.models.load_model(folder_path + "/target_net_keras_cnn_" + input("enter file name *_cnn_???.h5") + ".h5", compile=False)

def predict_action(target_net, state, prev_action):
    pred = target_net.predict([state.reshape(1,100,100,1), prev_action], batch_size=1, verbose=2)
    # pred = np.clip(pred[0], 0.01, 0.99)
    # pred = pred/np.sum(pred)
    # return to_categorical(np.random.choice(3, 1, p=pred[0]), num_classes=3).reshape(1,3)
    return to_categorical(np.random.choice(3, 1, p=pred[0])[0], num_classes=3)

prev_action = np.array([0,1,0]).reshape(1,3)

def paddle_ai_RL(r_paddle, screen):
    global screen_current, screen_prev, state
    global target_net
    global first_run
    global prev_action
    if first_run == False:
        screen_current = screen
        screen_prev = screen
        state = screen
        state = process_frame(state)
        first_run = False
    else:
        screen_prev = screen_current
        screen_current = screen
        state = screen_current - screen_prev
        state = process_frame(state)
    
    action = predict_action(target_net, state, prev_action)
    prev_action = action.reshape(1,3)
    # print("action =", action)
    paddle_control_agent(r_paddle, np.argmax(action))

    
def main_RL():
    global clk
    global prev_reward
    global ENABLE_CLK
    run_loop = True
    if ENABLE_CLK == True: clk = pygame.time.Clock()
    #updates the window
    ball, l_paddle, r_paddle = new_game_action()

    score_total_1 = 0
    score_total_2 = 0
    while run_loop:
        if clk != False: clk.tick(CLK)
        draw(screen, [l_paddle, r_paddle, ball])
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run_loop = False
                break
        
        key_press = pygame.key.get_pressed()
        movement_handler(key_press, l_paddle, r_paddle)
        ball.move()
        paddle_ai_RL(r_paddle, pygame.surfarray.array3d(pygame.display.get_surface()))
        is_paddle_colission, is_l_paddle = collision_handler(ball, l_paddle, r_paddle)
        score = 0
        if ball.x < 0 and not is_paddle_colission:
            score = 1
            score_total_1 += 1
            print("Score total 1 = ", score_total_1)
            ball.reset_ref()
            # l_paddle.reset()
        elif ball.x > WINDOW_W and not is_paddle_colission:
            score = -1
            score_total_2 += 1
            print("Score total 2 = ", score_total_2)
            ball.reset_ref()
            # r_paddle.reset()

        prev_reward = score

        if score_total_2 > WIN_SCORE:
            print("Player 2 Wins")
            ball, l_paddle, r_paddle = new_game_action()
            score_total_1 = 0
            score_total_2 = 0
        
        elif score_total_1 > WIN_SCORE:
            print("Player 1 Wins")
            ball, l_paddle, r_paddle = new_game_action()
            score_total_1 = 0
            score_total_2 = 0
        # paddle_ai(l_paddle, ball)
    pygame.quit()

def main_trad():
    clk = False
    run_loop = True
    if ENABLE_CLK == True: clk = pygame.time.Clock()
    ball, l_paddle, r_paddle = new_game_action()
    score_total_1 = 0
    score_total_2 = 0
    while run_loop:
        if clk != False: clk.tick(CLK)
        draw(screen, [l_paddle, r_paddle, ball])
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run_loop = False
                break
        
        key_press = pygame.key.get_pressed()
        movement_handler(key_press, l_paddle, r_paddle)
        ball.move()
        is_paddle_colission, is_l_paddle = collision_handler(ball, l_paddle, r_paddle)
        
        if ball.x < 0 and not is_paddle_colission:
            score_total_1 += 1
            ball.reset_ref()
            # l_paddle.reset()
        elif ball.x > WINDOW_W and not is_paddle_colission:
            score_total_2 += 1
            ball.reset_ref()
            # r_paddle.reset()

        print("Score (Right Player) =", score_total_1)
        print("Score (Left Player) =", score_total_2)

        if score_total_1 > WIN_SCORE:
            print("Player (Right) Wins")
            ball, l_paddle, r_paddle = new_game_action()
            score_total_1 = 0
            score_total_2 = 0
            
        elif score_total_2 > WIN_SCORE:
            print("Player (Left) Wins")
            ball, l_paddle, r_paddle = new_game_action()
            score_total_1 = 0
            score_total_2 = 0

        paddle_ai(r_paddle, ball)
    pygame.quit()

def main_RL_trad():
    global clk
    global prev_reward
    global ENABLE_CLK
    run_loop = True
    if ENABLE_CLK == True: clk = pygame.time.Clock()
    ball, l_paddle, r_paddle = new_game_action()
    score_total_1, score_total_2 = 0, 0
    while run_loop:
        if clk != False: clk.tick(CLK)
        draw(screen, [l_paddle, r_paddle, ball])
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run_loop = False
                break
        
        key_press = pygame.key.get_pressed()
        ball.move()
        paddle_ai_RL(r_paddle, pygame.surfarray.array3d(pygame.display.get_surface()))
        is_paddle_colission, is_l_paddle = collision_handler(ball, l_paddle, r_paddle)
        score = 0
        if ball.x < 0 and not is_paddle_colission:
            score = 1
            score_total_1 += 1
            print("Score =", score)
            print("Total Score (RL) =", score_total_1)
            print("Total Score (TR) =", score_total_2)
            ball.reset_ref()
            # l_paddle.reset()
        elif ball.x > WINDOW_W and not is_paddle_colission :
            score = -1
            score_total_2 += 1
            print("Score = ", score)
            print("Total Score (RL) =", score_total_1)
            print("Total Score (TR) =", score_total_2)
            ball.reset_ref()
            # r_paddle.reset()

        prev_reward = score

        if score_total_2 > WIN_SCORE:
            print("TR Player Wins")
            ball, l_paddle, r_paddle = new_game_action()
            score_total_2 = 0
            l_paddle.reset()
            r_paddle.reset()
            ball.reset()
        elif score_total_1 > WIN_SCORE:
            print("RL Player Wins")
            ball, l_paddle, r_paddle = new_game_action()
            score_total_1 = 0
            l_paddle.reset()
            r_paddle.reset()
            ball.reset()
        paddle_ai(l_paddle, ball)
    pygame.quit()


if __name__ == "__main__":
    sel = input("1. Player vs. RL AI \n2. Player vs. hard-coded AI\n3. hard-coded AI vs. RL AI\n")
    if sel == "1": main_RL()
    elif sel == "2": main_trad()
    elif sel == "3": main_RL_trad()

        
