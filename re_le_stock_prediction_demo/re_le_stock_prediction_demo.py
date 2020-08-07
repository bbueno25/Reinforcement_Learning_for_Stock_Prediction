"""
DOCSTRING
"""
import collections
import keras
import math
import random
import sys

class Agent:
    """
    DOCSTRING
    """
    def __init__(self, state_size, is_eval=False, model_name=""):
        """
        Initialization
        """
        self.state_size = state_size
        self.action_size = 3
        self.memory = collections.deque(maxlen=1000)
        self.inventory = list()
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = keras.models.load_model(
            "models/" + model_name) if is_eval else self._model()

    def _model(self):
        """
        DOCSTRING
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(keras.layers.Dense(units=32, activation="relu"))
        model.add(keras.layers.Dense(units=8, activation="relu"))
        model.add(keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))
        return model

    def act(self, state):
        """
        DOCSTRING
        """
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return numpy.argmax(options[0])

    def exp_replay(self, batch_size):
        """
        DOCSTRING
        """
        mini_batch = list()
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * numpy.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

class Evaluate:
    """
    DOCSTRING
    """
    def __call__(self):
        """
        Call
        """
        if len(sys.argv) != 3:
            print("Usage: python evaluate.py [stock] [model]")
            exit()
        stock_name, model_name = sys.argv[1], sys.argv[2]
        model = keras.models.load_model("models/" + model_name)
        window_size = model.layers[0].input.shape.as_list()[1]
        agent = Agent(window_size, True, model_name)
        data = Functions.get_stock_data_vec(stock_name)
        l = len(data) - 1
        batch_size = 32
        state = Functions.get_state(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = list()
        for t in range(l):
            action = agent.act(state)
            next_state = Functions.get_state(data, t + 1, window_size + 1)
            reward = 0
            if action == 1:
                agent.inventory.append(data[t])
                print("Buy: " + Functions.format_price(data[t]))
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                print("Sell: {} | Profit: {}".format(
                    Functions.format_price(data[t]),
                    Functions.format_price(data[t] - bought_price)))
            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                print(stock_name + " Total Profit: " + Functions.format_price(total_profit))

class Functions:
    """
    DOCSTRING
    """
    def format_price(self, n):
        """
        prints formatted price
        """
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))
    
    def get_state(self, data, t, n):
        """
        returns an an n-day state representation ending at time t
        """
        d = t - n + 1
        block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
        res = list()
        for i in range(n - 1):
            res.append(self.sigmoid(block[i + 1] - block[i]))
        return numpy.array([res])

    def get_stock_data_vec(self, key):
        """
        returns the vector containing stock data from a fixed file
        """
        vec = list()
        lines = open("data/" + key + ".csv", "r").read().splitlines()
        for line in lines[1:]:
            vec.append(float(line.split(",")[4]))
        return vec

    def sigmoid(self, x):
        """
        returns the sigmoid
        """
        return 1 / (1 + math.exp(-x))

class Train:
    """
    DOCSTRING
    """
    def __call__(self):
        """
        Call
        """
        if len(sys.argv) != 4:
            print("Usage: python train.py [stock] [window] [episodes]")
            exit()
        stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        agent = Agent(window_size)
        data = Functions.get_stock_data_vec(stock_name)
        l = len(data) - 1
        batch_size = 32
        for e in range(episode_count + 1):
            print("Episode " + str(e) + "/" + str(episode_count))
            state = Functions.get_state(data, 0, window_size + 1)
            total_profit = 0
            agent.inventory = list()
            for t in range(l):
                action = agent.act(state)
                next_state = Functions.get_state(data, t + 1, window_size + 1)
                reward = 0
                if action == 1:
                    agent.inventory.append(data[t])
                    print("Buy:",Functions.format_price(data[t]))
                elif action == 2 and len(agent.inventory) > 0:
                    bought_price = agent.inventory.pop(0)
                    reward = max(data[t] - bought_price, 0)
                    total_profit += data[t] - bought_price
                    print("Sell: {} | Profit: {}".format(
                        Functions.format_price(data[t]),
                        Functions.format_price(data[t] - bought_price)))
                done = True if t == l - 1 else False
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    print("Total Profit:", Functions.format_price(total_profit))
                if len(agent.memory) > batch_size:
                    agent.exp_replay(batch_size)
            if e % 10 == 0:
                agent.model.save("models/model_ep" + str(e))
