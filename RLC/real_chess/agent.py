from keras.layers import Input, Dense, Flatten, Concatenate, Conv2D, Dropout
from keras.losses import mean_squared_error
from keras.models import Model, clone_model
from keras.optimizers import RMSprop
import numpy as np

class MyopicAgent(object):

    def __init__(self, color=-1):
        self.color = color

    def predict(self, layer_board, noise=True):
        layer_board1 = layer_board[0, :, :, :]
        pawns = 1 * np.sum(layer_board1[0, :, :])
        rooks = 5 * np.sum(layer_board1[1, :, :])
        minor = 3 * np.sum(layer_board1[2:4, :, :])
        queen = 9 * np.sum(layer_board1[4, :, :])
        maxscore = 40
        material = pawns + rooks + minor + queen
        board_value = self.color * material / maxscore
        if noise:
            added_noise = np.random.randn() / 1e3
        return board_value + added_noise

class Agent(object):
    def __init__(self, lr=0.003):
        self.optimizer = RMSprop(lr=lr)
        self.model = Model()
        # self.proportional_error = False
        self.our_net()

    def fix_model(self):
        #uodate the fixed model
        self.fixed_model = clone_model(self.model)
        self.fixed_model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae']) #compile the fixed model
        self.fixed_model.set_weights(self.model.get_weights())


    def our_net(self):
        layer_state = Input(shape=(8, 8, 8), name='state')
        conv_layers = [Conv2D(filters, kernel_size, activation='relu')(layer_state)
                    for filters, kernel_size in zip([2, 4, 6, 8, 5, 3, 3], 
                                                    [(1, 1), (2, 2), (3, 3), (4, 4), (8, 8), (1, 8), (8, 1)])]
        flattened_layers = [Flatten()(conv) for conv in conv_layers]
        concatenated = Concatenate(name='concatenate')(flattened_layers)#concatenate the flattened layers
        dense = Dense(256, activation='relu')(concatenated)#dense layer
        dense = Dropout(0.5)(dense)  # prevent overfitting
        dense = Dense(128, activation='relu')(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dense(32, activation='relu')(dense)
        value_head = Dense(1)(dense)
        self.model = Model(inputs=layer_state, outputs=value_head)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    # def predict_distribution(self, states, batch_size=256):
    #     """
    #     :param states: list of distinct states
    #     :param n:  each state is predicted n times
    #     :return:
    #     """
    #     predictions_per_state = int(batch_size / len(states))
    #     state_batch = []
    #     for state in states:
    #         state_batch = state_batch + [state for x in range(predictions_per_state)]

    #     state_batch = np.stack(state_batch, axis=0)
    #     predictions = self.model.predict(state_batch)
    #     predictions = predictions.reshape(len(states), predictions_per_state)
    #     mean_pred = np.mean(predictions, axis=1)
    #     std_pred = np.std(predictions, axis=1)
    #     upper_bound = mean_pred + 2 * std_pred

    #     return mean_pred, std_pred, upper_bound

    def predict(self, board_layer):
        return self.model.predict(board_layer)

    def TD_update(self, states, rewards, sucstates, episode_active, gamma=0.9):
        # Compute the TD target.
        suc_state_values = self.fixed_model.predict(sucstates)
        V_target = np.array(rewards) + np.array(episode_active) * gamma * np.squeeze(suc_state_values)
        # Perform a step of minibatch Gradient Descent.
        self.model.fit(x=states, y=V_target, epochs=1, verbose=0)
        V_state = self.model.predict(states)  # V_state is a list of lists
        td_errors = V_target - np.squeeze(V_state)
        return td_errors

    def MC_update(self, states, returns):
        # Perform a step of minibatch Gradient Descent.
        self.model.fit(x=states, y=returns, epochs=0, verbose=0)
        V_state = np.squeeze(self.model.predict(states))
        td_errors = returns - V_state
        return td_errors
