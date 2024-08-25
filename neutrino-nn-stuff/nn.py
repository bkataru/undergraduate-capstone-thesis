import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import Callback
from keras.optimizers import Adam, RMSprop

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from pytrino import oscprobs

def RNNmodel(probabilities, parameters):
    # Generate some dummy data for demonstration
    # You should replace this with your actual data
    num_samples = len(probabilities)
    num_probabilities = 9
    num_parameters = 8

    x_train, x_test, y_train, y_test = train_test_split(probabilities, parameters, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = np.reshape(x_train, (x_train.shape[0], 1, 1, num_probabilities))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, 1, num_probabilities))

    model = keras.Sequential()
    model.add(layers.LSTM(32, activation='relu', input_shape=(1, num_probabilities)))
    model.add(layers.Dense(num_parameters))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    class TrainingMonitor(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f'Epoch {epoch + 1}/{self.params["epochs"]} - loss: {logs["loss"]:.4f}')

    training_monitor = TrainingMonitor()

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_test, y_test), callbacks=[training_monitor])

    # Save the trained model weights
    model.save_weights('trained_model_weights.h5')

    # Evaluate the model
    loss = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}')

    # Generate random probabilities for prediction
    probabilities_to_predict = np.array([[0.9025862572768633, 0.09640388804920827, 0.0010098546739280726, 0.09619372529756111, 0.902826795045462, 0.0009794796569765783, 0.0012200174255752511, 0.0007693169053294073, 0.9980106656690956]])
    probabilities_to_predict = scaler.transform(probabilities_to_predict)
    probabilities_to_predict = np.reshape(probabilities_to_predict, (1, 1, 1, num_probabilities))

    predicted_params = model.predict(probabilities_to_predict)

    print('Predicted Parameters:', predicted_params)

    # # Compare the predicted parameters with the actual parameters
    # for i in range(pred_samples):
    #     print(f'Example {i + 1}:')
    #     print('Probabilities:', probabilities_to_predict[i])
    #     print('Predicted Parameters:', predicted_parameters[i])
    #     print()


a = 0.2 # fix this, change to V and delmsq31, delmsq21
theta13 = np.pi/20
theta12 = np.pi/6
theta23 = np.pi/4
deltacp = np.pi/6
alpha = 0.03
delta = lambda L, En: (1.27 * (2e-3) * L)/En

train_params = []
train_probs = []

num = 1000

def generate_data():
    a = np.random.uniform(0.1, 0.0001, num)
    theta13 = np.random.uniform(0, np.pi/2, num)
    theta12 = np.random.uniform(0, np.pi/2, num)
    theta23 = np.random.uniform(0, np.pi/2, num)
    deltacp = np.random.uniform(0, 2 * np.pi, num)

    alpha = np.array([0.03] * num)

    baseline = np.random.uniform(1e-2, 1e+2, num)
    energy = np.random.uniform(0.001, 0.1, num)

    # be_prod = np.array(np.meshgrid(baselines, energies)).T.reshape(-1,2)
    # deltas = [delta(l, en) for l, en in be_prod]

    params = [alpha, a, baseline, energy, deltacp, theta12, theta13, theta23]

    return params

data = generate_data()

for alpha, a, baseline, energy, deltacp, theta12, theta13, theta23 in zip(*data):
    try:
        prob = oscprobs.Identities(alpha, a, delta(baseline, energy), deltacp, theta12, theta13, theta23)

        probmatrix = prob.probabilities()
        train_params.append([alpha, a, baseline, energy, deltacp, theta12, theta13, theta23])
        train_probs.append([probmatrix.flatten()])
    except Exception:
        print([alpha, a, baseline, energy, deltacp, theta12, theta13, theta23])

# print(list(train_probs[len(train_probs) // 2]))
# print(list(train_params[len(train_params) // 2]))

# print(len(train_probs))

print("=" * 50)
print("Model training")

RNNmodel(np.array(train_probs), np.array(train_params))