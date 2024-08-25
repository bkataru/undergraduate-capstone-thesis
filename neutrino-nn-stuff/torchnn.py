import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from snake.activations import Snake

from pytrino import oscprobs

# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()

#     def forward(self, outputs_pred, outputs):
        
#         outputs_pred = outputs_pred.squeeze()

#         # alpha_pred, a_pred, baseline_pred, energy_pred, deltacp_pred, theta12_pred, theta13_pred, theta23_pred = [outputs_pred[:, i] for i in range(0, 8)]
#         # alpha, a, baseline, energy, deltacp, theta12, theta13, theta23 = [outputs[:, i] for i in range(0, 8)]
        
#         # loss_alpha = torch.abs(alpha_pred - alpha)

#         # loss_a = torch.abs(a_pred - a)
#         # loss_baseline = torch.abs(baseline_pred - baseline)
#         # loss_energy = torch.abs(energy_pred - energy)
#         # loss_deltacp = torch.abs(deltacp_pred - deltacp)
#         # loss_theta12 = torch.abs(theta12_pred - theta12)
#         # loss_theta13 = torch.abs(theta13_pred - theta13)
#         # loss_theta23 = torch.abs(theta23_pred - theta23)
        
#         # loss = loss_alpha.mean() + loss_a.mean() + loss_baseline.mean() + loss_energy.mean() + loss_deltacp.mean() + loss_theta12.mean() + loss_theta13.mean() + loss_theta23.mean()
        
#         diff = torch.abs(torch.sin(outputs_pred - outputs))
#         loss = torch.mean(diff)
#         return loss 

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predictions, targets):
        predictions = predictions.squeeze()
        absolute_errors = torch.abs(targets - predictions)
        percentage_errors = absolute_errors / torch.abs(targets)
        mape = torch.mean(percentage_errors) * 100.0
        return mape

def torchModel(probabilities, parameters):
    # Define the architecture of the feedforward neural network
# Define the architecture of the feedforward neural network
    class OscillationModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(OscillationModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)

            return out

    # Prepare your dataset: inputs (probabilities) and outputs (oscillation parameters)
    inputs = probabilities  # Example input dataset of 1000 samples, each with 9 probabilities
    outputs = parameters  # Example output dataset of 1000 samples, each with 8 oscillation parameters

    # Convert the numpy arrays to PyTorch tensors
    inputs = torch.from_numpy(inputs).float()
    outputs = torch.from_numpy(outputs).float()

    # Define the hyperparameters
    input_size = 9
    hidden_size = 512
    output_size = 7
    learning_rate = 0.001
    num_epochs = 5000

    # Create an instance of the OscillationModel
    model = OscillationModel(input_size, hidden_size, output_size)

    # Define the loss function and optimizer
    criterion = MAPELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # reduce_lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs_pred = model(inputs)

        # Compute the loss
        loss = criterion(outputs_pred, outputs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # reduce_lr_scheduler.step(loss)  # ReduceLROnPlateau scheduler
        # step_lr_scheduler.step()  # StepLR scheduler

        # Print the loss at every epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # After training, you can use the trained model to make predictions
    # For example, given a new set of probabilities, you can predict the corresponding oscillation parameters
    new_inputs = np.array([[0.88420576, 0.01198492, 0.10380932, 0.01242602, 0.98623742, 0.00133656, 0.10336822, 0.00177766, 0.89485412]])  # Example new input with 1 sample and 9 probabilities

    # Convert new_inputs to a PyTorch tensor
    new_inputs = torch.from_numpy(new_inputs).float()

    # Pass new_inputs through the model
    predicted_outputs = model(new_inputs)
    print("Predicted Oscillation Parameters:")
    print(predicted_outputs.detach().numpy())

a = 0.2 # fix this, change to delmsq21, delmsq31
theta13 = np.pi/20
theta12 = np.pi/6
theta23 = np.pi/4
deltacp = np.pi/6
alpha = 0.03
delta = lambda L, En: (1.27 * (2e-3) * L)/En # THIS IS WRONG THERE ARE TWO MASS SQUARED DIFFERENCES RIGHT?

train_params = []
train_probs = []

num = 1000

def generate_data():
    a = np.random.uniform(0.1, 0.0001, num)
    theta13 = np.random.uniform(0, np.pi/2, num)
    theta12 = np.random.uniform(0, np.pi/2, num)
    theta23 = np.random.uniform(0, np.pi/2, num)
    deltacp = np.random.uniform(0, 2 * np.pi, num)

    baseline = np.random.uniform(1e-2, 1e+2, num)
    energy = np.random.uniform(0.001, 0.1, num)

    # be_prod = np.array(np.meshgrid(baselines, energies)).T.reshape(-1,2)
    # deltas = [delta(l, en) for l, en in be_prod]

    params = [a, baseline, energy, deltacp, theta12, theta13, theta23]

    return params

data = generate_data()

for a, baseline, energy, deltacp, theta12, theta13, theta23 in zip(*data):
    try:
        prob = oscprobs.Identities(alpha, a, delta(baseline, energy), deltacp, theta12, theta13, theta23)

        probmatrix = prob.probabilities()
        train_params.append([a, baseline, energy, deltacp, theta12, theta13, theta23])
        train_probs.append([probmatrix.flatten()])
    except Exception:
        print([a, baseline, energy, deltacp, theta12, theta13, theta23])

print(list(train_probs[0]))
print(list(train_params[0]))

# print(len(train_probs))

print("=" * 50)
print("Model training")

torchModel(np.array(train_probs), np.array(train_params))