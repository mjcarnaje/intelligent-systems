# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Load regular season data
file_paths = [
    "data/regular_season_box_scores_2010_2024_part_1.csv",
    "data/regular_season_box_scores_2010_2024_part_2.csv",
    "data/regular_season_box_scores_2010_2024_part_3.csv",
]
regular_season_data = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

# Filter out rows where there is a comment - It means that the player did not play
regular_season_data = regular_season_data[regular_season_data['comment'].isna()]

# Load Styles data
styles_data = pd.read_csv("data/basketball_players_styles.csv")

# Merge the datasets on player names
regular_season_data = pd.merge(
    regular_season_data,
    styles_data[['Player', 'Style']],
    left_on='personName',
    right_on='Player',
    how='left'
)

# Map 'Style' to numerical labels
style_mapping = {'Offensive': 0, 'Defensive': 1, 'Balanced': 2}
regular_season_data['style_label'] = regular_season_data['Style'].map(style_mapping)

# Drop rows with missing labels
regular_season_data = regular_season_data.dropna(subset=['style_label'])

# Convert 'style_label' to integer
regular_season_data['style_label'] = regular_season_data['style_label'].astype(int)

# Print the number of all data
print(f"Number of all data: {len(regular_season_data)}")

# Select features based on the provided schema
features = [
    'minutes', 'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage', 'threePointersMade', 'threePointersAttempted', 'threePointersPercentage', 'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage', 'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points', 'plusMinusPoints'
]

# Fill NaN values in position with 'Unknown'
regular_season_data['position'] = regular_season_data['position'].fillna('Unknown')

# One-hot encode the position
position_encoded = pd.get_dummies(regular_season_data['position'], prefix='pos')
regular_season_data = pd.concat([regular_season_data, position_encoded], axis=1)

# Add position columns to features
position_features = position_encoded.columns.tolist()
features.extend(position_features)

def time_to_float(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes + seconds / 60
    except (ValueError, AttributeError):
        return 0.0

# Convert minutes to float: '5:00' -> 5.0
regular_season_data['minutes'] = regular_season_data['minutes'].apply(time_to_float)

# Handle missing values by filling them with zeros
regular_season_data[features] = regular_season_data[features].fillna(0)

# Prepare the features and target variable
X_raw = regular_season_data[features].values
y_raw = regular_season_data['style_label'].values.reshape(-1, 1)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y_raw)

# Split the data into training, validation, and testing sets

# Ensure that the same player's data doesn't appear in both training and test sets
# Group by 'personName' and split the groups
player_groups = regular_season_data.groupby('personName')
players = list(player_groups.groups.keys())

train_players, temp_players = train_test_split(
    players, test_size=0.3, random_state=42
)
val_players, test_players = train_test_split(
    temp_players, test_size=0.5, random_state=42
)

train_data = regular_season_data[regular_season_data['personName'].isin(train_players)]
val_data = regular_season_data[regular_season_data['personName'].isin(val_players)]
test_data = regular_season_data[regular_season_data['personName'].isin(test_players)]

X_train_raw = train_data[features].values
y_train_raw = train_data['style_label'].values.reshape(-1, 1)
y_train_onehot = encoder.transform(y_train_raw)

X_val_raw = val_data[features].values
y_val_raw = val_data['style_label'].values.reshape(-1, 1)
y_val_onehot = encoder.transform(y_val_raw)

X_test_raw = test_data[features].values
y_test_raw = test_data['style_label'].values.reshape(-1, 1)
y_test_onehot = encoder.transform(y_test_raw)

# Standardize features using z-score normalization
scaler = StandardScaler()

# Fit the scaler on training data only to prevent data leakage
X_train = scaler.fit_transform(X_train_raw)

# Transform validation and test data using the fitted scaler
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# Implementing a Neural Network from scratch using NumPy

# Define the network architecture
input_size = X_train.shape[1]  # Number of features
hidden_sizes = [64, 32]        # List of hidden layer sizes
output_size = 3                # Number of classes

# Initialize weights and biases
np.random.seed(42)
parameters = {}
layer_sizes = [input_size] + hidden_sizes + [output_size]

for l in range(1, len(layer_sizes)):
    parameters['W' + str(l)] = np.random.randn(layer_sizes[l-1], layer_sizes[l]) * np.sqrt(2. / layer_sizes[l-1])
    parameters['b' + str(l)] = np.zeros((1, layer_sizes[l]))

# Activation functions and their derivatives
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Loss function with L2 regularization
def compute_loss(y_true, y_pred, parameters, lambd):
    m = y_true.shape[0]
    epsilon = 1e-8  # Small value to prevent log(0)
    data_loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    reg_loss = 0
    for l in range(1, len(layer_sizes)):
        reg_loss += np.sum(np.square(parameters['W' + str(l)]))
    reg_loss = (lambd / (2 * m)) * reg_loss
    return data_loss + reg_loss

# Forward pass
def forward_pass(X, parameters):
    caches = {}
    A = X
    L = len(layer_sizes) - 1  # Number of layers

    for l in range(1, L):
        Z = np.dot(A, parameters['W' + str(l)]) + parameters['b' + str(l)]
        A = relu(Z)
        caches['A' + str(l)] = A
        caches['Z' + str(l)] = Z

    # Output layer
    ZL = np.dot(A, parameters['W' + str(L)]) + parameters['b' + str(L)]
    AL = softmax(ZL)
    caches['A' + str(L)] = AL
    caches['Z' + str(L)] = ZL

    return AL, caches

# Backward pass
def backward_pass(X, y_true, parameters, caches, lambd):
    grads = {}
    m = X.shape[0]
    L = len(layer_sizes) - 1  # Number of layers

    # Output layer gradients
    dZL = caches['A' + str(L)] - y_true
    grads['dW' + str(L)] = np.dot(caches['A' + str(L-1)].T, dZL) / m + (lambd / m) * parameters['W' + str(L)]
    grads['db' + str(L)] = np.sum(dZL, axis=0, keepdims=True) / m

    # Propagate through hidden layers
    for l in reversed(range(1, L)):
        dA = np.dot(dZL, parameters['W' + str(l+1)].T)
        dZ = dA * relu_derivative(caches['Z' + str(l)])
        if l == 1:
            A_prev = X
        else:
            A_prev = caches['A' + str(l-1)]
        grads['dW' + str(l)] = np.dot(A_prev.T, dZ) / m + (lambd / m) * parameters['W' + str(l)]
        grads['db' + str(l)] = np.sum(dZ, axis=0, keepdims=True) / m
        dZL = dZ

    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(layer_sizes) - 1  # Number of layers
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters

# Implementing the Adam optimizer
def initialize_adam(parameters):
    L = len(layer_sizes) - 1
    v = {}
    s = {}
    for l in range(1, L + 1):
        v['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        v['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])
        s['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        s['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])
    return v, s

def update_parameters_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
    L = len(layer_sizes) - 1
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        # Moving average of the gradients
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]

        # Compute bias-corrected first moment estimate
        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - beta1 ** t)
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - beta1 ** t)

        # Moving average of the squared gradients
        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * (grads['dW' + str(l)] ** 2)
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * (grads['db' + str(l)] ** 2)

        # Compute bias-corrected second raw moment estimate
        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - beta2 ** t)
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - beta2 ** t)

        # Update parameters
        parameters['W' + str(l)] -= learning_rate * v_corrected['dW' + str(l)] / (np.sqrt(s_corrected['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] -= learning_rate * v_corrected['db' + str(l)] / (np.sqrt(s_corrected['db' + str(l)]) + epsilon)

    return parameters, v, s

# Calculate class weights to handle class imbalance
def compute_class_weights(y):
    class_counts = np.bincount(y.flatten())
    total_samples = len(y)
    num_classes = len(class_counts)
    class_weights = {}
    for i in range(num_classes):
        class_weights[i] = total_samples / (num_classes * class_counts[i])
    return class_weights

class_weights = compute_class_weights(y_train_raw)

# Training parameters
learning_rate = 0.001
num_epochs = 100
batch_size = 64
lambd = 0.01  # Regularization parameter

# Adam optimizer parameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Initialize Adam optimizer variables
v, s = initialize_adam(parameters)
t = 0  # Initialization of time step

best_val_loss = np.inf
patience_counter = 0
patience = 10  # Early stopping patience

# Training the network
for epoch in range(num_epochs):
    # Shuffle the training data
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train_onehot[permutation]
    y_train_labels_shuffled = y_train_raw[permutation].flatten()

    # Mini-batch gradient descent
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        y_batch_labels = y_train_labels_shuffled[i:i+batch_size]

        # Forward pass
        AL, caches = forward_pass(X_batch, parameters)

        # Compute loss with class weights
        sample_weights = np.array([class_weights[label] for label in y_batch_labels])
        loss = compute_loss(y_batch, AL, parameters, lambd)
        weighted_loss = loss * sample_weights.mean()

        # Backward pass
        grads = backward_pass(X_batch, y_batch, parameters, caches, lambd)

        # Update time step
        t += 1

        # Update weights and biases using Adam optimizer
        parameters, v, s = update_parameters_adam(
            parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon
        )

    # Compute training loss
    AL_train, _ = forward_pass(X_train, parameters)
    train_loss = compute_loss(y_train_onehot, AL_train, parameters, lambd)

    # Compute validation loss
    AL_val, _ = forward_pass(X_val, parameters)
    val_loss = compute_loss(y_val_onehot, AL_val, parameters, lambd)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model parameters
        best_parameters = parameters.copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Optionally, print loss every few epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Load the best model parameters
parameters = best_parameters

# Evaluate the model on the validation set
AL_val, _ = forward_pass(X_val, parameters)
y_pred_val = np.argmax(AL_val, axis=1)
y_val_true = y_val_raw.flatten()

# Calculate evaluation metrics
val_accuracy = accuracy_score(y_val_true, y_pred_val)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print("Validation Classification Report:")
print(classification_report(y_val_true, y_pred_val, target_names=style_mapping.keys()))

# Evaluate the model on the test set
AL_test, _ = forward_pass(X_test, parameters)
y_pred_test = np.argmax(AL_test, axis=1)
y_test_true = y_test_raw.flatten()

test_accuracy = accuracy_score(y_test_true, y_pred_test)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:")
print(classification_report(y_test_true, y_pred_test, target_names=style_mapping.keys()))
