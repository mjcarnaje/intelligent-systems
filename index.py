import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import matplotlib.pyplot as plt


# ------------------------------
# 1. DATA LOADING AND PREP
# ------------------------------
file_paths = [
    "data/regular_season_box_scores_2010_2024_part_1.csv",
    "data/regular_season_box_scores_2010_2024_part_2.csv",
    "data/regular_season_box_scores_2010_2024_part_3.csv",
]
regular_season_data = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

# Filter out rows where there is a comment => means player did not play
regular_season_data = regular_season_data[regular_season_data['comment'].isna()]

# Load style data
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

print(f"Number of all data (row-level): {len(regular_season_data)}")

# ------------------------------
# OPTIONAL: CREATE ADVANCED STATS (Example: True Shooting %)
# ------------------------------
def time_to_float(time_str):
    """Converts a time string 'MM:SS' to float minutes."""
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes + seconds / 60
    except (ValueError, AttributeError):
        return 0.0

regular_season_data['minutes'] = regular_season_data['minutes'].apply(time_to_float).fillna(0)

# True Shooting% = Points / [2 * (FGA + 0.44 * FTA)]
# Make sure these columns exist in your data
regular_season_data['TS%'] = 0.0
mask = (regular_season_data['fieldGoalsAttempted'] + 0.44 * regular_season_data['freeThrowsAttempted']) != 0
regular_season_data.loc[mask, 'TS%'] = (
    regular_season_data.loc[mask, 'points'] / 
    (2 * (regular_season_data.loc[mask, 'fieldGoalsAttempted'] 
           + 0.44 * regular_season_data.loc[mask, 'freeThrowsAttempted']))
)

# ------------------------------
# 2. WEIGHTED AVERAGING BY MINUTES AT THE PLAYER LEVEL
# ------------------------------
# Fill missing numeric values with 0 just in case
numeric_cols = [
    'minutes',
    'fieldGoalsAttempted', 
    'fieldGoalsMade', 
    'fieldGoalsPercentage',
    'threePointersAttempted', 
    'threePointersMade', 
    'threePointersPercentage',
    'freeThrowsAttempted', 
    'freeThrowsMade', 
    'freeThrowsPercentage',
    'reboundsOffensive', 
    'reboundsDefensive', 
    'reboundsTotal',
    'assists', 
    'steals', 
    'blocks', 
    'turnovers', 
    'foulsPersonal', 
    'points',
    'plusMinusPoints', 
    'TS%' 
]
for col in numeric_cols:
    regular_season_data[col] = regular_season_data[col].fillna(0)

regular_season_data['position'] = regular_season_data['position'].fillna('Unknown')

# -- Weighted average aggregator --
def weighted_mean(series, weights):
    return 0 if weights.sum() == 0 else np.average(series, weights=weights)

def aggregate_player_stats(player_group):
    minutes_played = player_group['minutes']
    
    # Calculate weighted averages for all numeric stats
    aggregated_stats = {
        col: weighted_mean(player_group[col], minutes_played) 
        for col in numeric_cols
    }
    
    # Add non-numeric fields
    aggregated_stats.update({
        'style_label': player_group['style_label'].iloc[0],  # Take first style label
        'position': player_group['position'].mode()[0],      # Take most common position
        'personName': player_group['personName'].iloc[0]     # Keep player name
    })
    
    return pd.Series(aggregated_stats)

# Group data by player and calculate aggregated stats
grouped_data = (regular_season_data.groupby('personName')
                                 .apply(aggregate_player_stats)
                                 .reset_index(drop=True))

print(f"Number of data after grouping by player: {len(grouped_data)}")

# show pie graph of style_label
plt.figure(figsize=(10, 6))
style_counts = grouped_data['style_label'].value_counts()
plt.pie(style_counts, labels=style_counts.index, autopct='%1.1f%%')
plt.title('Style Distribution')
plt.legend(style_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()

# ------------------------------ 
# 3. OVERSAMPLING THE DATA
# ------------------------------
from imblearn.over_sampling import SMOTE

# Store non-numeric columns before SMOTE
non_numeric_cols = ['personName', 'position']
numeric_data = grouped_data.drop(columns=non_numeric_cols)
labels = grouped_data['style_label']

# Apply SMOTE only on numeric features
smote = SMOTE(random_state=42)
numeric_resampled, labels_resampled = smote.fit_resample(numeric_data, labels)

# Create new personName for synthetic samples
original_players = grouped_data['personName'].values
synthetic_players = [f"SYNTHETIC_PLAYER_{i}" for i in range(len(labels_resampled) - len(labels))]
all_players = np.concatenate([original_players, synthetic_players])

# Create new positions for synthetic samples
# Use random sampling from original positions for synthetic samples
original_positions = grouped_data['position'].values
synthetic_positions = np.random.choice(original_positions, size=len(synthetic_players))
all_positions = np.concatenate([original_positions, synthetic_positions])

# Reconstruct the DataFrame with both original and synthetic samples
grouped_data = pd.DataFrame(numeric_resampled, columns=numeric_data.columns)
grouped_data['personName'] = all_players
grouped_data['position'] = all_positions
grouped_data['style_label'] = labels_resampled

# show pie graph of style_label
plt.figure(figsize=(10, 6))
style_counts = grouped_data['style_label'].value_counts()
plt.pie(style_counts, labels=[k for k,v in style_mapping.items() if v in style_counts.index], 
        autopct='%1.1f%%')
plt.title('Style Distribution After SMOTE')
plt.legend([k for k,v in style_mapping.items() if v in style_counts.index], 
          loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()

# We'll keep the numeric_cols plus the one-hot positions
position_encoded = pd.get_dummies(grouped_data['position'], prefix='pos')
grouped_data = pd.concat([grouped_data, position_encoded], axis=1)

# Final list of features
position_features = list(position_encoded.columns)
all_features = numeric_cols + position_features

X_raw = grouped_data[all_features].values
y_raw = grouped_data['style_label'].values.reshape(-1, 1)

# One-hot encode the target
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y_raw)

# ------------------------------
# 4. TRAIN/VAL/TEST SPLIT (BY PLAYER)
# ------------------------------
players = grouped_data['personName'].unique().tolist()

train_players, temp_players = train_test_split(players, test_size=0.3, random_state=42)
val_players, test_players = train_test_split(temp_players, test_size=0.5, random_state=42)

train_data = grouped_data[grouped_data['personName'].isin(train_players)]
val_data   = grouped_data[grouped_data['personName'].isin(val_players)]
test_data  = grouped_data[grouped_data['personName'].isin(test_players)]

X_train_raw = train_data[all_features].values
y_train_raw = train_data['style_label'].values.reshape(-1, 1)
y_train_onehot = encoder.transform(y_train_raw)

X_val_raw = val_data[all_features].values
y_val_raw = val_data['style_label'].values.reshape(-1, 1)
y_val_onehot = encoder.transform(y_val_raw)

X_test_raw = test_data[all_features].values
y_test_raw = test_data['style_label'].values.reshape(-1, 1)
y_test_onehot = encoder.transform(y_test_raw)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

# ------------------------------
# 5. CLASS WEIGHTS FOR IMBALANCE
# ------------------------------
def compute_class_weights(y):
    """Compute class weights as total_samples / (num_classes * class_counts[i])."""
    class_counts = np.bincount(y.flatten())
    total_samples = len(y)
    num_classes = len(class_counts)
    class_weights_ = {}
    for i in range(num_classes):
        class_weights_[i] = total_samples / (num_classes * class_counts[i])
    return class_weights_

class_weights = compute_class_weights(y_train_raw)

# ------------------------------
# 6. NEURAL NETWORK FROM SCRATCH
# ------------------------------
np.random.seed(42)

# Adjust the network architecture or try bigger layers
input_size = X_train.shape[1]  
hidden_sizes = [128, 64, 32]   # Larger than before
output_size = 3                
layer_sizes = [input_size] + hidden_sizes + [output_size]

# Hyperparameters
learning_rate = 0.0005
lambd = 0.001
num_epochs = 100
batch_size = 32
dropout_rate = 0.3

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Early stopping
patience = 15
best_val_loss = np.inf
patience_counter = 0

# -------------------------
# WEIGHTS INITIALIZATION
# -------------------------
parameters = {}
for l in range(1, len(layer_sizes)):
    # He initialization
    parameters['W' + str(l)] = (
        np.random.randn(layer_sizes[l-1], layer_sizes[l]) 
        * np.sqrt(2. / layer_sizes[l-1])
    )
    parameters['b' + str(l)] = np.zeros((1, layer_sizes[l]))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# -------------------------
# FORWARD PASS
# -------------------------
def forward_pass(X, parameters, keep_prob=1.0, training=True):
    caches = {}
    A = X
    L = len(layer_sizes) - 1

    for l in range(1, L):
        Z = np.dot(A, parameters['W' + str(l)]) + parameters['b' + str(l)]
        A = relu(Z)
        
        if training:
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob
            caches['D' + str(l)] = D

        caches['A' + str(l)] = A
        caches['Z' + str(l)] = Z

    ZL = np.dot(A, parameters['W' + str(L)]) + parameters['b' + str(L)]
    AL = softmax(ZL)
    caches['A' + str(L)] = AL
    caches['Z' + str(L)] = ZL

    return AL, caches

# -------------------------
# WEIGHTED CROSS-ENTROPY
# -------------------------
def compute_loss(y_true, y_pred, parameters, lambd, sample_weights=None):
    m = y_true.shape[0]
    epsilon = 1e-8

    if sample_weights is not None:
        ce_individual = -np.sum(y_true * np.log(y_pred + epsilon), axis=1)
        weighted_ce = np.sum(sample_weights * ce_individual) / (np.sum(sample_weights) + 1e-8)
        data_loss = weighted_ce
    else:
        data_loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m

    # L2 regularization
    reg_loss = 0
    L = len(layer_sizes) - 1
    for l in range(1, L + 1):
        reg_loss += np.sum(np.square(parameters['W' + str(l)]))
    reg_loss = (lambd / (2 * m)) * reg_loss

    return data_loss + reg_loss

# -------------------------
# BACKWARD PASS
# -------------------------
def backward_pass(X, y_true, parameters, caches, lambd, sample_weights=None, keep_prob=1.0):
    grads = {}
    m = X.shape[0]
    L = len(layer_sizes) - 1

    AL = caches['A' + str(L)]
    dZL = AL - y_true
    
    if sample_weights is not None:
        dZL = dZL * sample_weights.reshape(-1, 1)

    A_prev = caches['A' + str(L-1)]
    grads['dW' + str(L)] = (np.dot(A_prev.T, dZL) / m) + (lambd / m) * parameters['W' + str(L)]
    grads['db' + str(L)] = np.sum(dZL, axis=0, keepdims=True) / m

    dA_prev = np.dot(dZL, parameters['W' + str(L)].T)

    for l in reversed(range(1, L)):
        Z = caches['Z' + str(l)]
        dZ = dA_prev * relu_derivative(Z)

        if ('D' + str(l)) in caches:
            D = caches['D' + str(l)]
            dZ = dZ * D
            dZ = dZ / keep_prob

        A_prev = X if l == 1 else caches['A' + str(l-1)]

        grads['dW' + str(l)] = (np.dot(A_prev.T, dZ) / m) + (lambd / m) * parameters['W' + str(l)]
        grads['db' + str(l)] = np.sum(dZ, axis=0, keepdims=True) / m

        dA_prev = np.dot(dZ, parameters['W' + str(l)].T)

    return grads

# -------------------------
# ADAM OPTIMIZER
# -------------------------
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

        # Bias-corrected first moment
        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - beta1 ** t)
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - beta1 ** t)

        # Moving average of the squared gradients
        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * (grads['dW' + str(l)] ** 2)
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * (grads['db' + str(l)] ** 2)

        # Bias-corrected second moment
        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - beta2 ** t)
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - beta2 ** t)

        # Update
        parameters['W' + str(l)] -= learning_rate * v_corrected['dW' + str(l)] / (np.sqrt(s_corrected['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] -= learning_rate * v_corrected['db' + str(l)] / (np.sqrt(s_corrected['db' + str(l)]) + epsilon)

    return parameters, v, s

v, s = initialize_adam(parameters)
t = 0
best_parameters = None
keep_prob = 1.0 - dropout_rate  # e.g. dropout=0.3 => keep_prob=0.7

# -------------------------
# 7. TRAINING LOOP
# -------------------------
for epoch in range(num_epochs):
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train_onehot[permutation]
    y_labels_shuffled = y_train_raw[permutation].flatten()

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        y_batch_labels = y_labels_shuffled[i:i+batch_size]

        # Forward pass (training mode with dropout)
        AL, caches = forward_pass(X_batch, parameters,
                                  keep_prob=keep_prob,
                                  training=True)
        
        # Compute weighted loss
        sample_w = np.array([class_weights[label] for label in y_batch_labels])
        loss = compute_loss(y_batch, AL, parameters, lambd, sample_weights=sample_w)

        # Backward pass
        grads = backward_pass(X_batch, y_batch, parameters, caches, lambd,
                              sample_weights=sample_w, keep_prob=keep_prob)

        # Adam update
        t += 1
        parameters, v, s = update_parameters_adam(
            parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon
        )

    # Evaluate on full train and val (no dropout)
    AL_train, _ = forward_pass(X_train, parameters, keep_prob=1.0, training=False)
    train_loss = compute_loss(y_train_onehot, AL_train, parameters, lambd)

    AL_val, _ = forward_pass(X_val, parameters, keep_prob=1.0, training=False)
    val_loss = compute_loss(y_val_onehot, AL_val, parameters, lambd)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_parameters = {k: v.copy() for k, v in parameters.items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Restore best parameters
parameters = best_parameters

# -------------------------
# 8. EVALUATION
# -------------------------
# Validation
AL_val, _ = forward_pass(X_val, parameters, keep_prob=1.0, training=False)
y_pred_val = np.argmax(AL_val, axis=1)
y_val_true = y_val_raw.flatten()
val_accuracy = accuracy_score(y_val_true, y_pred_val)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print("Validation Classification Report:")
print(classification_report(y_val_true, y_pred_val, target_names=style_mapping.keys()))

# Test
AL_test, _ = forward_pass(X_test, parameters, keep_prob=1.0, training=False)
y_pred_test = np.argmax(AL_test, axis=1)
y_test_true = y_test_raw.flatten()
test_accuracy = accuracy_score(y_test_true, y_pred_test)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:")
print(classification_report(y_test_true, y_pred_test, target_names=style_mapping.keys()))
