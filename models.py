import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import zlib
from math import log2
from scipy.fft import fft, ifft
from scipy.special import erfc

# File path
file_path = '/Users/shreykhater/YQuantumHackathon/AI_2qubits_training_data.txt'

# Read the data from the file
data = []
with open(file_path, 'r') as file:
    for line in file:
        if line.strip():
            binary_number, label = line.strip().split()
            data.append((binary_number, int(label)))

# Convert the data into a DataFrame
df = pd.DataFrame(data, columns=['binary_number', 'label'])

# concatenate optimal number of input rows for highest accuracy (pros and cons)
num_concats = 4

new_df = pd.DataFrame({'Concatenated_Data': [''] * (len(df) // num_concats), 'label': [''] * (len(df) // num_concats)})

# Loop through each group of 10 rows and concatenate their 'Data' strings
for i in range(0, len(df), num_concats):
    new_df.iloc[i // num_concats, 0] = ''.join(df['binary_number'][i:i+num_concats])
    new_df.iloc[i // num_concats, 1] = df['label'][i]

print(new_df)


def calculate_2bit_shannon_entropy(binary_string):
    # Ensure the string length is a multiple of 2 for exact 2-bit grouping
    if len(binary_string) % 4 != 0:
        raise ValueError("Binary string length must be a multiple of 2.")
    
    # Define possible 2-bit combinations
    #patterns = ['0000', '1000', '1100', '1110', '1111', '0100', '0110', '0111', '0010', '0011', '0001', '1001', '1101', '0110', '0101', '1010']
    patterns = ['00', '10', '11', '01']
    frequency = {pattern: 0 for pattern in patterns}
    
    # Count frequency of each pattern
    for i in range(0, len(binary_string), 2):
        segment = binary_string[i:i+2]
        if segment in patterns:
            frequency[segment] += 1
    
    # Calculate total segments counted
    total_segments = sum(frequency.values())
    
    # Calculate probabilities and entropy
    entropy = 0
    for count in frequency.values():
        if count > 0:
            probability = count / total_segments
            entropy -= probability * log2(probability)
    
    return entropy

print(calculate_2bit_shannon_entropy('0101010101010101'))


def classic_spectral_test(bit_string):
    """
    Perform the classic spectral test using Discrete Fourier Transform (DFT)
    on the input bit string.

    Args:
        bit_string (str): A string of 0s and 1s representing the bit sequence.

    Returns:
        float: The P-value of the test.
    """
    # Convert bit string to numpy array of -1 and 1
    bit_array = 2 * np.array([int(bit) for bit in bit_string]) - 1

    # Compute the DFT of the bit array
    dft = fft(bit_array)

    # Calculate the number of bits for the first half of the DFT output
    n_half = len(bit_string) // 2 + 1

    # Compute the modulus of the first half of the DFT output
    mod_dft = np.abs(dft[:n_half])

    # Compute the 95% peak height threshold
    threshold = np.sqrt(np.log(1 / 0.05) / len(bit_string))

    # Count the number of peaks below the threshold
    peaks_below_threshold = np.sum(mod_dft < threshold)

    # Compute the expected number of peaks
    expected_peaks = 0.95 * n_half

    # Compute the test statistic
    d = (peaks_below_threshold - expected_peaks) / np.sqrt(len(bit_string) * 0.95 * 0.05)

    # Compute the P-value using the complementary error function
    p_value = erfc(np.abs(d) / np.sqrt(2)) / 2

    return d

print(classic_spectral_test('0100111111110000000000101110100011011011110000110101001001111101000110011111011100111010001111110010011001101111000111111000100000111001111001110011010000110001110110010000110111111101000000011000110111101001011101111111011011001101011111010100010110111100110100010011011001101010111101100001100000111101000011100100011101001110111100011110100111010100010001111100111111010100011001101111011101111101000000000010010100110110000010100010111011000101011001011011011001001001011100000100111011001111001100011111110111111000110111100001001100100111010011001000011000001011110001001010011110111100110110010001110101110111111100000111111111011010111010110101110101011010000011011011001001100011000111110011101011111101010111101011110101001001011000110011111110010010000100111111001100101110111011110000111010100001011001111111001111011111111111010101100011011110100100111101001111000000101000011111100101010001010010101101101100100000011101011101011001000001101101110111100110110000111111010111101000100111'))

#new_df['spectral_randomness'] = new_df['Concatenated_Data'].apply(classic_spectral_test)

new_df['shannon_entropy'] = new_df['Concatenated_Data'].apply(calculate_2bit_shannon_entropy)

#print(new_df['spectral_randomness'])


'''def compression_complexity(data):
    # Convert binary string data to bytes if it's not already in byte form
    if isinstance(data, str):
        data = data.encode('utf-8')

    compressed_data = zlib.compress(data)
    compressed_length = len(compressed_data)
    return compressed_length
df['compression_complexity'] = df['binary_number'].apply(compression_complexity)
'''


# Get the number of rows and columns in the DataFrame
#num_rows, num_columns = df.shape

#print("Number of rows:", num_rows)
#print("Number of columns:", num_columns)

# Preprocess the binary_number column to convert each bit to a separate feature column
df_features = pd.DataFrame(new_df['Concatenated_Data'].apply(list).tolist())
new_df = pd.concat([new_df.drop(columns='Concatenated_Data'), df_features], axis=1)

#print(df)

#print(df.head(10))


# Split the data into features (X) and labels (y)
X = new_df.drop(columns='label').values
#print(X)
y = new_df['label'].values
y=y.astype('int')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_test, y_test)

def gradient_boosting():
    # Create the Gradient Boosting classifier
    gb_model = GradientBoostingClassifier(random_state=42)

    # Train the model
    gb_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_gb = gb_model.predict(X_test)

    # Calculate the accuracy of the Gradient Boosting model
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print("Gradient Boosting Accuracy:", accuracy_gb)

def gradient_boosting_grid_search():
    gb_model = GradientBoostingClassifier(random_state=42)

    # Define the hyperparameter grid for Grid Search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }

    # Perform Grid Search with cross-validation (cv=5) to find the best hyperparameters
    grid_search = GridSearchCV(gb_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_model = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    y_pred_gb = best_model.predict(X_test)

    # Calculate the accuracy of the Gradient Boosting model with the best hyperparameters
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print("Gradient Boosting Accuracy:", accuracy_gb)

gradient_boosting()