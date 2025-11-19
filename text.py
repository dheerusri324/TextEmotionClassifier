# Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
# Add these lines near your other sklearn imports
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    data = pd.read_csv("train.txt", sep=';')
    data.columns = ["Text", "Emotions"]
    data.dropna(inplace=True)  # <--- ADD THIS LINE TO REMOVE BLANK ROWS
except FileNotFoundError:
    print("Error: 'train.txt' not found. Make sure it's in the correct directory.")
    exit()

# Check the balance of the dataset
print("Dataset balance:")
print(data['Emotions'].value_counts())
print("-" * 50)

# Extract texts and labels
texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Determine the maximum sequence length
max_length = max(len(seq) for seq in sequences)

# Pad sequences to ensure uniform length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encode the labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)



from imblearn.over_sampling import RandomOverSampler

# 1. Define the oversampler
ros = RandomOverSampler(random_state=42)

# 2. Resample the padded sequences and encoded labels
print("\nBalancing the dataset with RandomOverSampler...")
resampled_sequences, resampled_labels = ros.fit_resample(padded_sequences, encoded_labels)

# 3. Check the new balance of the labels
print("New dataset balance:")
# We need to temporarily convert the resampled integer labels back to text labels to count them
temp_df = pd.DataFrame({'Emotions': label_encoder.inverse_transform(resampled_labels)})
print(temp_df['Emotions'].value_counts())
print("-" * 50)



# One-hot encode the NEW resampled labels
one_hot_labels = to_categorical(resampled_labels)

# Split the NEW resampled data into training and testing sets
xtrain, xtest, ytrain, ytest = train_tst_split(
    resampled_sequences, one_hot_labels, test_size=0.2, random_state=42
)

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=128))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.5)) # Added Dropout layer to combat overfitting
model.add(Dense(units=one_hot_labels.shape[1], activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print the model summary
model.summary()

# Train the model
print("\nStarting model training...")
epochs = 10
batch_size = 32
history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_data=(xtest, ytest))
print("Model training finished.")

# Evaluate the model on the test set
print("\nEvaluating model on the test set...")
loss, accuracy = model.evaluate(xtest, ytest)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("-" * 50)

print("\nGenerating Confusion Matrix...")
# 1. Get the model's predictions for the test set
y_pred_probs = model.predict(xtest)
# 2. Convert prediction probabilities into single class labels (like 0, 1, 2)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
# 3. Convert the one-hot encoded true labels (ytest) back into single class labels
y_true_labels = np.argmax(ytest, axis=1)
# 4. Get the actual class names (e.g., 'joy', 'anger')
class_names = label_encoder.classes_
# 5. Generate the confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
# 6. Display the confusion matrix using a heatmap for better readability
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("-" * 50)

# Function to predict emotion from user input
def predict_emotion(text):
    # Preprocess the input text
    user_sequence = tokenizer.texts_to_sequences([text])
    padded_user_sequence = pad_sequences(user_sequence, maxlen=max_length, padding='post')
    # Make the prediction
    predictions = model.predict(padded_user_sequence)
    # Decode the prediction
    predicted_class_index = np.argmax(predictions)
    predicted_emotion = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_emotion

# Interactive loop for live predictions
while True:
    user_input = input("\nEnter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predicted_emotion = predict_emotion(user_input)
    print(f"--> Predicted emotion: {predicted_emotion}")