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

# Load the dataset
try:
    data = pd.read_csv("train.txt", sep=';')
    data.columns = ["Text", "Emotions"]
except FileNotFoundError:
    print("Error: 'train.txt' not found. Make sure it's in the correct directory.")
    exit()

# Extract texts and labels
texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Determine the maximum sequence length
max_length = max(len(seq) for seq in sequences)

# Pad sequences to ensure uniform length
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Encode the labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# One-hot encode the labels
one_hot_labels = to_categorical(encoded_labels)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(
    padded_sequences, one_hot_labels, test_size=0.2, random_state=42
)

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=128, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=one_hot_labels.shape[1], activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
epochs = 10
batch_size = 32
model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_data=(xtest, ytest))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(xtest, ytest)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Function to predict emotion from user input
def predict_emotion(text):
    # 1. Preprocess the input text
    user_sequence = tokenizer.texts_to_sequences([text])
    padded_user_sequence = pad_sequences(user_sequence, maxlen=max_length)

    # 2. Make the prediction
    predictions = model.predict(padded_user_sequence)

    # 3. Decode the prediction
    predicted_class_index = np.argmax(predictions)
    predicted_emotion = label_encoder.inverse_transform([predicted_class_index])[0]

    return predicted_emotion

# Get user input and predict
while True:
    user_input = input("\nEnter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predicted_emotion = predict_emotion(user_input)
    print(f"Predicted emotion: {predicted_emotion}")