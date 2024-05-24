import os
import json
import cv2
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import keras
import pandas as pd
from typing import List, Tuple
import random



# train_path = r"C:\Users\Theo\Documents\Advanced Analytics\train_images"
# test_path = r"C:\Users\Theo\Documents\Advanced Analytics\testimages"


# Step 1: Read the JSON file and extract image filenames and corresponding tags
def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    image_data = []
    for game in data:
        for screenshot in game['screenshots']:
            image_data.append((screenshot, game['tags'], game['title']))
    return image_data

# Step 2: Preprocess images
def preprocess_images(image_data, image_dir, target_size=(224, 224)):
    preprocessed_data = []
    for filename, tags, title in image_data:
        # Load image
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image '{filename}' not found in directory '{image_dir}'")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image '{filename}'")
            continue
        # Resize image
        img = np.array(cv2.resize(img, target_size))
        if img.shape == (224, 224, 3):
            preprocessed_data.append((img, tags, title))
    return preprocessed_data 

# Example usage
# image_dir = r"C:\Users\Theo\Documents\Advanced Analytics\images_downscaled"
# json_file = r"C:\Users\Theo\Documents\Advanced Analytics\datasetj.json"
json_file = r"C:\Users\WI137\Documents\datasetj.json"
image_dir = r"C:\Users\WI137\Documents\images_downscaled"
image_data = read_json(json_file)
preprocessed_data = preprocess_images(image_data, image_dir)
print(f"Preprocessed {len(preprocessed_data)} images")



def create_train_test_split(preprocessed_data, test_size=0.2, random_state=42):
    game_titles = [title for _, _, title in preprocessed_data]
    unique_game_titles = set(game_titles)

    train_game_titles, test_game_titles = train_test_split(list(unique_game_titles), test_size=test_size, random_state=random_state)
    train_data = [(img, tags, title) for img, tags, title in preprocessed_data if title in train_game_titles]
    test_data = [(img, tags, title) for img, tags, title in preprocessed_data if title in test_game_titles]
    
    return train_data, test_data



# Step 4: Create multilabel dataset with top 20 most common tags
def create_multilabel_dataset(preprocessed_data, top_n=20):
    # Count tag occurrences
    all_tags = [tag for _, tags, _ in preprocessed_data for tag in tags]
    tag_counter = Counter(all_tags)
    
    # Select top n most common tags
    top_tags = [tag for tag, _ in tag_counter.most_common(top_n)]
    
    # Create multilabel dataset
    X = np.array([img for img, _, _, in preprocessed_data])
    y = []
    for _, tags, _ in preprocessed_data:
        label = [1 if tag in tags else 0 for tag in top_tags]
        y.append(label)
    y = np.array(y)
    
    return X, y, top_tags

def create_train_test_multilabel_dataset(preprocessed_data, test_size=0.2, random_state=42, top_n=20):
    # Step 1: Splitting data into train and test sets
    game_titles = [title for _, _, title in preprocessed_data]
    unique_game_titles = set(game_titles)

    train_game_titles, test_game_titles = train_test_split(list(unique_game_titles), test_size=test_size, random_state=random_state)
    train_data = [(img, tags, title) for img, tags, title in preprocessed_data if title in train_game_titles]
    test_data = [(img, tags, title) for img, tags, title in preprocessed_data if title in test_game_titles]
    
    # Step 2: Creating multilabel dataset with top 20 most common tags
    # Count tag occurrences
    all_tags = [tag for _, tags, _ in preprocessed_data for tag in tags]
    tag_counter = Counter(all_tags)
    
    # Select top n most common tags
    top_tags = [tag for tag, _ in tag_counter.most_common(top_n)]
    
    # Create multilabel dataset for training data
    X_train = np.array([img for img, _, _ in train_data])
    y_train = []
    for _, tags, _ in train_data:
        label = [1 if tag in tags else 0 for tag in top_tags]
        y_train.append(label)
    y_train = np.array(y_train)
    
    # Create multilabel dataset for test data
    X_test = np.array([img for img, _, _ in test_data])
    y_test = []
    for _, tags, _ in test_data:
        label = [1 if tag in tags else 0 for tag in top_tags]
        y_test.append(label)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test, top_tags


# Example usage

# multilabel data first
# train test split after
def create_multilabel_dataset_with_train_test_split(preprocessed_data, test_size=0.2, random_state=42, top_n=20):
    # Use the provided function to create train-test split
    train_data, test_data = create_train_test_split(preprocessed_data, test_size=test_size, random_state=random_state)
    
    all_tags = [tag for _, tags, _ in train_data + test_data for tag in tags]
    tag_counter = Counter(all_tags)
    
    # Select top n most common tags
    top_tags = [tag for tag, _ in tag_counter.most_common(top_n)]
    print(top_tags)
    
    # Create multilabel dataset for train data
    X_train = np.array([img for img, _, _ in train_data])
    y_train = []
    for _, tags, _ in train_data:
        label = [1 if tag in tags else 0 for tag in top_tags]
        y_train.append(label)
    y_train = np.array(y_train)
    
    # Create multilabel dataset for test data
    X_test = np.array([img for img, _, _ in test_data])
    y_test = []
    for _, tags, _ in test_data:
        label = [1 if tag in tags else 0 for tag in top_tags]
        y_test.append(label)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test, top_tags







X_train, X_test, y_train, y_test, top_tags = create_train_test_multilabel_dataset(preprocessed_data)
# train_data, test_data = create_train_test_split(preprocessed_data)
# X_train, y_train, _ = create_multilabel_dataset(train_data)
# print(y_train)
# X_test, y_test, _ = create_multilabel_dataset(test_data)
print(f"Train dataset: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Test dataset: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(y_train[0:10])
print(y_test[0:10])


model = tf.keras.Sequential([
    keras.Input(shape=(224,224,3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(20, activation='sigmoid')  # Output layer with 20 neurons for multilabel classification
])

# Explanation of Model components:
# Conv2D produces tensor outputs using a convolution kernel, and these tensors are applied with a relu function
# MaxPooling2D downsamples the invput along 2 dimensions 
# Flatten turns the multidimensional tensors as input into single-dimensional output tensors



# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary crossentropy for multilabel classification
              metrics=['accuracy'])

print(model.summary())



# Train the model
history = model.fit(X_train,
                    y_train, 
                    epochs=3,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                )

# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("Model, accuracy: {:5.2f}%".format(100 * acc))

model.save('C:\\Users\\WI137\\Downloads\\multilabel_classification_model.keras')
 
predictions = model.predict(X_test)
binary_predictions = np.round(predictions)
np.savetxt('C:\\Users\\WI137\\Downloads\\binary_predictions.txt', binary_predictions, fmt='%d')
pd.DataFrame(binary_predictions).to_csv('C:\\Users\\WI137\\Downloads\\binary_predictions.csv', index=False, header = False)
print(model.predict(X_test))

print(y_test[0:10])
print(binary_predictions[0:10])

model.save('C:\\Users\\WI137\\Downloads\\multilabel_classification_model.keras')

