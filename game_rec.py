import os
import json
import cv2
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf





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
image_dir = r"Path\images_downscaled"
json_file = r"Path\datasetj.json"
image_data = read_json(json_file)
preprocessed_data = preprocess_images(image_data, image_dir)
print(f"Preprocessed {len(preprocessed_data)} images")

# Step 3: Create train-test split based on game titles
# def create_train_test_split(preprocessed_data, test_size=0.2, random_state=42):
#     # Get unique game titles
#     game_titles = set([game['title'] for game in preprocessed_data])
    
#     # Split game titles into train and test sets
#     train_titles, test_titles = train_test_split(list(game_titles), test_size=test_size, random_state=random_state)
    
#     # Split preprocessed data into train and test sets
#     train_data = [(img, tags) for img, tags in preprocessed_data if any(title in tags for title in train_titles)]
#     test_data = [(img, tags) for img, tags in preprocessed_data if any(title in tags for title in test_titles)]
    
#     return train_data, test_data

# def create_train_test_split(preprocessed_data, test_size=0.2, random_state=42):
#     # Extracting image tags and game titles
    
#     game_titles = [title for _, title, _ in preprocessed_data]
    
#     # Get unique game titles
#     unique_game_titles = set(game_titles)
    
#     # Splitting the game titles into train and test sets
#     train_game_titles, test_game_titles = train_test_split(list(unique_game_titles), test_size=test_size, random_state=random_state)
    
#     # Splitting the preprocessed data into train and test sets based on game titles
#     train_data = [(img, title, tags) for img, title, tags in preprocessed_data if title in train_game_titles]
#     test_data = [(img, title, tags) for img, title, tags in preprocessed_data if title in test_game_titles]
    
#     return train_data, test_data

def create_train_test_split(preprocessed_data, test_size=0.2, random_state=42):
    game_titles = [title for _, _, title in preprocessed_data]
    unique_game_titles = set(game_titles)

    train_game_titles, test_game_titles = train_test_split(list(unique_game_titles), test_size=test_size, random_state=random_state)
    train_data = [(img, tags, title) for img, tags, title in preprocessed_data if title in train_game_titles]
    test_data = [(img, tags, title) for img, tags, title in preprocessed_data if title in test_game_titles]
    
    return train_data, test_data


# Example usage
train_data, test_data = create_train_test_split(preprocessed_data)

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

# Example usage
train_data, test_data = create_train_test_split(preprocessed_data)
X_train, y_train, _ = create_multilabel_dataset(train_data)
X_test, y_test, _ = create_multilabel_dataset(test_data)
print(f"Train dataset: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Test dataset: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
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

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary crossentropy for multilabel classification
              metrics=['accuracy'])

print(model.summary())



# Train the model
history = model.fit(X_train,
                    y_train, 
                    epochs=5,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                )

# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("Model, accuracy: {:5.2f}%".format(100 * acc))

model.save('C:\\Path\\multilabel_classification_model.keras')

