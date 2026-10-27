import os
import zipfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Embedding, LSTM, Add
)

# ==========================================================
# METHOD 1: BUILT-IN FASHION-MNIST DATASET
# Images are loaded into NumPy arrays, not an image folder.
# This is a demonstration dataset and has labels, not captions.
# ==========================================================

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Use a smaller subset to keep the demonstration manageable.
x_train = x_train[:1000].astype("float32") / 255.0
y_train = y_train[:1000]

x_test = x_test[:100].astype("float32") / 255.0
y_test = y_test[:100]

class_names = [
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot"
]

# Add the grayscale channel dimension expected by the CNN.
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Generate simple demonstration captions from the class labels.
# These are synthetic captions, not original human-written captions.

captions = []

for label in y_train:
    name = class_names[label]
    article = "an" if name[0] in "aeiou" else "a"

    templates = [
        "this is " + article + " " + name,
        "look at " + article + " " + name
    ]

    captions.append("startseq " + random.choice(templates) + " endseq")


# ==========================================================
# METHOD 2: CUSTOM IMAGE FOLDER + CSV
# If using your own dataset, comment out METHOD 1 above.
# Uncomment this section and update the paths.
#
# DATASET_PATH = r"C:\path\to\dataset"
# IMAGE_FOLDER = os.path.join(DATASET_PATH, "images")
# CAPTION_FILE = os.path.join(DATASET_PATH, "captions.csv")
#
# captions_df = pd.read_csv(CAPTION_FILE)
#
# images = []
# captions = []
#
# for _, row in captions_df.iterrows():
#     image_path = os.path.join(IMAGE_FOLDER, row["image_name"])
#     if os.path.exists(image_path):
#         img = load_img(image_path, target_size=(128, 128))
#         images.append(img_to_array(img) / 255.0)
#         captions.append("startseq " + str(row["caption"]) + " endseq")
#
# x_train = np.array(images)
# y_train = None
#
# # For a proper experiment, split the images and captions into
# # training and testing sets together, keeping their order aligned.
# # Set x_test and y_test to the held-out images and labels/captions.
#
#
# ==========================================================
# METHOD 3: CUSTOM ZIP FILE
# Extract the ZIP first, then use the folder-loading approach.
#
# ZIP_PATH = r"C:\path\to\dataset.zip"
# EXTRACT_PATH = r"C:\path\to\extracted_dataset"
#
# with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
#     zip_ref.extractall(EXTRACT_PATH)
#
# # Then set DATASET_PATH = EXTRACT_PATH in METHOD 2.
# ==========================================================

print("Training image shape:", x_train.shape)
print("Number of captions:", len(captions))
# The model processes word IDs instead of raw text.
# The Tokenizer builds a vocabulary from the training captions.

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(captions)

max_len = max(len(seq) for seq in sequences)

print("Vocabulary size:", vocab_size)
print("Maximum caption length:", max_len)

# Create next-word prediction examples.
# Example:
# Input: "startseq this"
# Target: "is"

X_img = []
X_seq = []
Y = []

for i, seq in enumerate(sequences):
    for j in range(1, len(seq)):
        X_img.append(x_train[i])
        X_seq.append(seq[:j])
        Y.append(seq[j])

# Pad all partial captions to the same length.
X_seq = pad_sequences(X_seq, maxlen=max_len, padding="pre")

X_img = np.array(X_img, dtype="float32")
X_seq = np.array(X_seq)
Y = np.array(Y)

print("Image inputs:", X_img.shape)
print("Caption inputs:", X_seq.shape)
print("Next-word targets:", Y.shape)
# CNN branch: extracts features from the image.
image_input = Input(shape=x_train.shape[1:], name="image_input")

x = Conv2D(32, (3, 3), activation="relu", padding="same")(image_input)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2))(x)

x = GlobalAveragePooling2D()(x)
image_features = Dense(64, activation="relu")(x)


# LSTM branch: processes the caption words generated so far.
caption_input = Input(shape=(max_len,), name="caption_input")

x = Embedding(
    input_dim=vocab_size,
    output_dim=64,
    mask_zero=True
)(caption_input)

text_features = LSTM(64)(x)


# Combine image and text features.
merged = Add()([image_features, text_features])
merged = Dense(64, activation="relu")(merged)

# Predict the next word in the caption.
output = Dense(vocab_size, activation="softmax")(merged)

model = Model(
    inputs=[image_input, caption_input],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# The model learns to predict the next word from the image
# and the words already present in the partial caption.

history = model.fit(
    [X_img, X_seq],
    Y,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)
# Visualize training progress.
# This displays the graph without saving it as a PNG file.

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("CNN + LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()
# Create a mapping from numerical word IDs back to words.
index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}

# Generate captions for six test images.
for i in range(min(6, len(x_test))):

    img = x_test[i]
    generated_caption = "startseq"

    for j in range(max_len):

        # Convert the current caption into word IDs.
        seq = tokenizer.texts_to_sequences([generated_caption])[0]
        seq = pad_sequences([seq], maxlen=max_len, padding="pre")

        # Predict the next word.
        prediction = model.predict(
            [img[np.newaxis, ...], seq],
            verbose=0
        )

        next_word_id = np.argmax(prediction)

        # Stop if padding or an unknown token is predicted.
        if next_word_id == 0:
            break

        next_word = index_to_word.get(next_word_id, "")

        # Stop when the caption reaches its end token.
        if next_word == "endseq" or next_word == "":
            break

        generated_caption += " " + next_word

    display_caption = generated_caption.replace("startseq", "").strip()

    # Show the image and generated caption.
    plt.figure(figsize=(4, 4))
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(
        "Actual: " + class_names[y_test[i]] +
        "\nGenerated: " + display_caption
    )
    plt.axis("off")
    plt.show()
