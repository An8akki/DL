"""
Experiment 4: Image Caption Generator using CNN and LSTM
==========================================================

Dataset: Fashion-MNIST (tf.keras.datasets.fashion_mnist) — a dataset
that ships DIRECTLY with Keras, so nothing needs to be downloaded from
an external source and the lab is guaranteed to work offline-first / on
any machine, and it trains fast (28x28 grayscale images).

Why Fashion-MNIST for a *captioning* task?
-------------------------------------------
Real captioning datasets (Flickr8k, MS-COCO) need multi-GB downloads and
several hours of training, which does not fit a lab session. Fashion-MNIST
already gives us labeled clothing images (10 classes), so we auto-generate
short, varied natural-language captions from the class labels using a
handful of templates (e.g. "a photo of a sneaker", "this picture shows a
dress"). This keeps the full CNN encoder + LSTM decoder architecture, the
tokenizer/vocabulary building, the training loop, and greedy caption
inference all intact and end-to-end trainable in a couple of minutes on
CPU — while still being a genuine "CNN extracts image features -> LSTM
generates a sentence word by word" image-captioning pipeline.

(If you want to swap in a real photo-captioning dataset later, e.g.
Flickr8k, only the "1. Build (image, caption) pairs" section below needs
to change — the model architecture and training/inference code is the
same.)

Framework: TensorFlow / Keras
    pip install tensorflow numpy matplotlib
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense,
    Embedding, LSTM, Add, Dropout
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------------------
# 1. Build (image, caption) pairs from Fashion-MNIST
# -------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize to [0,1] and add a channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

# Simplified class names (avoiding slashes/punctuation, which is friendlier
# for a word-level tokenizer than the official "T-shirt/top" label).
class_names = ["t-shirt", "trouser", "pullover", "dress", "coat",
               "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# A few caption templates per class -> gives linguistic variety instead
# of one fixed sentence per class, which is what makes this a genuine
# sequence-generation problem rather than pure classification.
TEMPLATES = [
    "a photo of a {}",
    "this picture shows a {}",
    "an image containing a {}",
    "a small picture of a {}",
    "here is a {}",
]


def make_caption(label_idx):
    name = class_names[label_idx]
    article_fix = "an " + name if name[0] in "aeiou" else "a " + name
    template = random.choice(TEMPLATES)
    return "startseq " + template.format(name).replace("a " + name, article_fix) + " endseq"


# To keep training fast for a lab session, use a subset of Fashion-MNIST.
N_TRAIN, N_TEST = 4000, 500
train_idx = np.random.choice(len(x_train), N_TRAIN, replace=False)
test_idx = np.random.choice(len(x_test), N_TEST, replace=False)

train_images = x_train[train_idx]
train_captions = [make_caption(int(y_train[i])) for i in train_idx]
test_images = x_test[test_idx]
test_captions = [make_caption(int(y_test[i])) for i in test_idx]

print("Example captions:", train_captions[:3])

# -------------------------------------------------------------------
# 2. Tokenize captions and build the vocabulary
# -------------------------------------------------------------------
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(train_captions)
vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(c.split()) for c in train_captions)
print(f"Vocab size: {vocab_size}, Max caption length: {max_len}")


def encode_captions(captions):
    return tokenizer.texts_to_sequences(captions)


train_seqs = encode_captions(train_captions)
test_seqs = encode_captions(test_captions)

# -------------------------------------------------------------------
# 3. Expand into (image, partial_caption) -> next_word training pairs
#    e.g. for "startseq a dress endseq" we create:
#       (img, [startseq])           -> a
#       (img, [startseq, a])        -> dress
#       (img, [startseq, a, dress]) -> endseq
# -------------------------------------------------------------------


def build_sequences(images, seqs):
    X_img, X_seq, y = [], [], []
    for img, seq in zip(images, seqs):
        for t in range(1, len(seq)):
            in_seq, out_word = seq[:t], seq[t]
            in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
            X_img.append(img)
            X_seq.append(in_seq)
            y.append(out_word)
    return np.array(X_img), np.array(X_seq), to_categorical(y, num_classes=vocab_size)


X_img_train, X_seq_train, y_train_cap = build_sequences(train_images, train_seqs)
print("Training pairs:", X_img_train.shape[0])

# -------------------------------------------------------------------
# 4. Model: CNN encoder + LSTM decoder (classic "merge" architecture)
# -------------------------------------------------------------------
EMBED_DIM = 128

# ---- CNN branch: extracts a fixed-length feature vector from the image
# Fashion-MNIST images are small (28x28, grayscale), so a compact CNN is
# enough to get good features.
image_input = Input(shape=(28, 28, 1), name="image_input")
cnn = Conv2D(32, (3, 3), activation="relu", padding="same")(image_input)
cnn = MaxPooling2D()(cnn)
cnn = Conv2D(64, (3, 3), activation="relu", padding="same")(cnn)
cnn = MaxPooling2D()(cnn)
cnn = GlobalAveragePooling2D()(cnn)
image_features = Dense(EMBED_DIM, activation="relu")(cnn)
image_features = Dropout(0.3)(image_features)

# ---- LSTM branch: encodes the partial caption seen so far
seq_input = Input(shape=(max_len,), name="caption_input")
seq_embed = Embedding(vocab_size, EMBED_DIM, mask_zero=True)(seq_input)
seq_embed = Dropout(0.3)(seq_embed)
seq_features = LSTM(EMBED_DIM)(seq_embed)

# ---- Merge both branches and predict the next word
merged = Add()([image_features, seq_features])
merged = Dense(EMBED_DIM, activation="relu")(merged)
output = Dense(vocab_size, activation="softmax")(merged)

caption_model = Model(inputs=[image_input, seq_input], outputs=output)
caption_model.compile(optimizer="adam", loss="categorical_crossentropy",
                       metrics=["accuracy"])
caption_model.summary()

# -------------------------------------------------------------------
# 5. Train
# -------------------------------------------------------------------
history = caption_model.fit(
    [X_img_train, X_seq_train], y_train_cap,
    epochs=10, batch_size=64, validation_split=0.1, verbose=1
)

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Caption model training loss")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross-entropy")
plt.legend()
plt.tight_layout()
plt.savefig("caption_training_loss.png")
plt.close()

# -------------------------------------------------------------------
# 6. Inference: generate a caption word-by-word (greedy decoding)
# -------------------------------------------------------------------
index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}


def generate_caption(image, max_length=max_len):
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        pred = caption_model.predict([np.expand_dims(image, 0), seq], verbose=0)
        next_idx = int(np.argmax(pred))
        next_word = index_to_word.get(next_idx, "")
        if next_word == "" or next_word == "endseq":
            break
        in_text += " " + next_word
    return in_text.replace("startseq", "").strip()


# -------------------------------------------------------------------
# 7. Show a few sample predictions on unseen test images
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
sample_indices = np.random.choice(len(test_images), 6, replace=False)
for ax, idx in zip(axes.ravel(), sample_indices):
    img = test_images[idx]
    true_label = class_names[int(y_test[test_idx[idx]])]
    pred_caption = generate_caption(img)
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"True: {true_label}\nGenerated: {pred_caption}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("caption_predictions.png")
plt.close()

print("\nSaved plots: caption_training_loss.png, caption_predictions.png")
print("Example generated caption:", generate_caption(test_images[0]))