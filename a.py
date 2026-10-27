import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, classification_report

vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

x_train = pad_sequences(x_train, maxlen=max_len).astype("float32")
x_test = pad_sequences(x_test, maxlen=max_len).astype("float32")

# Normalize
x_train = x_train / vocab_size
x_test = x_test / vocab_size

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))
print("Input shape:", x_train.shape)
# Encoder
inp = Input(shape=(200,))
x = Dense(64, activation="relu")(inp)
latent = Dense(32, activation="relu")(x)

# Decoder
x = Dense(64, activation="relu")(latent)
out = Dense(200, activation="sigmoid")(x)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)
# Extract 32 compressed features
encoder = Model(inp, latent)

X_train = encoder.predict(x_train, verbose=0)
X_test = encoder.predict(x_test, verbose=0)

# Build classifier
classifier_input = Input(shape=(32,))
x = Dense(16, activation="relu")(classifier_input)
output = Dense(1, activation="sigmoid")(x)

classifier = Model(classifier_input, output)

classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

classifier.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)
probabilities = classifier.predict(X_test, verbose=0)
predictions = (probabilities > 0.5).astype(int).ravel()

print("Accuracy:", accuracy_score(y_test, predictions))

print(classification_report(
    y_test,
    predictions,
    target_names=["Negative", "Positive"]
))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("IMDB Autoencoder Learning Curve")
plt.legend()
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)

test_loader = DataLoader(
    test_data, batch_size=128, shuffle=False
)

print("Device:", device)
# Encoder
encoder = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU()
).to(device)

mean_layer = nn.Linear(128, 2).to(device)
logvar_layer = nn.Linear(128, 2).to(device)

# Decoder
decoder = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Sigmoid()
).to(device)

# Combine parameters for training
parameters = (
    list(encoder.parameters())
    + list(mean_layer.parameters())
    + list(logvar_layer.parameters())
    + list(decoder.parameters())
)

optimizer = torch.optim.Adam(parameters, lr=0.001)
epochs = 10
loss_curve = []

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        x = images.view(-1, 784).to(device)

        optimizer.zero_grad()

        # Encoder
        h = encoder(x)
        mean = mean_layer(h)
        logvar = logvar_layer(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)

        # Decoder
        output = decoder(z)

        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(
            output, x, reduction="sum"
        )

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mean.pow(2) - logvar.exp()
        )

        # Total loss
        loss = recon_loss + kl_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    loss_curve.append(avg_loss)

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {avg_loss:.4f}"
    )
plt.plot(loss_curve, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Curve")
plt.show()
encoder.eval()
decoder.eval()
mean_layer.eval()
logvar_layer.eval()

with torch.no_grad():
    z = torch.randn(10, 2).to(device)
    generated = decoder(z).cpu().numpy()

plt.figure(figsize=(10, 2))

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(
        generated[i].reshape(28, 28),
        cmap="gray"
    )
    plt.axis("off")

plt.suptitle("Digits Generated by VAE")
plt.show()
all_z = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        x = images.view(-1, 784).to(device)

        h = encoder(x)
        mean = mean_layer(h)

        all_z.append(mean.cpu().numpy())
        all_labels.append(labels.numpy())

all_z = np.concatenate(all_z)
all_labels = np.concatenate(all_labels)

plt.figure(figsize=(7, 5))

plt.scatter(
    all_z[:, 0],
    all_z[:, 1],
    c=all_labels,
    cmap="tab10",
    s=3
)

plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("VAE Latent Space")
plt.colorbar(label="Digit Label")
plt.show()
