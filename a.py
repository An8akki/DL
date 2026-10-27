# SENTIMENT CLASSIFICATION USING AUTOENCODER

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, classification_report

# Load IMDB dataset
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Make all reviews same length
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

# Normalize
x_train = x_train.astype('float32') / vocab_size
x_test = x_test.astype('float32') / vocab_size

# -------- Autoencoder --------
input_layer = Input(shape=(max_len,))

# Encoder
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(32, activation='relu', name='latent_layer')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(latent)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(max_len, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# -------- Extract latent features --------
encoder = Model(input_layer, latent)

X_train_features = encoder.predict(x_train, verbose=0)
X_test_features = encoder.predict(x_test, verbose=0)

# -------- Sentiment Classifier --------
classifier_input = Input(shape=(32,))
x = Dense(16, activation='relu')(classifier_input)
output = Dense(1, activation='sigmoid')(x)

classifier = Model(classifier_input, output)

classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

classifier.fit(
    X_train_features, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# -------- Prediction --------
y_prob = classifier.predict(X_test_features, verbose=0)
y_pred = (y_prob > 0.5).astype(int).ravel()

# -------- Results --------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Negative', 'Positive']
))

# VARIATIONAL AUTOENCODER (VAE) ON MNIST -- PYTORCH VERSION
# ------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os # Import the os module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the output directory if it doesn't exist
output_dir = "/mnt/user-data/outputs/"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------
# 1. Data: MNIST handwritten digits (via torchvision)
# -------------------------------------------------
original_dim = 784
latent_dim = 2   # 2D so we can easily visualize/generate & plot the latent space
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),                       # scales pixels to [0, 1]
    transforms.Lambda(lambda x: x.view(-1)),      # flatten 28x28 -> 784
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------------------------
# 2. VAE model: encoder -> reparameterize -> decoder
# -------------------------------------------------
class VAE(nn.Module):
    def __init__(self, original_dim=784, latent_dim=2):
        super().__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(original_dim, 256)
        self.enc_fc2 = nn.Linear(256, 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 64)
        self.dec_fc2 = nn.Linear(64, 256)
        self.dec_out = nn.Linear(256, original_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, z_mean, z_logvar):
        # The "reparameterization trick": sample z = mean + std * epsilon
        # so gradients can flow through the sampling step.
        std = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        reconstruction = self.decode(z)
        return reconstruction, z_mean, z_logvar

# -------------------------------------------------
# 3. Loss = reconstruction loss + KL divergence
# -------------------------------------------------
def vae_loss(reconstruction, x, z_mean, z_logvar):
    # Reconstruction loss: binary cross-entropy summed over pixels
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")
    # KL divergence: keeps the latent distribution close to N(0, 1)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# -------------------------------------------------
# 4. Train
# -------------------------------------------------
model = VAE(original_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 15

for epoch in range(1, epochs + 1):
    model.train()
    total_loss, total_recon, total_kl, n_samples = 0.0, 0.0, 0.0, 0

    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()

        reconstruction, z_mean, z_logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(reconstruction, x, z_mean, z_logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_samples += x.size(0)

    print(
        f"Epoch {epoch:2d}/{epochs} | "
        f"loss: {total_loss / n_samples:.4f} | "
        f"recon: {total_recon / n_samples:.4f} | "
        f"kl: {total_kl / n_samples:.4f}"
    )

# -------------------------------------------------
# 5. Generate new digits by sampling a grid over latent space
# -------------------------------------------------
model.eval()
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)[::-1]

with torch.no_grad():
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            x_decoded = model.decode(z_sample).cpu().numpy()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="Greys_r")
plt.axis("off")
plt.title("Digits generated by sampling the VAE latent space")
plt.savefig(os.path.join(output_dir, "vae_pytorch_generated_digits.png"), dpi=150, bbox_inches="tight")
plt.close()

# -------------------------------------------------
# 6. Visualize how real digits cluster in latent space (colored by true label)
# -------------------------------------------------
all_z_mean = []
all_labels = []

with torch.no_grad():
    for x, y in train_loader:
        x = x.to(device)
        z_mean, _ = model.encode(x)
        all_z_mean.append(z_mean.cpu().numpy())
        all_labels.append(y.numpy())

all_z_mean = np.concatenate(all_z_mean, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(all_z_mean[:, 0], all_z_mean[:, 1], c=all_labels, cmap="tab10", s=2)
plt.colorbar(scatter, label="Digit label")
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.title("MNIST digits mapped into 2D latent space")
plt.savefig(os.path.join(output_dir, "vae_pytorch_latent_space.png"), dpi=150, bbox_inches="tight")
plt.close()
plt.show()
print("Done. Saved: vae_pytorch_generated_digits.png and vae_pytorch_latent_space.png")