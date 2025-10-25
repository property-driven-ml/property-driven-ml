import idx2numpy
import torch
import numpy as np
import matplotlib.pyplot as plt

from examples.datasets import create_dice_datasets

SIZE = (
    68  # test set currently has 68 images, probably okay to use all of them by default
)

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

_, test_loader, _, _, _ = create_dice_datasets(SIZE, normalise=False, seed=SEED)

all_images, all_labels = [], []

for imgs, labels in test_loader:
    imgs = imgs.cpu()  # imgs: [B, C, H, W]
    labels = labels.cpu()

    imgs_np = imgs.numpy()
    labels_np = labels.numpy()

    all_images.append(imgs_np)
    all_labels.append(labels_np)

all_images = np.concatenate(all_images, axis=0)  # [N, C, H, W]
all_labels = np.concatenate(all_labels, axis=0)

# choose arg.size random images
assert len(all_images) == len(all_labels)  # nosec
idx = np.random.choice(len(all_images), size=SIZE, replace=False)

print(f"indices={idx}")

# save images and labels
idx2numpy.convert_to_file(f"dice-images-{SIZE}.idx", all_images[idx])
idx2numpy.convert_to_file(f"dice-labels-{SIZE}.idx", all_labels[idx].astype(np.uint8))

# check if saving worked:
images = idx2numpy.convert_from_file(f"dice-images-{SIZE}.idx")
labels = idx2numpy.convert_from_file(f"dice-labels-{SIZE}.idx")

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

print(f"first image={images[0]}")

plt.imshow(np.transpose(images[0], (1, 2, 0)))
plt.title(f"Labels: {labels[0].tolist()}")
plt.show()
