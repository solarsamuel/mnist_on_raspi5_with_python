import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = "test11.png"  # Change this to your image path
image = Image.open(image_path)

# Resize the image to 56x56 pixels
image = image.resize((56, 56))

# Convert the image to grayscale
image = image.convert("L")

# Convert the image to a NumPy array
image_array = np.array(image)

# Perform max-pooling to reduce the image size to 28x28
downsampled_image_array = np.zeros((28, 28), dtype=np.uint8)
for i in range(28):
    for j in range(28):
        downsampled_image_array[i, j] = np.max(image_array[i*2:i*2+2, j*2:j*2+2])

# Display the shape of the downsampled array
print("Shape of the downsampled array:", downsampled_image_array.shape)

# Plot the downsampled image
plt.imshow(downsampled_image_array, cmap='gray')
plt.title("Downsampled Image (28x28)")
plt.axis('off')  # Hide axis ticks and labels
plt.show()
