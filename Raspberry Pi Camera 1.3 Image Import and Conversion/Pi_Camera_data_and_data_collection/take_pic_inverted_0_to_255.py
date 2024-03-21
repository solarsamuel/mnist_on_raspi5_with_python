import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
start_time = time.time()

# Define the command to take a picture
command = "libcamera-jpeg -o temp_pic.png --width 56 --height 56"  # Take a picture with dimensions 56x56

# Run the command using subprocess to capture the image
subprocess.run(command, shell=True)

# Load the captured image
image_path = "temp_pic.png"

# Load the image
image = Image.open(image_path)

# Resize the image to 56x56 pixels
image = image.resize((56, 56))

# Convert the image to grayscale
image = image.convert("L")

# Convert the image to a NumPy array
image_array = np.array(image)

# Perform max-pooling to reduce the image size to 28x28
downsampled_image_array = np.zeros((28, 28), dtype=np.float32)
for i in range(28):
    for j in range(28):
        downsampled_image_array[i, j] = np.max(image_array[i*2:i*2+2, j*2:j*2+2])

# Invert the downsampled image array
inverted_downsampled_image_array = 255 - downsampled_image_array

# Scale the values to be in the range from 0 to 255
scaled_downsampled_image_array = (inverted_downsampled_image_array / 255) * 255

# Set values less than 55 to 0
scaled_downsampled_image_array[scaled_downsampled_image_array < 75] = 0

# Display the shape of the downsampled array
print("Shape of the downsampled array:", scaled_downsampled_image_array.shape)

# Save the downsampled image array to CSV
np.savetxt('snapshot_normalized.csv', scaled_downsampled_image_array, fmt='%d', delimiter=',')

# Flatten the downsampled image array to 1D
flattened_array = scaled_downsampled_image_array.flatten()

# Save the flattened array to CSV
np.savetxt('flattened_snapshot_normalized.csv', flattened_array.reshape(1, -1), fmt='%d', delimiter=',')

# Plot the downsampled image
plt.imshow(scaled_downsampled_image_array, cmap='gray', vmin=0, vmax=255)
plt.title("Downsampled Image (28x28) - Inverted, Scaled, and Thresholded")
plt.axis('off')  # Hide axis ticks and labels
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
