import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

start_time = time.time()

# Define the command to take a picture
command = "libcamera-jpeg -o cam_pic.png --width 56 --height 56"  # Take a picture with dimensions 56x56

# Run the command using subprocess to capture the image
subprocess.run(command, shell=True)

# Load the captured image
image_path = "cam_pic.png"

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
        downsampled_image_array[i, j] = (np.max(image_array[i*2:i*2+2, j*2:j*2+2]) / 255) - 0.5

# Invert the black and white coloring
downsampled_image_array = 1.0 - downsampled_image_array

# Flatten and transpose the downsampled image array to one row and 784 columns
flattened_array = downsampled_image_array.flatten()
flattened_array = flattened_array.reshape((1, -1))

# Display the shape of the flattened array
print("Shape of the flattened array:", flattened_array.shape)

# Save the flattened and transposed array to CSV
np.savetxt('1x784_pic.csv', flattened_array, fmt='%f', delimiter=',')

# Plot the downsampled image
plt.imshow(downsampled_image_array, cmap='gray')
plt.title("Downsampled Image (28x28) - Inverted Grayscale")
plt.axis('off')  # Hide axis ticks and labels
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
