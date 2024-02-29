#put the code below into the thonny python shell after the model has been trained then press enter to plot the weights and biases

# Visualize W1
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(W1[i].reshape(28, 28), cmap='gray')
    plt.title(f'Weight {i}')
    plt.axis('off')
plt.suptitle('Visualization of W1')
plt.show()

# Visualize b1
plt.figure(figsize=(10, 5))
plt.plot(np.arange(10), b1)
plt.title('Visualization of b1')
plt.xlabel('Neuron Index')
plt.ylabel('Bias Value')
plt.xticks(np.arange(10))
plt.grid(True)
plt.show()

# Visualize W2
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(W2[i].reshape(10, 1), cmap='gray')  # Adjust reshape according to the shape of W2
    plt.title(f'Weight {i}')
    plt.axis('off')
plt.suptitle('Visualization of W2')
plt.show()

# Visualize b2
plt.figure(figsize=(10, 5))
plt.plot(np.arange(10), b2)
plt.title('Visualization of b2')
plt.xlabel('Neuron Index')
plt.ylabel('Bias Value')
plt.xticks(np.arange(10))
plt.grid(True)
plt.show()