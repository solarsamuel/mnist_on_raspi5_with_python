#save W1 and b1 to CSV output files after model has been trained
#enter the code below into the thonny python shell after training and press enter

import numpy as np 

# Output W1
with open('W1_output84.csv', 'w') as f:
    for i in range(10):
        f.write(f'Weights for neuron {i}\n')
        np.savetxt(f, W1[i].reshape(28, 28), delimiter=',', fmt='%f')

# Output b1
with open('b1_output84.csv', 'w') as f:
    f.write('Biases\n')
    np.savetxt(f, b1, delimiter=',', fmt='%f')

# Output W2
with open('W2_output84.csv', 'w') as f:
    for i in range(10):
        f.write(f'Weights for neuron {i}\n')
        np.savetxt(f, W2[i], delimiter=',', fmt='%f')

# Output b2
with open('b2_output84.csv', 'w') as f:
    f.write('Biases\n')
    np.savetxt(f, b2, delimiter=',', fmt='%f')