# mnist_on_raspi5_with_python
We train and run inference with python on a Raspberry Pi 5 - single hidden layer of 10 neurons.

This is run on a Raspberry Pi 5 with 4GB RAM running the Debian 12 Bookworm OS with Python version 3.11.2. 

Here is the original reference video and inspiration by Samson Zhang (Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math): https://www.youtube.com/watch?app=desktop&v=w8yWXqWQYmU

The neural network architecture is show around the 3 minute mark and has a 784 neuron input layer, a 10 neuron hidden layer, and a 10 neuron output layer. The code is here: https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook

We wanted to see if this would run locally on a RasPi5 instead of a cloud-based notebook like kaggle, and if interference (forward propogation) would be similar, which it was. 

Here are steps to get this working:
1. Get training set in CSV format: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
2. Unzip to a place you'll remember like desktop. There will be 2 files: mnist_train.csv and mnist_test.csv
3. Install python libraries: matplotlib and pandas. If you receive an "error: externally-managed-environment" then follow this link and use this 1 line of code in the terminal:
https://www.jeffgeerling.com/blog/2023/how-solve-error-externally-managed-environment-when-installing-pip3

sudo rm -rf /usr/lib/python3.11/EXTERNALLY-MANAGED

Then in the terminal type:

pip install matplotlib

pip install pandas

5. Setup the MNIST architecture and train the model using the mnist_train.csv file. Copy the code from the file mnist_setup_and_train.py and run with the green triangle in the Thonny editor.

You can adjust the training below. Iterations run roughly at 10 iterations in 10 seconds or 1 iteration per second. 500 iterations yields roughly 85% accuracy. 1000 iterations --> 88% accuracy. 10,000 iterations --> 93% accuracy. 40,000 iterations --> 95% accuracy. 

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

5. Once the neural network has been trained (We recommend starting with 500 iterations, around 10 minutes to run) then you can do a number of things in the Thonny Python shell below the setup and training code:
   
  A. Test accuracy - Copy the code from the file: accuracy_testing_after_training.py to the Thonny Shell and press enter. Results should be between 85% and 95% depending on iterations.
  
  B. Plot weights and biases - Copy the code from the file: plot_weights_and_biases_after_training.py to the Thonny Shell and press enter.
  
  C. Save the weights and biases - Copy the code from the file: save_weights_and_biases_csv_after_training.py to the Thonny Shell and press enter.


   
