#after training, start identifying: copy these lines to the shell down below once it is done training to test 1 at at time

#test_prediction(0, W1, b1, W2, b2)
#test_prediction(1, W1, b1, W2, b2)
#test_prediction(2, W1, b1, W2, b2)
#test_prediction(3, W1, b1, W2, b2)

#enter 2 lines below after training into the shell to get the accuracy

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)

