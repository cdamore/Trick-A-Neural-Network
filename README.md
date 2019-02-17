# Trick-A-Neural-Network
Trick a trained Lenet on the MNIST dataset to confidently classifiy a handwritten image of a 7 as a 0. This is done by inputing the image into the neural network, computing the loss as the distance to the class we want the network to classify (0 in this case), then training/tuning the image (instead of the neural net) to be recognized as a 0 instead of a 7 without drastically changing the image.
#
This program runs in python3 and uses tensorflow. The CNN follows the lenet architecture proposed by Yann LeCun, and the saved tensorflow model can be found in the LeNet/ckpt directory. 
