# Binary Classification

- This project was created with Tensorflow.

# NEURAL NETWORK ARCHITECTURE
![neural network](https://github.com/octaviusp/MACHINE-LEARNING/blob/main/Cat%20or%20Dog/NEURAL%20NETWORK%20ARCHITECTURE.png)

The idea behind this was to set a sequential model and insert two conv layer, also reduction layer aka "MaxPooling2D" with a
2x2 region. Also a flatten layer to reduce 2d -> 1d.
Finally dense layer with 64 units to perform last decisions and feed into the last layer with sigmoid activation.
Another approach could be set final layer as linear activation "no activation", and finally perform the prediction operation treshhold
out of the neural net, this approach is more computationally perfomant.


