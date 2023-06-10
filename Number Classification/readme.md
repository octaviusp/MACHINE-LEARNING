# Number Classification
This is a neural network that classify image numbers to a label [0...9].

# Neural network architecture
It uses convolutional layers to recognize more complex patterns in the digits, as different width, angles, etc...
Some MaxPooling2D Layers to grab 1 pixel from images, i decided to put 1x1 because i thought that it could be more precise grabbing
all pixels and not in 2x2 whereas only 1 pixel is selected, because the digit resolution image is 28x28, therefore,
if we grab 1 pixel from 2x2, it will reduce a lot of information and patterns, and i think that shouldn't great for the learning.
Finally, two dense layers, one for last decisions and the last one for softmax (Because we have more than 2 classes, isn't binary class).
Of course, we could set the final layer as linear activation or "no activation", and therefore calculate the probabilities out of the net.
