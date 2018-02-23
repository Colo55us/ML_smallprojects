/////////////////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

To learn how a convolutional neural  network works I made a small CNN which will classify 
the input images of size 28 * 28 pixels into 10 classes using a fashion-mnist dataset which is present in keras datasets.

I used keras on top of tensowrflow. This Model consist of 5 layers from which 2 are dense for output and 3 are Convolutional 2-Dimensional layers along with Max pooling(as its a CNN).Also flattening is done to reduce the dimensionality of the tensor inputs.

The model is trained with training as well as validation data for the final tune adjustment of the weights and biases.  

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\///////////////////////////////////////////////////