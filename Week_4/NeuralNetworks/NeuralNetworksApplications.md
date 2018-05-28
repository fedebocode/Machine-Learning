# Examples and Intuitions I

A simple example of applying neural networks is by predicting _x1_ AND _x2_, which is the logical 'and' operator and is only true if both _x1_ and _x2_ are 1.

The graph of our functions will look like:

![alt text](/Week_4/NeuralNetworks/Assets/9.png)

Remember that _x0_ is our bias variable and is always 1.

Let's set our first theta matrix as:

![alt text](/Week_4/NeuralNetworks/Assets/10.png)

This will cause the output of our hypothesis to only be positive if both _x1_ and _x2_ are 1. In other words:

![alt text](/Week_4/NeuralNetworks/Assets/11.png)

So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can also be used to simulate all the other logical gates. The following is an example of the logical operator 'OR', meaning either _x1_ is true or _x2_ is true, or both:

![alt text](/Week_4/NeuralNetworks/Assets/12.png)

Where _g(z)_ is the following:

![alt text](/Week_4/NeuralNetworks/Assets/13.png)

# Examples and Intuitions II

The _Θ(1)_ matrices for AND, NOR, and OR are:

![alt text](/Week_4/NeuralNetworks/Assets/14.png)

We can combine these to get the XNOR logical operator (which gives 1 if _x1_ and _x2_ are both 0 or both 1).

![alt text](/Week_4/NeuralNetworks/Assets/15.png)

For the transition between the first and second layer, we'll use a _Θ(1)_ matrix that combines the values for AND and NOR:

![alt text](/Week_4/NeuralNetworks/Assets/16.png)

For the transition between the second and third layer, we'll use a _Θ(2)_ matrix that uses the value for OR:

![alt text](/Week_4/NeuralNetworks/Assets/17.png)

Let's write out the values for all our nodes:

![alt text](/Week_4/NeuralNetworks/Assets/18.png)

And there we have the XNOR operator using a hidden layer with two nodes! The following summarizes the above algorithm:

![alt text](/Week_4/NeuralNetworks/Assets/19.png)

# Multi Class Classification

To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four categories. We will use the following example to see how this classification is done. This algorithm takes as input an image and classifies it accordingly:

![alt text](/Week_4/NeuralNetworks/Assets/20.png)

We can define our set of resulting classes as y:

![alt text](/Week_4/NeuralNetworks/Assets/21.png)

Each _y(i)_ represents a different image corresponding to either a car, pedestrian, truck, or motorcycle. The inner layers, each provide us with some new information which leads to our final hypothesis function. The setup looks like:

![alt text](/Week_4/NeuralNetworks/Assets/21.png)

Our resulting hypothesis for one set of inputs may look like:

![alt text](/Week_4/NeuralNetworks/Assets/23.png)

In which case our resulting class is the third one down, or _hΘ(x)3_, which represents the motorcycle.
