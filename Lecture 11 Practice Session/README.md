# Artificial Intelligence in Automotive Technology - Lecture 10 Practice Session

In the practice session of lecture 10 we will learn about recurrent neural networks. Because this topic is a little bit more complex then the neural networks we heared about before, we start with a few classical mathematical calculations. After that we will step deeper into a python code example that helps us to classify images with a recurrent neural network.
We will take the tensorflow python package which helps us to generate a recurrent neural network based on a LSTM model. After setting up the LSTM network we will train it with the help of an image dataset. Afterwards we use the LSTM for classifying images regarding the number that is displayed on this image.


## Things you need

1. For the mathematical calculation exercise we will need the [Mathematical Exercise](https://github.com/TUMFTM/Lecture_AI_in_Automotive_Technology/blob/master/Lecture%2010%20Practice%20Session/Mathematical_Exercise.pdf). The solutions of the exercises can be found in the lecture video.

2. Our recurrent neural network should take some image data from the MNIST Dataset as an input. As an output the network gives back the recognized number on an image.. For training the network we use the MNIST dataset given with this lecture. The code for training and evaluting the recurrent neural network is set up in a jupyter-notebook an can be started with the following command:

```
jupyter RNN.ipynb
```
