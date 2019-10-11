# Artificial Intelligence in Automotive Technology - Lecture 11 Practice Session

In the practice session of lecture 11 we will learn about Reeinforcement Learning. Because this topic is a little bit more complex then the neural networks we heared about before, we start with a few classical mathematical calculations. After that we will step deeper into a python code example that helps us to get in touch with Q-Learning algorithm in the field of Reeinforcement Learning.



## Things you need

1. For the mathematical calculation exercise we will need the [Mathematical Exercise](https://github.com/TUMFTM/Lecture_AI_in_Automotive_Technology/blob/master/Lecture%2011%20Practice%20Session/Exercise11.pdf). The solutions of the exercises can be found in the lecture video.

2. Our Q-Learning algorithm is applied to the Grid world. While the agent explores the conditions, he always has the choice between two general learning strategies. He must decide whether to visit poorly explored paths through the state room to find out more about them, or whether to adhere to his current strategy to verify their accuracy and perhaps less to explore explored states, which he only achieves through decisions that have already been found to be good.

The code for the Q-Learning is set up in a jupyter-notebook an can be started with the following command:

```
jupyter notebook Qlearn_GridWorld.ipynb
```
