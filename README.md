# Predicting labels for microscope images

## Cytoplasm, label 0

### The first model is a neural network with the following layers:

input layer: dimension is 100x100x3
first layer: convolution - 2 5x5 filters and a relu activation
second layer: max pool - 7x7 pool size 
third layer: flatten layer
fourth layer: Dense layer, with 10 or 100 neurons and relu activation
fifth layer: Dense output layer, with softmax activation

I searched over learning rate=[.1, 1, 10], momentum=[0, .9], and fifth layer neurons=[10, 100] with 10% of the training data.
The results were not very sensitive to fifth layer neuron count. All combinations of learning rate
and momentum lead to no improvements in training or testing loss, except for lr=.1 and momentum=0.
Figures 1 through 4 below show results.

<img src="/readmePics/model5100neurons.png" alt="Best (100 neurons)" width="350"/> <img src="/readmePics/model510neurons.png" alt="Best (10 neurons)" width="350"/>
 <img src="/readmePics/model5noImprov.png" alt="typical (10 neurons)" width="350"/> <img src="/readmePics/model5noImprov1.png" alt="typical (100 neurons)" width="350"/>

Conclusion: model5, lr=.1, m=0, and fifth layer neurons=10 might be able to pick up a general pattern if I train it on more of the data

Training this model with the above set of optimal training paramters does not give successful results. The training and validation losses during the training session are given below, as well as its precision and recall. 


### The next model to consider has the following layers

input layer: dimension is 512x512X3
The following layers are: convolution(f), pooling, convolution(2f), pooling, flatten, dense(N), dense(2) where f is the number of filters in the given layer and N is the number of neurons. A search was conducted over the hyperparameters learning rate=[.1, 1], momentum=[0, .5], N=[2, 10], and f=[4,10]. The best results were for learning rate=.1, momentum=0, N=4, and f=10. Below the training and validation loss as well as the frequency of predicted 1's and actual 1's (presence of cyctoplasm=1) are given for the best hyperparameter sets.

<img src="/readmePics/model6_best.png" alt=".." width="350"/> <img src="/readmePics/model6_best1.png" alt=".." width="350"/>

Below are what the typical results looked like for most of the search space.

<img src="/readmePics/model6_typical.png" alt=".." width="350"/>


 


