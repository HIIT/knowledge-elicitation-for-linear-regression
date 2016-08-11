# Prior Elicitation for linear models
This is the simulation scenario for prior elicitation when the user can give feedback to different features of the model with the goal to improve the prediction on the test data. 

The starting point of the code is main or main_all scripts.

main:
    runs the simulation for fixed number of training data and dimensions.
main_all: 
    the number of training data or dimensions can be arrays, e.g, 2:10:300.
    
Depending on the MODE parameter, the type of model and feedback changes:
           MODE = 
           0: Feedback on weight values. Model: Gaussian prior 
           1: Feedback on weight values. Model: spike and slab prior
           2: Feedback on relevance of features. Model: spike and slab prior

