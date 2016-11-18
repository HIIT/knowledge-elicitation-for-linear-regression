# Prior Elicitation for linear models
This is the simulation and real data code for prior elicitation when the user can give feedback to different features of the model with the goal to improve the prediction on the test data. The model uses a spike-and-slab prior and therefore it is specially usefull for the problems with "small n, large p", i.e. when the number of training data are less than the number of dimensions. 

The posterior inference can be found in "linreg_sns_ep.m". 

The starting point of the code are one of the  main scripts.

main:
    runs the simulation for fixed number of training data and dimensions.
main_all: 
    the number of training data or dimensions can be arrays, e.g, 2:10:300.
main_real_data.m:
    This script is useful for real data. Your data should contain a matrix of X with size (n*p) and a matrix of Y with size (n*1). In the code there is an example of using the method for Amazon and Yelp dataset. 
    
Depending on the MODE parameter, the type of model and feedback changes:
           MODE = 
           
           0: Feedback on weight values. Model: Gaussian prior        
           1: Feedback on weight values. Model: spike and slab prior         
           2: Feedback on relevance of features. Model: spike and slab prior

Mode 2 is the most realistic case for the real data.           

