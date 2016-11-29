# Prior Elicitation for linear models
This is the simulation and real data code for prior elicitation when the user can give feedback to different features of the model with the goal to improve the prediction on the test data. The model uses a spike-and-slab prior and therefore it is specially usefull for the problems with "small n, large p", i.e. when the number of training data are less than the number of dimensions. 


<p align="center">
  <img src="/Plate diagram.JPG" width="500"/>
</p>

## Basic Usage

```matlab

[fa, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, w_feedbacks, gamma_feedbacks, si)

%Inputs: 
%pr:               prior and other fixed model parameters (see the plate diagram). For example:
pr  = struct('tau2', 1^2 , 'eta2',0.1^2,'p_u', 0.95, 'rho', 0.3); 
%op:               options for the EP algorithm. For example:
op = struct('damp',0.8, 'damp_decay',0.95, 'robust_updates',2, 'verbosity',0,...
    'max_iter',1000, 'threshold',1e-5, 'min_site_prec',1e-6);
% y                target values (n x 1)
% x                covariates (n x m)
% w_feedbacks      values (1st column) and indices (2nd column) of feedback (n_w_feedbacks x 2)
% gamma_feedbacks  values (1st column, 0/1) and indices (2nd column) of feedback (n_gamma_feedbacks x 2)
% Outputs:
% fa         EP posterior approximation (struct)
% si         EP site terms (struct)
% converged  did EP converge or hit max_iter (1/0)
```
## Simulation and real data study 

The starting point of the code are one of the  main scripts.

main:
    runs the simulation for fixed number of training data and dimensions.
main_all: 
    the number of training data or dimensions can be arrays, e.g, 2:10:300.
main_real_data.m:
    This script is useful for real data. Your data should contain a matrix of X with size (n by p) and a matrix of Y with size (n by 1). In the code there is an example of using the method for Amazon and Yelp dataset. 
    
Depending on the MODE parameter, the type of model and feedback changes:
           MODE = 
                   
           1: Feedback on weight values. Model: spike and slab prior         
           2: Feedback on relevance of features. Model: spike and slab prior

Mode 2 is the most realistic case and it is used for the real data (Amazon and Yelp dataset).           

## Citation

Pedram Daee, Tomi Peltola, Marta Soare and Samuel Kask (2016). Knowledge Elicitation via Sequential Probabilistic Inference for High-Dimensional Prediction

## Team

[![Pedram Daee](https://sites.google.com/site/pedramdaee/_/rsrc/1428612543885/home/Pedram.jpg?height=200&width=152)](https://github.com/PedramDaee) | [![Tomi Peltola](http://research.cs.aalto.fi/pml/personnelpics/tomi.jpg?s=500)](https://github.com/to-mi) | [![Marta Soare](https://users.ics.aalto.fi/msoare/picture.jpg?s=144)](https://users.ics.aalto.fi/msoare/)
---|---|---
[Pedram Daee](https://sites.google.com/site/pedramdaee/home) | [Tomi Peltola](https://github.com/to-mi) | [Marta Soare](https://users.ics.aalto.fi/msoare/)


## License

This project is licensed under the XXXX License - see the [LICENSE.md](LICENSE.md) file for details
