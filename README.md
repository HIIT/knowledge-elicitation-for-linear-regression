# Expert Knowledge Elicitation for Linear Models
This repository contains code for expert knowledge elicitation in sparse linear regression where the user can provide feedback about the covariates (coefficient value or relevance) with the goal of improving the predictions. The model uses a spike-and-slab prior on the regression coefficients and therefore it is especially useful for "small n, large p" problems, i.e., when the number of training samples is smaller than the number of dimensions. 


<p align="center">
  <img src="/Plate diagram.JPG" width="500"/>
</p>

## Basic Usage

The script [linreg_sns_ep.m](linreg_sns_ep.m) contains the posterior inference and the function [decision_policy.m](decision_policy.m) has implementation of different query methods [1]. 

```matlab
%Inputs: 
% x                covariates (n x m). For example:
% y                target values (n x 1). For example: 
% pr:              prior and other fixed model parameters (see the plate diagram). For example:
% op:              options for the EP algorithm. For example:
% w_feedbacks      values (1st column) and indices (2nd column) of feedback (n_w_feedbacks x 2)
% gamma_feedbacks  values (1st column, 0/1) and indices (2nd column) of feedback (n_gamma_feedbacks x 2)
% Outputs:
% fa         EP posterior approximation (struct)
% si         EP site terms (struct)
% converged  did EP converge or hit max_iter (1/0)

[fa, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, w_feedbacks, gamma_feedbacks, si)
```
### Example scenario
 100 dimensional data, only 10 first coefficients are relvant, only 10 training data is available. Feedback only on relevance of features.

```matlab
%unknown coefficinet values (only first ten are non-zero)
num_features = 100;
num_relevant = 10;
w_star = [randn(num_relevant, 1); zeros(num_features-num_relevant,1)];
%create 10 training data
x = randn(10,num_features);
y = normrnd(x*w_star, 1);
%create some test data
x_test = randn(1000,num_features);
y_test = normrnd(x_test*w_star, 1);
%initialize the inputs
pr  = struct('tau2', 1^2 , 'eta2',0.1^2,'p_u', 0.95, 'rho', 0.3, ...
    'sigma2_prior',true,'sigma2_a',1,'sigma2_b',1 );
op = struct('damp',0.8, 'damp_decay',0.95, 'robust_updates',2, 'verbosity',0,...
    'max_iter',1000, 'threshold',1e-5, 'min_site_prec',1e-6);
%we only have feedback about relevance of coefficients
gamma_feedbacks = [[ones(num_relevant,1);zeros(num_features-num_relevant,1)], [1:num_features]'];
%results with feedback (the proposed model)
[fa_fb, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, [], gamma_feedbacks, []);
MSE_with_fb = mean((x_test*fa_fb.w.Mean- y_test).^2); 
%results without feedback (only spike and slab model)
[fa, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, [], [], []);
MSE_without_fb = mean((x_test*fa.w.Mean- y_test).^2);
%ridge regression solution
w_ridge = inv(eye(num_features) + (x'*x)) * (x'*y);
MSE_ridge = mean((x_test*w_ridge- y_test).^2);

disp('Mean Squared Error on test data:')
disp(['The proposed method with user feedback:',num2str(MSE_with_fb)])
disp(['The proposed method without user feedback:',num2str(MSE_without_fb)])
disp(['Ridge regression:',num2str(MSE_ridge)])
```
For more examples please check the main scripts.

## Simulation and real data study 

The starting point of the code are one of the main scripts.
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

[1] Pedram Daee, Tomi Peltola, Marta Soare and Samuel Kaski (2016). Knowledge Elicitation via Sequential Probabilistic Inference for High-Dimensional Prediction

## Team

[![Pedram Daee](https://sites.google.com/site/pedramdaee/_/rsrc/1428612543885/home/Pedram.jpg?height=200&width=152)](https://github.com/PedramDaee) | [![Tomi Peltola](http://research.cs.aalto.fi/pml/personnelpics/tomi.jpg?s=500)](https://github.com/to-mi) | [![Marta Soare](https://users.ics.aalto.fi/msoare/picture.jpg?s=144)](https://users.ics.aalto.fi/msoare/)
---|---|---
[Pedram Daee](https://sites.google.com/site/pedramdaee/home) | [Tomi Peltola](https://github.com/to-mi) | [Marta Soare](https://users.ics.aalto.fi/msoare/)


## License

This project is licensed under the XXXX License - see the [LICENSE.md](LICENSE.md) file for details
