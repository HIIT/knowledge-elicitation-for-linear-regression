clear all
close all

load('Loss_functions')
load('decisions')


num_methods = size(decisions,1);
num_iterations = size(decisions,2);
num_runs = size(decisions,3);

figure
plot(mean(Loss_1,3)','.-');
legend('UCB','Uniformly random','random on the relevelant features','max variance')
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value X(theta-theta_star)')

figure
plot(mean(Loss_2,3)','.-');
legend('UCB','Uniformly random','random on the relevelant features','max variance')
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value (theta-theta_star)')

for method =1 : num_methods
    figure
    data = reshape(decisions(method,:,:),[num_iterations*num_runs,1]);
    hist(data,10)
end


    