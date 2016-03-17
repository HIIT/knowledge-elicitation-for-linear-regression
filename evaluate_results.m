clear all
close all

load('results')


num_methods = size(decisions,1);
num_iterations = size(decisions,2);
num_runs = size(decisions,3);
Method_list = char('Max(90% UCB,90% LCB)','Uniformly random','random on the relevelant features','max variance');

figure
plot(mean(Loss_1,3)','.-');
legend(Method_list)
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value X(theta - theta*)')

figure
plot(mean(Loss_2,3)','.-');
legend(Method_list)
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value (theta - theta*)')



% hold on
for method =1 : num_methods
    figure
    plot(decisions(method,:,1)','.');
    legend(Method_list(method,:));
end
% legend(Method_list)

% for method =1 : num_methods
%     figure
%     data = reshape(decisions(method,:,:),[num_iterations*num_runs,1]);
%     hist(data,10)
% end
% 

    