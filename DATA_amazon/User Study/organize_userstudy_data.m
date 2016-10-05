clear all
close all

load('Experiment');

num_keywords = size(keywords,1);
num_users    = size(Responses,1);
z_star_user_exp = zeros(num_users, num_keywords);
%re-ordered the user feedbacks to be similar to Amazon keyword list
for i=1:num_keywords
    z_star_user_exp(:,order(i)) = Responses(:,i);
end
save('z_star_user_exp','z_star_user_exp');