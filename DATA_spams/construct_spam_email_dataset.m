%% spam email database 
%ftp://ftp.ics.uci.edu/pub/machine-learning-databases/spambase/spambase.DOCUMENTATION
%https://archive.ics.uci.edu/ml/datasets/Spambase
%4601 observations, 57 features + 1 class (y value)

A = load('spam_dataset.txt'); % directly taken from ftp://ftp.ics.uci.edu/pub/machine-learning-databases/spambase/spambase.data

X_all=A(:,1:57);
Y_all=A(:,58);

save('spam_email_data','X_all','Y_all','-v7.3');
