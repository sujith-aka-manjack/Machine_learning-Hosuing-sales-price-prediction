% Registration ID: 210110050
% Name: Sujith Kurian James

% This is a function to split the dataset into training data and validation data and plot the learning curve using linear regression model
% Keep all the files in the same folder while running
% Change the validation ratio as required within the function
function [] = learning_curve() 
%The cleaned and preprocessed data from Orange is saved into a file and loaded into the MATLAB program
%Name of the dataset file is CW_processed_data.mat
data = load('CW_processed_data.mat');   %loading the data
data = data.CWprocesseddata;
rng (1); %Assign a seed to the random number generator in order to ensure that it produces the same
             %random sequence all the time
validation_ratio = 0.2;  %Change this to change the ratio at which data is split 

%Logic to create random indices for selecting training and validation data
%Taken from lab sheet 3 and modified accordingly
shuffled_indices = ceil(rand(height(data),1) * height(data));
validation_set_size = int16(height(data) * validation_ratio);
validation_indices = shuffled_indices(1: validation_set_size);
train_indices = shuffled_indices(validation_set_size:end);
train_data = data(train_indices, 1:end);
validation_data = data(validation_indices, 1:end);

%Preallocation for faster processing of program
training_rmse = [ones(1,height(train_data))];    
validation_rmse = [ones(1,height(train_data))];

%Calculating the data points for the learning curve
for m=1:height(train_data)
   X_train = train_data(1:m,1:end-1);   %Independent variables of training dataset
   X_train = table2array(X_train);      %Converting table into array
   Y_train = train_data(1:m,end);       %Target variables of training dataset
   Y_train = table2array(Y_train);      
   X_valid = validation_data(:,1:end-1); %Independent variables of validation dataset
   X_valid = table2array(X_valid);
   Y_valid = validation_data(:,end);     %Target variables of validation dataset
   Y_valid = table2array(Y_valid);
   
   %Calculating the coefficients for the linear regression using normal equation and predicting target value
   %Taken from lab sheet 2 and modified accordingly
   Psi = [ones(m,1),X_train];
   theta_hat = pinv(Psi'*Psi)*Psi'*Y_train; %Using psuedo inverse since regular inverse does not exist
   predict_train = Psi*theta_hat;           %Predicting target variable for the training dataset
   Psi_star = [ones(height(X_valid),1) X_valid];
   predict_validation = Psi_star*theta_hat; %Predicting target variable for the validation dataset
   
   %Calculating errors
   training_errors = (predict_train - Y_train(1:m));
   training_rmse(m) = sqrt(mean((training_errors).^2));     %RMSE value for the training data
   validation_errors = (predict_validation - Y_valid(1:end));
   validation_rmse(m) = sqrt(mean((validation_errors).^2)); %RMSE value for the validation data
end

   %Plotting the learning curve
   %Plotting the data from 20 in order to get proper scaling of the axis
   figure
   plot(25:length(training_rmse), training_rmse(25:end), 'r')
   hold on
   plot(25:length(validation_rmse), validation_rmse(25:end), 'b')
   xlabel('Training set size');
   ylabel('RMSE');
   legend('training rmse','validation rmse');
end