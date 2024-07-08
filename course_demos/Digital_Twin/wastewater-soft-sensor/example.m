%% Import data
plant_data = readtable('plant_data.csv'); % data at the wastewater treatment plant
DT = plant_data.DT; % date and time, DD-MMM-YYYY, HH:MM:SS
TI1 = plant_data.TI1; % temperature of water inlet to rector 1, in Kelvin
WF = plant_data.WF; % flow of water, in kg/h
TI2 = plant_data.TI2; % temperature of water inlet to rector 2, in Kelvin
T2 = plant_data.T2; % temperature in rector 2, in Kelvin
TO = plant_data.TO; % temperature of water outlet, in Kelvin
S = plant_data.S; % concentration of sulfur compounds predicted by the the old soft sensor, in ppm

lab_data = readtable('lab_data.csv'); % data measured in lab
DTlab = lab_data.DT; % time point of lab data (training + testing)
LS = lab_data.LS; % concentration of sulfur compounds analytically measured in the lab, in ppm

test_data = readtable('test_data.csv'); % data used to test the accuracy of the model
DTtest = test_data.DT; % time point of the testing data

%% Extract data for training and testing
[~, ip, il] = intersect(plant_data.DT, lab_data.DT); % get the indices of the plant and lab data that overlap
training = [plant_data(ip, :), lab_data(il, 'LS')]; % match input data with the acctual sulfur concentration

[~, ip, it] = intersect(plant_data.DT, test_data.DT); % get the indices of the plant and test data that overlap
testing = [plant_data(ip, :), test_data(it, 'LS')]; % match input data with the acctual sulfur concentration

%% Fit regression models
%Xtrain = [training.TI1, training.WF, training.TI2, training.T2, training.TO];
Xtrain = [training.TI1, training.WF, training.TO];
Ytrain = training.LS;

% Machine learning models in MATLAB: https://www.mathworks.com/discovery/machine-learning-models.html

f_lm = @() fitlm(Xtrain,Ytrain); % linear regression
t_lm = timeit(f_lm);
mdl_lm = f_lm();

f_glm = @() fitglm(Xtrain,Ytrain); % generalized linear regression
t_glm = timeit(f_glm);
mdl_glm = f_glm();

f_gpr = @() fitrgp(Xtrain,Ytrain); % Gaussian process regression
t_gpr = timeit(f_gpr);
mdl_gpr = f_gpr();

f_svm = @() fitrsvm(Xtrain,Ytrain); % support vector machine
t_svm = timeit(f_svm);
mdl_svm = f_svm();

f_nn = @() fitrnet(Xtrain,Ytrain); % shallow neutral network
t_nn = timeit(f_nn);
mdl_nn = f_nn();

%% Test regression models
Xtest = [testing.TI1, testing.WF, testing.TI2, testing.T2, testing.TO];
Ytest = testing.LS;

pred_lm = predict(mdl_lm, Xtest);
pred_glm = predict(mdl_glm, Xtest);
pred_gpr = predict(mdl_gpr, Xtest);
pred_svm = predict(mdl_svm, Xtest);
pred_nn = predict(mdl_nn, Xtest);

RMSEold = rmse(testing.LS, testing.S); % RMSE of the old soft sensor
RMSElm = rmse(testing.LS, pred_lm);
RMSEglm = rmse(testing.LS, pred_glm);
RMSEgpr = rmse(testing.LS, pred_gpr);
RMSEsvm = rmse(testing.LS, pred_svm);
RMSEnn = rmse(testing.LS, pred_nn);

%% Print results

fprintf('Results \n');
fprintf('------- \n');

fprintf('Old soft sensor \n');
fprintf('RMSE: %.4f \n \n', RMSEold);

fprintf('Linear regression: \n');
fprintf('RMSE: %.4f \n', RMSElm);
fprintf('Time: %.4f \n \n', t_lm);

fprintf('Generalized linear regression: \n');
fprintf('RMSE: %.4f \n', RMSEglm);
fprintf('Time: %.4f \n \n', t_glm);

fprintf('Gaussian process regression: \n');
fprintf('RMSE: %.4f \n', RMSEgpr);
fprintf('Time: %.4f \n \n', t_gpr);

fprintf('Support vector machine: \n');
fprintf('RMSE: %.4f \n', RMSEsvm);
fprintf('Time: %.4f \n \n', t_svm);

fprintf('Shallow neutral network: \n');
fprintf('RMSE: %.4f \n', RMSEnn);
fprintf('Time: %.4f \n \n', t_nn);