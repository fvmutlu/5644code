clear variables; close all; clc;
%% Generate Data
[train_data, train_labels] = generateMultiringDataset(2,1000);
[test_data, test_labels] = generateMultiringDataset(2,10000);

%% Load and visualize data used in submission

load('hw3q2data.mat');
subplot(1,2,1);
ax = gca;
ax.FontSize = 16;
X = train_data(1, train_labels==1);
Y = train_data(2, train_labels==1);
scatter(X,Y,'b.');
hold on;
X = train_data(1, train_labels==2);
Y = train_data(2, train_labels==2);
scatter(X,Y,'r.');
set(gca, 'FontSize', 14);
xlabel('$x_1$','Interpreter','latex','FontSize',16);
ylabel('$x_2$','Interpreter','latex','FontSize',16);
title('Test dataset','FontSize',16);
legend('Class 1','Class 2','FontSize',16,'Location','northeast');
subplot(1,2,2);
ax = gca;
ax.FontSize = 16;
X = test_data(1, test_labels==1);
Y = test_data(2, test_labels==1);
scatter(X,Y,'b.');
hold on;
X = test_data(1, test_labels==2);
Y = test_data(2, test_labels==2);
scatter(X,Y,'r.');
set(gca, 'FontSize', 14);
xlabel('$x_1$','Interpreter','latex','FontSize',16);
ylabel('$x_2$','Interpreter','latex','FontSize',16);
title('Test dataset','FontSize',16);
legend('Class 1','Class 2','FontSize',16,'Location','northeast');

%% Murphy book suggestion for box constraint and kernel scale ranges

C_arr = 2.^(-3:2:19);
gamma_arr = 2.^(-11:2:15);
sigma_arr = 1./sqrt(2*gamma_arr);

%% Prof. Erdogmus' code values for box constraint and kernel scale ranges

C_arr = 10.^linspace(-1,9,11);
sigma_arr = 10.^linspace(-2,3,13);

%% Non-standarized
err_nonstd_mat = zeros(numel(C_arr),numel(sigma_arr));
C_idx = 0;
for C = C_arr
    C_idx = C_idx + 1;
    sigma_idx = 0;
    for sigma = sigma_arr
        sigma_idx = sigma_idx + 1;
        CVSVM_model = fitcsvm(train_data',train_labels,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma,'CrossVal','on'); % This is equivalent to the above two lines
        err = zeros(1,10);
        for k = 1:10
            fold_decisions = CVSVM_model.Trained{k}.predict(train_data')';
            err(k) = sum(fold_decisions ~= train_labels) / numel(train_labels);
        end
        err_nonstd_mat(C_idx, sigma_idx) = mean(err);
    end
end
[~, j] = min(min(err_nonstd_mat));
[~, i] = min(err_nonstd_mat);
i = i(j);

figure;
surf(C_arr,sigma_arr,err_nonstd_mat');
hold on;
set(gca,'xscale','log');
set(gca,'yscale','log');
plot3(C_arr(i),sigma_arr(j),err_nonstd_mat(i,j),'ro','markersize',15);
xlabel("$C$",'FontSize',16,'Interpreter','latex');
ylabel("$\sigma$",'FontSize',16,'Interpreter','latex');
zlabel("Min. avg. cv error rate",'FontSize',16);

SVM_model_nonstd = fitcsvm(train_data',train_labels,'BoxConstraint',C_arr(i),'KernelFunction','gaussian','KernelScale',sigma_arr(j));
test_decisions = SVM_model_nonstd.predict(test_data')';
err_nonstd = sum(test_decisions ~= test_labels) / numel(test_labels);

%% Standardized for comparison using Murphy book suggestion

err_std_mat = zeros(numel(C_arr),numel(sigma_arr));
for C_idx = 1:numel(C_arr)
    C = C_arr(C_idx);
    parfor sigma_idx = 1:numel(sigma_arr)
        sigma = sigma_arr(sigma_idx);
        CVSVM_model = fitcsvm(train_data',train_labels,'Standardize',true,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma,'CrossVal','on'); % This is equivalent to the above two lines
        err = zeros(1,10);
        for k = 1:10
            fold_decisions = CVSVM_model.Trained{k}.predict(train_data')';
            err(k) = sum(fold_decisions ~= train_labels) / numel(train_labels);
        end
        err_std_mat(C_idx, sigma_idx) = mean(err);
    end
end

[~, j] = min(min(err_std_mat));
[~, i] = min(err_std_mat);
i = i(j);

figure;
surf(C_arr,sigma_arr,err_std_mat');
hold on;
set(gca,'xscale','log');
set(gca,'yscale','log');
plot3(C_arr(i),sigma_arr(j),err_std_mat(i,j),'ro','markersize',15);
xlabel("$C$",'FontSize',16,'Interpreter','latex');
ylabel("$\sigma$",'FontSize',16,'Interpreter','latex');
zlabel("Min. avg. cv error rate",'FontSize',16);

SVM_model_std = fitcsvm(train_data',train_labels,'Standardize',true,'BoxConstraint',C_arr(i),'KernelFunction','gaussian','KernelScale',sigma_arr(j));
test_decisions = SVM_model_std.predict(test_data')';
err_std = sum(test_decisions ~= test_labels) / numel(test_labels);