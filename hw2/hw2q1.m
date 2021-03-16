clear variables; close all; clc;

% Note that I don't have a fixed set of data or other workspace variables
% for this question. Your exact numerical results may differ from the
% results in my submission when running this code; they should have the
% same trends.

[xTrain,yTrain,xValidate,yValidate] = hw2q1_datagen;

N = length(xTrain);

phi = @(x)([1; x(1); x(2); x(1)^2; x(1)*x(2); x(2)^2; x(1)^3; x(1)^2*x(2); x(1)*x(2)^2; x(2)^3]);
xTrain_cell = mat2cell(xTrain',ones(1,length(xTrain')));

Phi = cell2mat(cellfun(phi,xTrain_cell,'UniformOutput',false)')';
w_ml = (Phi' * Phi)\(Phi' * yTrain');

var_ml = (1/N) * sum((yTrain - w_ml' * Phi').^2);

xValidate_cell = mat2cell(xValidate',ones(1,length(xValidate')));
Phi_validate = cell2mat(cellfun(phi,xValidate_cell,'UniformOutput',false)')'; % this is the vector representing \phi(x_n) values from 1 to N
ml_err = (1/2) * sum((yValidate - w_ml' * Phi_validate').^2);

steps = 10000;
gamma_range = linspace(1e-4,1,steps);
w_map = @(gamma)(((var_ml/gamma)*eye(length(Phi' * Phi)) + Phi' * Phi)\(Phi' * yTrain'));
%w_map2 = @(gamma)(((1/gamma)*eye(length(Phi' * Phi)) + Phi' * Phi)\(Phi'*yTrain')); % The variables/functions ending with a '2' are for test purposes to see the effects of using ML estimate for sigma^2 as opposed to just picking it to be 1
w_map = arrayfun(w_map, gamma_range,'UniformOutput',false);
%w_map2 = arrayfun(w_map2, gamma_range,'UniformOutput',false);
map_err = cellfun(@(w)((1/2) * sum((yValidate - w' * Phi_validate').^2)),w_map);
%map_err2 = cellfun(@(w)((1/2) * sum((yValidate - w' * Phi_validate').^2)),w_map2);

figure;
plot(gamma_range, map_err/N,'LineWidth',1.5);
title('Avg square error for MAP estimate in validation set with varying $\gamma$','FontSize',16,'FontWeight','bold','Interpreter','latex');
xlabel('$\gamma$','FontSize',16,'FontWeight','bold','Interpreter','latex');
ylabel('Avg. square error','FontSize',16,'FontWeight','bold','Interpreter','latex');
%hold on;
%plot(gamma_range, map_err2/N);
%legend('$\lambda = \frac{\sigma^2}{\gamma}$','$\lambda = \frac{1}{\gamma}$','FontSize',16,'FontWeight','bold','Interpreter','latex','Location','southeast');


