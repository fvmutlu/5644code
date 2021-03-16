clear variables; close all; clc;

%% Sample generation

mu_01 = [3 0]';
sigma_01 = [2 0; 0 1];
mu_02 = [0 3]';
sigma_02 = [1 0; 0 2];
pdf_0 = @(x)(0.5*mvnpdf(x, mu_01, sigma_01) + 0.5*mvnpdf(x, mu_02, sigma_02)); % Class conditional pdf for class 0

mu_1 = [2 2]';
sigma_1 = [1 0; 0 1];
pdf_1 = @(x)(mvnpdf(x, mu_1, sigma_1));

priors = [0.65 0.35];
zero_sampler = @(coin)(coin * mvnrnd(mu_01,sigma_01) + (1-coin) * mvnrnd(mu_02,sigma_02));
one_sampler = @()(mvnrnd(mu_1,sigma_1));
sampler = @(label)(label * one_sampler() + (1-label) * zero_sampler(randi([0 1])));

L20_train = discreteSample(priors,20) - 1;
D20_train = arrayfun(sampler, L20_train, 'UniformOutput', false);

L200_train = discreteSample(priors,200) - 1;
D200_train = arrayfun(sampler, L200_train, 'UniformOutput', false);

L2K_train = discreteSample(priors,2e3) - 1;
D2K_train = arrayfun(sampler, L2K_train, 'UniformOutput', false);

L10K_validate = discreteSample(priors,1e4) - 1;
D10K_validate = arrayfun(sampler, L10K_validate, 'UniformOutput', false);

%% Min-P(error) classification
N = 1e4;
steps = 250;
tp = zeros(1,steps);
fp = zeros(1,steps);
fn = zeros(1,steps);
gamma = [linspace(0,5,steps/2) exp(linspace(log(6),25,steps/2))-1]; % Array for varying gamma from 0 to infinity; values up to a certain level have more variety between them, so first half is linear and second half is exponential

parfor k = 1:steps % Parallelizing decision process for speed
    decisions = cell2mat(cellfun(@(X){(pdf_1(X') / pdf_0(X')) > gamma(k)}, D10K_validate));
    tp(k) = sum(L10K_validate & decisions);
    fp(k) = sum(~L10K_validate & decisions);
    fn(k) = sum(L10K_validate & ~decisions);
end

tpr = tp / sum(L10K_validate); % True Positive Rate (TPR)
fpr = fp / (N - sum(L10K_validate)); % False Positive Rate (FPR)

% Minimum P(error) calculation
err_exp = (fp + fn) / N;
[min_err_exp, ind_min_err] = min(err_exp);
min_err_gamma = gamma(ind_min_err);

% Decision rule and TPR, FPR, P(error) calculations for 'theoretically optimal' threshold
decisions_opt = cell2mat(cellfun(@(X){(pdf_1(X') / pdf_0(X')) > priors(1)/priors(2)},D10K_validate));
tpr_opt = sum(L10K_validate & decisions_opt) / sum(L10K_validate);
fpr_opt = sum(~L10K_validate & decisions_opt) / (N - sum(L10K_validate));
err_opt = (sum(L10K_validate ~= decisions_opt)) / N;

disp(['Empirical min. error: ' num2str(min_err_exp,'%.4f') ' | Theoretical min. error ' num2str(err_opt,'%.4f')]);
disp(['Empirical (TPR,FPR): (' num2str(tpr(ind_min_err),'%.4f') ',' num2str(fpr(ind_min_err),'%.4f') ') | Theoretical (TPR,FRP): (' num2str(tpr_opt,'%.4f') ',' num2str(fpr_opt,'%.4f') ')']);

[x1, x2] = meshgrid(linspace(-5,10,1e3), linspace(-5,10,1e3));
x1x2 = [x1(:) x2(:)]';
x1x2 = mat2cell(x1x2,2,ones(1,1e6));
decision_boundary_opt = cell2mat(cellfun(@(X){(pdf_1(X) / pdf_0(X)) > priors(1)/priors(2)},x1x2));
decision_boundary_opt = reshape(decision_boundary_opt, size(x1));

D20_train = cell2mat(D20_train.')';
D200_train = cell2mat(D200_train.')';
D2K_train = cell2mat(D2K_train.')';
D10K_validate = cell2mat(D10K_validate.')';

%%

%load('hw2q2data.mat'); % You can uncomment this and skip all the parts
%before this to operate on the data that I used for the report submission

%% Part 1 plots
d10k_opt_fig = figure;
boundaryVisualizer(D10K_validate,decision_boundary_opt,L10K_validate,d10k_opt_fig);
title('Min-P(error) boundary (validation set)','FontSize',16,'FontWeight','bold');

% ROC curve plot
figure;
plot(fpr,tpr,'LineWidth',1.5);
hold on;
ylabel('True Positive Rate','FontSize',16);
xlabel('False Positive Rate','FontSize',16);
title('ROC Curve','FontWeight','bold','FontSize',16);
ax = gca;
ax.FontSize = 16;
plot(fpr(ind_min_err),tpr(ind_min_err),'r*','LineWidth',2.5);
txt1 = ['$\leftarrow$ Minimum error point, $\gamma_{emp} = ' num2str(min_err_gamma,'%.4f') '$']; 
txt2 = ['$(TPR_{emp},FPR_{emp}) = (' num2str(tpr(ind_min_err),'%.4f') ',' num2str(fpr(ind_min_err),'%.4f') ')$'];
text(fpr(ind_min_err)+0.01,tpr(ind_min_err),txt1,'Interpreter','latex','FontSize',16);
text(fpr(ind_min_err)+0.01,tpr(ind_min_err)-0.1,txt2,'Interpreter','latex','FontSize',16);

%% Logistic-linear ML

sigm = @(x)(1/(1+exp(-x)));
% D20
A = @(w)(w' * [ones(1,length(D20_train)); D20_train]);
Y = @(w)(arrayfun(sigm,A(w)));
nll = @(w)((-1) * (L20_train*log(Y(w)') + (1-L20_train)*log(1-Y(w)')));
w_0 = 0.5*ones(3,1);
w_ml20 = fminsearch(nll,w_0);

% D200
A = @(w)(w' * [ones(1,length(D200_train)); D200_train]);
Y = @(w)(arrayfun(sigm,A(w)));
nll = @(w)((-1) * (L200_train*log(Y(w)') + (1-L200_train)*log(1-Y(w)')));
w_0 = 0.5*ones(3,1);
w_ml200 = fminsearch(nll,w_0);

% D2K
A = @(w)(w' * [ones(1,length(D2K_train)); D2K_train]);
Y = @(w)(arrayfun(sigm,A(w)));
nll = @(w)((-1) * (L2K_train*log(Y(w)') + (1-L2K_train)*log(1-Y(w)')));
w_0 = 0.5*ones(3,1);
w_ml2k = fminsearch(nll,w_0);

D10K_validate = mat2cell(D10K_validate,2,ones(1,1e4));

decisions_20 = cell2mat(cellfun(@(x){sigm(w_ml20' * [1; x(1); x(2)]) > 0.5},D10K_validate));
decision_boundary_20 = cell2mat(cellfun(@(x){sigm(w_ml20' * [1; x(1); x(2)]) > 0.5},x1x2));
decision_boundary_20 = reshape(decision_boundary_20, size(x1));

decisions_200 = cell2mat(cellfun(@(x){sigm(w_ml200' * [1; x(1); x(2)]) > 0.5},D10K_validate));
decision_boundary_200 = cell2mat(cellfun(@(x){sigm(w_ml200' * [1; x(1); x(2)]) > 0.5},x1x2));
decision_boundary_200 = reshape(decision_boundary_200, size(x1));

decisions_2K = cell2mat(cellfun(@(x){sigm(w_ml2k' * [1; x(1); x(2)]) > 0.5},D10K_validate));
decision_boundary_2K = cell2mat(cellfun(@(x){sigm(w_ml2k' * [1; x(1); x(2)]) > 0.5},x1x2));
decision_boundary_2K = reshape(decision_boundary_2K, size(x1));

%% Logistic-quadratic ML
phi = @(x)([1; x(1); x(2); x(1)^2; x(1)*x(2); x(2)^2]);

% D20
quad20_train = cell2mat(cellfun(phi, mat2cell(D20_train,2,ones(1,20)), 'UniformOutput', false));
A = @(w)(w' * quad20_train);
Y = @(w)(arrayfun(sigm,A(w)));
nll = @(w)((-1) * (L20_train*log(Y(w)') + (1-L20_train)*log(1-Y(w)')));
w_0 = 0.5*ones(6,1);
quad_w_ml20 = fminsearch(nll,w_0);
quad_decisions_20 = cell2mat(cellfun(@(x){sigm(quad_w_ml20' * phi(x)) > 0.5},D10K_validate));
quad_decision_boundary_20 = cell2mat(cellfun(@(x){sigm(quad_w_ml20' * phi(x)) > 0.5},x1x2));
quad_decision_boundary_20 = reshape(quad_decision_boundary_20, size(x1));

% D200
quad200_train = cell2mat(cellfun(phi, mat2cell(D200_train,2,ones(1,200)), 'UniformOutput', false));
A = @(w)(w' * quad200_train);
Y = @(w)(arrayfun(sigm,A(w)));
nll = @(w)((-1) * (L200_train*log(Y(w)') + (1-L200_train)*log(1-Y(w)')));
w_0 = 0.5*ones(6,1);
quad_w_ml200 = fminsearch(nll,w_0);
quad_decisions_200 = cell2mat(cellfun(@(x){sigm(quad_w_ml200' * phi(x)) > 0.5},D10K_validate));
quad_decision_boundary_200 = cell2mat(cellfun(@(x){sigm(quad_w_ml200' * phi(x)) > 0.5},x1x2));
quad_decision_boundary_200 = reshape(quad_decision_boundary_200, size(x1));

% D2K
quad2K_train = cell2mat(cellfun(phi, mat2cell(D2K_train,2,ones(1,2e3)), 'UniformOutput', false));
A = @(w)(w' * quad2K_train);
Y = @(w)(arrayfun(sigm,A(w)));
nll = @(w)((-1) * (L2K_train*log(Y(w)') + (1-L2K_train)*log(1-Y(w)')));
w_0 = 0.5*ones(6,1);
quad_w_ml2k = fminsearch(nll,w_0);
quad_decisions_2K = cell2mat(cellfun(@(x){sigm(quad_w_ml2k' * phi(x)) > 0.5},D10K_validate));
quad_decision_boundary_2K = cell2mat(cellfun(@(x){sigm(quad_w_ml2k' * phi(x)) > 0.5},x1x2));
quad_decision_boundary_2K = reshape(quad_decision_boundary_2K, size(x1));

%% Part 2 linear plots
d20_fig = figure;
boundaryVisualizer(D20_train,decision_boundary_20,L20_train,d20_fig);
title('$D^{20}_{train}$ linear boundary (training set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

d200_fig = figure;
boundaryVisualizer(D200_train,decision_boundary_200,L200_train,d200_fig);
title('$D^{200}_{train}$ linear boundary (training set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

d2K_fig = figure;
boundaryVisualizer(D2K_train,decision_boundary_2K,L2K_train,d2K_fig);
title('$D^{2000}_{train}$ linear boundary (training set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

D10K_validate = cell2mat(D10K_validate);

d10k_20_fig = figure;
boundaryVisualizer(D10K_validate,decision_boundary_20,L10K_validate,d10k_20_fig);
title('$D^{20}_{train}$ linear boundary (validation set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

d10k_200_fig = figure;
boundaryVisualizer(D10K_validate,decision_boundary_200,L10K_validate,d10k_200_fig);
title('$D^{200}_{train}$ linear boundary (validation set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

d10k_2K_fig = figure;
boundaryVisualizer(D10K_validate,decision_boundary_2K,L10K_validate,d10k_2K_fig);
title('$D^{2000}_{train}$ linear boundary (validation set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

%% Part 2 quadratic plots
quad_d20_fig = figure;
boundaryVisualizer(D20_train,quad_decision_boundary_20,L20_train,quad_d20_fig);
title('$D^{20}_{train}$ quadratic boundary (training set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

quad_d200_fig = figure;
boundaryVisualizer(D200_train,quad_decision_boundary_200,L200_train,quad_d200_fig);
title('$D^{200}_{train}$ quadratic boundary (training set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

quad_d2K_fig = figure;
boundaryVisualizer(D2K_train,quad_decision_boundary_2K,L2K_train,quad_d2K_fig);
title('$D^{2000}_{train}$ quadratic boundary (training set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

d10k_quad20_fig = figure;
boundaryVisualizer(D10K_validate,quad_decision_boundary_20,L10K_validate,d10k_quad20_fig);
title('$D^{20}_{train}$ quadratic boundary (validation set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

d10k_quad200_fig = figure;
boundaryVisualizer(D10K_validate,quad_decision_boundary_200,L10K_validate,d10k_quad200_fig);
title('$D^{200}_{train}$ quadratic boundary (validation set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

d10k_quad2K_fig = figure;
boundaryVisualizer(D10K_validate,quad_decision_boundary_2K,L10K_validate,d10k_quad2K_fig);
title('$D^{2000}_{train}$ quadratic boundary (validation set)','Interpreter','latex','FontSize',16,'FontWeight','bold');

%% Evaluation
err = @(decisions)(sum(decisions ~= L10K_validate)/1e4);

disp([err(decisions_20), err(decisions_200), err(decisions_2K), err(quad_decisions_20), err(quad_decisions_200), err(quad_decisions_2K)]);

%% Functions

function visualizer(data,labels,fig)
    set(0, 'CurrentFigure', fig);
    ax = gca;
    ax.FontSize = 16;
    X = data(1, labels==0);
    Y = data(2, labels==0);
    scatter(X,Y,'b.');
    hold on;
    X = data(1, labels==1);
    Y = data(2, labels==1);
    scatter(X,Y,'r.');
    set(gca, 'FontSize', 14);
    xlim([-5 10]);
    ylim([-5 10]);
    xlabel('$x_1$','Interpreter','latex','FontSize',16);
    ylabel('$x_2$','Interpreter','latex','FontSize',16);
    legend('Class 1','Class 2','FontSize',16,'Location','northeast');
end

function boundaryVisualizer(data,boundary,labels,fig)
    set(0, 'CurrentFigure', fig);
    imagesc(linspace(-5,10,1e3),linspace(-5,10,1e3),boundary); % Plot decision boundaries
    hold on;
    set(gca,'ydir','normal');
    colormap([0.75 0.75 1; 1 0.75 0.75]);
    visualizer(data, labels, fig);
end