%% PART A (DATA)
m_0 = [-1 1 -1 1]';
C_0 = [2 -0.5 0.3 0; -0.5 1 -0.5 0; 0.3 -0.5 1 0; 0 0 0 2];
m_1 = [1 1 1 1]';
C_1 = [1 0.3 -0.2 0; 0.3 2 0.3 0; -0.2 0.3 1 0; 0 0 0 3];

priors = [0.7 0.3];
N = 1e4;
labels = discreteSample(priors, N) - 1; % Generating 10K labels
dataset = arrayfun(@(label){label * mvnrnd(m_1,C_1)' + (1-label) * mvnrnd(m_0,C_0)'}, labels); % Generating 10K samples in accordance with the labels

labels = logical(labels);

%load('q1data.mat'); % Uncommenting this line will let you use the same data I used in generating results for my submission

steps = 250;
tp = zeros(1,steps);
fp = zeros(1,steps);
fn = zeros(1,steps);
gamma = [linspace(0,5,steps/2) exp(linspace(log(6),25,steps/2))-1]; % Array for varying gamma from 0 to infinity; values up to a certain level have more variety between them, so first half is linear and second half is exponential

parfor k = 1:steps % Parallelizing decision process for speed
    decisions = cell2mat(cellfun(@(X){(mvnpdf(X,m_1,C_1) / mvnpdf(X,m_0,C_0)) > gamma(k)},dataset));
    tp(k) = sum(labels & decisions);
    fp(k) = sum(~labels & decisions);
    fn(k) = sum(labels & ~decisions);
end

tpr = tp / sum(labels); % True Positive Rate (TPR)
fpr = fp / (N - sum(labels)); % False Positive Rate (FPR)

% Minimum P(error) calculation
err_exp = (fp + fn) / N;
[min_err_exp, ind_min_err] = min(err_exp);
min_err_gamma = gamma(ind_min_err);

% Decision rule and TPR, FPR, P(error) calculations for 'theoretically optimal' threshold
decisions_opt = cell2mat(cellfun(@(X){(mvnpdf(X,m_1,C_1) / mvnpdf(X,m_0,C_0)) > priors(1)/priors(2)},dataset));
tpr_opt = sum(labels & decisions_opt) / sum(labels);
fpr_opt = sum(~labels & decisions_opt) / (N - sum(labels));
err_opt = (sum(~labels & decisions_opt) + sum(labels & ~decisions_opt)) / N;

disp(['Empirical min. error: ' num2str(min_err_exp,'%.4f') ' | Theoretical min. error ' num2str(err_opt,'%.4f')]);
disp(['Empirical (TPR,FPR): (' num2str(tpr(ind_min_err),'%.4f') ',' num2str(fpr(ind_min_err),'%.4f') ') | Theoretical (TPR,FRP): (' num2str(tpr_opt,'%.4f') ',' num2str(fpr_opt,'%.4f') ')']);

% True data ROC curve plot
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

% Superimposed plot (part A)
fig_simp = figure;
plot(fpr,tpr,'b','LineWidth',1.5);
hold on;
ylabel('True Positive Rate','FontSize',16);
xlabel('False Positive Rate','FontSize',16);
title('ROC Curves','FontWeight','bold','FontSize',16);
ax_simp = gca;
ax_simp.FontSize = 16;

%% PART B

m_0 = [-1 1 -1 1]';
C_0 = [2 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2];
m_1 = [1 1 1 1]';
C_1 = [1 0 0 0; 0 2 0 0; 0 0 1 0; 0 0 0 3];

parfor k = 1:steps
    decisions = cell2mat(cellfun(@(X){(mvnpdf(X,m_1,C_1) / mvnpdf(X,m_0,C_0)) > gamma(k)},dataset));
    tp(k) = sum(labels & decisions);
    fp(k) = sum(~labels & decisions);
    fn(k) = sum(labels & ~decisions);
end

tpr = tp / sum(labels); % True Positive Rate (TPR)
fpr = fp / (N - sum(labels)); % False Positive Rate (FPR)

% Minimum P(error) calculation
err_exp = (fp + fn) / N;
[min_err_exp, ind_min_err] = min(err_exp);
min_err_gamma = gamma(ind_min_err);

% Decision rule and TPR, FPR, P(error) calculations for 'theoretically optimal' threshold
decisions_opt = cell2mat(cellfun(@(X){(mvnpdf(X,m_1,C_1) / mvnpdf(X,m_0,C_0)) > priors(1)/priors(2)},dataset));
tpr_opt = sum(labels & decisions_opt) / sum(labels);
fpr_opt = sum(~labels & decisions_opt) / (N - sum(labels));
err_opt = (sum(~labels & decisions_opt) + sum(labels & ~decisions_opt)) / N;

disp(['Empirical min. error: ' num2str(min_err_exp,'%.4f') ' | Theoretical min. error ' num2str(err_opt,'%.4f')]);
disp(['Empirical (TPR,FPR): (' num2str(tpr(ind_min_err),'%.4f') ',' num2str(fpr(ind_min_err),'%.4f') ') | Theoretical (TPR,FRP): (' num2str(tpr_opt,'%.4f') ',' num2str(fpr_opt,'%.4f') ')']);

% Naive Bayes ROC curve plot
figure;
plot(fpr,tpr,'LineWidth',1.5);
hold on;
ylabel('True Positive Rate','FontSize',16);
xlabel('False Positive Rate','FontSize',16);
title('ROC Curve (with incorrect $\Sigma$ assumption)','FontWeight','bold','FontSize',16,'Interpreter','latex');
ax = gca;
ax.FontSize = 16;
plot(fpr(ind_min_err),tpr(ind_min_err),'r*','LineWidth',2.5);
txt1 = ['$\leftarrow$ Minimum error point $\gamma_{emp} = ' num2str(min_err_gamma,'%.4f') '$']; 
txt2 = ['$(TPR_{emp},FPR_{emp}) = (' num2str(tpr(ind_min_err),'%.4f') ',' num2str(fpr(ind_min_err),'%.4f') ')$'];
text(fpr(ind_min_err)+0.01,tpr(ind_min_err),txt1,'Interpreter','latex','FontSize',16);
text(fpr(ind_min_err)+0.01,tpr(ind_min_err)-0.1,txt2,'Interpreter','latex','FontSize',16);

% Superimposed plot (part B)
plot(ax_simp,fpr,tpr,'r--','LineWidth',1.5);
legend(ax_simp, 'True data', 'Naive Bayes','Interpreter','latex','FontSize',14);