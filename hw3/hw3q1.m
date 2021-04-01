clear variables; close all; clc;

means = [1 1 1; 1 -1 1; -1 1 -1; -1 -1 -1]';
covs = {rand(3,3) rand(3,3) rand(3,3) rand(3,3)};
covs = cellfun(@(x)(x * x'), covs, 'UniformOutput', false);
covs = cellfun(@(x)( 0.5 + 0.9 * ( x / max(x(:)) ) ), covs, 'UniformOutput', false);

samplegen = @(label)(mvnrnd(means(:,label), covs{label}));
datasetgen = @(labels)(cell2mat(arrayfun(samplegen,labels,'UniformOutput',false).')');

train_labels = {randi([1 4],[1 100]) randi([1 4],[1 200]) randi([1 4],[1 500]) randi([1 4],[1 1000]) randi([1 4],[1 2000]) randi([1 4],[1 5000])};
train_targets = cellfun(@(x)(full(ind2vec(x))),train_labels,'UniformOutput',false);
train_data = cellfun(datasetgen, train_labels, 'UniformOutput', false);

test_labels = randi([1 4],[1 1e5]);
test_targets = full(ind2vec(test_labels));
test_data = datasetgen(test_labels);

%% Pre-generated data used in the solutions document

load('hw3q1data.mat'); % Start here if you'd like to replicate my results from my assignment submission

%% Visualize data

train_fig = figure;
titles = ["Training set, 100 samples", "Training set, 200 samples", "Training set, 500 samples", "Training set, 1000 samples", "Training set, 2000 samples", "Training set, 5000 samples"];
for set = 1:6
    subplot(2,3,set);
    visualizer(train_data{set},train_labels{set},train_fig);
    title(titles(set),'FontSize',16);
end
val_fig = figure;
visualizer(test_data,test_labels,val_fig);
title("Test dataset",'FontSize',16);

%% Theoretically optimal classifier

pdfs = @(X)([mvnpdf(X, means(:,1), covs{1}) mvnpdf(X, means(:,2), covs{2}) mvnpdf(X, means(:,3), covs{3}) mvnpdf(X, means(:,4), covs{4})]);
train_decisions_opt = cell(1,6);
train_err_opt = zeros(1,6);

for set = 1:6
    data = num2cell(train_data{set},1);
    likelihoods = cellfun(pdfs,data,'UniformOutput',false);
    [~, train_decisions_opt{set}] = cellfun(@max,likelihoods); % We don't need to include priors since the prior distribution is uniform (and therefore classes have the same prior probability)
    train_err_opt(set) = sum(train_decisions_opt{set} ~= train_labels{set}) / numel(train_labels{set});
end
data = num2cell(test_data,1);
likelihoods = cellfun(pdfs,data,'UniformOutput',false);
[~, test_decisions_opt] = cellfun(@max,likelihoods);
test_err_opt = sum(test_decisions_opt ~= test_labels) / numel(test_labels);

%% Model order selection

P_arrays = cell(1,6);
maxP = 12;
figure;
hold on;
for set = 1:6
    P_arrays{set} = xVal(train_data{set},train_labels{set},train_targets{set},maxP);
    plot(1:maxP,P_arrays{set},'LineWidth',1.5);
end
title('Accuracy w.r.t number of perceptrons','FontSize',16);
xlabel('Number of perceptrons','FontSize',16);
ylabel('Accuracy','FontSize',16);
legend(["100 samples" "200 samples" "500 samples" "1000 samples" "2000 samples" "5000 samples"],'FontSize',16);

%% Training the final MLPs

best_P = [4 5 3 4 3 4]; % These values are adjusted for the data I used (in hw3q1data.mat), they're hard-coded because the heuristic I used to pick the values is not easily represented in code
nets = cell(1,6);
err = zeros(1,6);
for set = 1:6
    nets{set} = trainModel(train_data{set},train_targets{set},best_P(set));
    [~,pred] = max(nets{set}(test_data));
    err(set) = sum(pred ~= test_labels)/numel(test_labels);
end
errfig = figure;
semilogx([100 200 500 1000 2000 5000], err, 'LineWidth', 1.5);
hold on;
semilogx([100 200 500 1000 2000 5000], test_err_opt * ones(1,6), 'LineWidth', 1.5);
xlim([100 5000]);
xticks([100 200 500 1000 2000 5000]);
xticklabels({'100','200','500','1000','2000','5000'});
title("Error rates on test set",'FontSize',16);
xlabel("Number of samples in training set",'FontSize',16);
ylabel("Error rate",'FontSize',16);
legend(["Trained networks" "Theoretically optimal"],'FontSize',16);

%% Auxiliary functions

function trainedNet = trainModel(data,targets,P) % This function is for training the MLPs without 10-fold cross validation
    net = patternnet(P,'traingdx','crossentropy');
    net.divideFcn = 'dividetrain';
    best_perf = Inf;
    for i = 1:50 % Reinitialize training 50 times and take the best local optimum
        net = init(net);
        net = train(net,data,targets);
        perf = perform(net,targets,net(data));
        if perf < best_perf
            best_perf = perf;
            trainedNet = net;
        end
    end
end

function acc_for_P = xVal(data,labels,targets,maxP)  % This function is for 10-fold cross validation purposes
    cv = cvpartition(length(data),'KFold',10);
    k_train_data = cell(1,10);
    k_train_labels = cell(1,10);
    k_train_targets = cell(1,10);
    k_test_data = cell(1,10);
    k_test_labels = cell(1,10);
    k_test_targets = cell(1,10);
    for k = 1:10
       k_train_data{k} = data(:,training(cv,k));
       k_train_labels{k} = labels(training(cv,k));
       k_train_targets{k} = targets(:,training(cv,k));
       k_test_data{k} = data(:,test(cv,k));
       k_test_labels{k} = labels(test(cv,k));
       k_test_targets{k} = targets(:,test(cv,k));
    end
    
    acc_for_P = zeros(1,maxP);    
    for P = 1:maxP
        err_for_fold = zeros(1,10);        
        parfor k = 1:10           
            net = patternnet(P,'traingdx','mse');
            net.trainParam.epochs = 2000;
            net.divideFcn = 'dividetrain';
            best_perf = Inf;
            pred = zeros(1,numel(k_test_labels{k}));
            for i = 1:10 % We only reinitialize 10 times here as cross-validation already takes a long time to complete
                net = init(net);
                net = train(net,k_train_data{k},k_train_targets{k});
                perf = perform(net,k_test_targets{k},net(k_test_data{k}));
                if perf < best_perf
                    best_perf = perf;
                    [~,pred] = max(net(k_test_data{k}));
                end
            end
            err_for_fold(k) = sum(pred ~= k_test_labels{k})/numel(k_test_labels{k});
        end
        fprintf("10-fold cross validation with %d samples: %3.2f%% complete\n", length(data), 100*(P/maxP));
        acc_for_P(P) = 1 - mean(err_for_fold);
    end
end

function visualizer(data,labels,fig) % This function visualizes data
    set(0, 'CurrentFigure', fig);
    ax = gca;
    ax.FontSize = 16;
    X = data(1, labels==1);
    Y = data(2, labels==1);
    Z = data(3, labels==1);
    scatter3(X,Y,Z,'b.');
    hold on;
    X = data(1, labels==2);
    Y = data(2, labels==2);
    Z = data(3, labels==2);
    scatter3(X,Y,Z,'r.');
    X = data(1, labels==3);
    Y = data(2, labels==3);
    Z = data(3, labels==3);
    scatter3(X,Y,Z,'k.');
    X = data(1, labels==4);
    Y = data(2, labels==4);
    Z = data(3, labels==4);
    scatter3(X,Y,Z,'g.');
    set(gca, 'FontSize', 14);
    xlim([-6 6]);
    ylim([-6 6]);
    zlim([-6 6]);
    xlabel('$x_1$','Interpreter','latex','FontSize',16);
    ylabel('$x_2$','Interpreter','latex','FontSize',16);
    zlabel('$x_3$','Interpreter','latex','FontSize',16);
    legend('Class 1','Class 2','Class 3', 'Class 4','FontSize',16,'Location','northeast');
end