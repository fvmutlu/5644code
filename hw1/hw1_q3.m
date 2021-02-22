%% WINE
winedata = readmatrix('hw1/winequality-white.csv')'; % Read data from file
winelabels = winedata(12,:); % Last row is the quality classification (label)
winedata = winedata(1:11,:); % There are 11 features for this set
N_wine = size(winedata,2); % Size of the sample set
winepriors = arrayfun(@(i)(sum(winelabels==i)), 0:10) / N_wine; % Calculate priors from the sample set
winevalid = find(winepriors ~= 0) - 1; % There are quality labels for which there is no data, we need to eliminate them to avoid errors
numclasses_wine = length(winevalid);

% MLE
mu_wine = arrayfun(@(i){mean(winedata(:,winelabels==i),2)}, winevalid); % MLE mean is just the sample mean
sigma_wine = arrayfun(@(i){cov(winedata(:,winelabels==i)')}, winevalid); % Duda Eq. (22) for MLE covariance matrix (including Bessel's correction)
lambda_wine = arrayfun(@(i)((1e-18)*trace(sigma_wine{i})/rank(sigma_wine{i})), 1:numclasses_wine); % \lambda value to regularize covariance matrix since it is ill-conditioned
sigma_wine_reg = arrayfun(@(i){sigma_wine{i} + lambda_wine(i) * eye(size(winedata,1))}, 1:numclasses_wine); % C_{regularized}
pdf_wine = @(X)(arrayfun(@(i)(mvnpdf(X,mu_wine{i},sigma_wine_reg{i})), 1:numclasses_wine)); % Class conditional pdfs

winedecisions = zeros(1,N_wine);
for i = 1:N_wine
    X = winedata(:,i);
    [~,ind] = max(pdf_wine(X) .* winepriors(winevalid)); % Duda Eq. (20)
    winedecisions(i) = winevalid(ind);
end

disp(sum(winedecisions~=winelabels) / N_wine);

conf_wine = zeros(numclasses_wine,numclasses_wine); % Confusion matrix
for i = 1:numclasses_wine
    for j = 1:numclasses_wine
        conf_wine(i,j) = sum(winedecisions(winelabels==winevalid(j)) == winevalid(i)) / sum(winelabels==winevalid(j));
    end
end

disp(conf_wine);

% Visualization
figure;
[A,~,~,~,per,~] = pca(winedata');
A = A(:,1:3); % Extract first three components
for label = winevalid
    X = winedata(:,winelabels==label)';
    Y = X * A;
    scatter3(Y(:,1),Y(:,2),Y(:,3),'filled');
    hold on;
end
title('Wine dataset visualization via PCA','FontWeight','bold','FontSize',16);
xlabel(['Component 1 (' num2str(per(1),'%2.2f') '%)'],'FontWeight','bold','FontSize',16);
ylabel(['Component 2 (' num2str(per(2),'%2.2f') '%)'],'FontWeight','bold','FontSize',16);
zlabel(['Component 3 (' num2str(per(3),'%2.2f') '%)'],'FontWeight','bold','FontSize',16);
legend('Quality ' + string(winevalid),'FontSize',14);

%% Human Activity Recognition (HAR)
hardata = cat(1,readmatrix('hw1/X_train.txt'),readmatrix('hw1/X_test.txt'))'; % Training and test sample sets together add up to the 10299 samples referred to in the assignment document
harlabels = cat(1,readmatrix('hw1/y_train.txt'),readmatrix('hw1/y_test.txt'))'; % Labels are in separate files
N_har = size(hardata,2); % Size of the sample set
numclasses_har = 6;
harpriors = arrayfun(@(i)(sum(harlabels==i)), 1:numclasses_har) / N_har; % Estimate priors from the sample set (there are no missing labels from the data so we don't need a "harvalid" vector)

% MLE
mu_har = arrayfun(@(i){mean(hardata(:,harlabels==i),2)}, 1:numclasses_har);
sigma_har = arrayfun(@(i){cov(hardata(:,harlabels==i)')}, 1:numclasses_har);
lambda_har = arrayfun(@(i)((0.2)*trace(sigma_har{i})/rank(sigma_har{i})), 1:numclasses_har);
sigma_har_reg = arrayfun(@(i){sigma_har{i} + lambda_har(i) * eye(size(hardata,1))}, 1:numclasses_har);
pdf_har = @(X)(arrayfun(@(i)(mvnpdf(X,mu_har{i},sigma_har_reg{i})), 1:numclasses_har));

hardecisions = zeros(1,N_har);
parfor i = 1:N_har
    X = hardata(:,i);
    [~,ind] = max(pdf_har(X) .* harpriors);
    hardecisions(i) = ind;
end

display(sum(hardecisions~=harlabels) / N_har);

conf_har = zeros(numclasses_har,numclasses_har);

for i = 1:numclasses_har
    for j = 1:numclasses_har
        conf_har(i,j) = sum(hardecisions(harlabels==j) == i) / sum(harlabels==j);
    end
end

disp(conf_har);

% Visualization
figure;
[A,~,~,~,per,~] = pca(hardata');
A = A(:,1:3); % Extract first three components
per = per(1:3); % Extract first three components' percentages (of total variance explained)

for label = 1:numclasses_har
    X = hardata(:,harlabels==label)';
    Y = X * A;
    scatter3(Y(:,1),Y(:,2),Y(:,3),'filled');
    hold on;
end
title('HAR dataset visualization via PCA','FontWeight','bold','FontSize',16);
xlabel(['Component 1 (' num2str(per(1),'%2.2f') '%)'],'FontWeight','bold','FontSize',16);
ylabel(['Component 2 (' num2str(per(2),'%2.2f') '%)'],'FontWeight','bold','FontSize',16);
zlabel(['Component 3 (' num2str(per(3),'%2.2f') '%)'],'FontWeight','bold','FontSize',16);
legend({'Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying'},'FontSize',14);