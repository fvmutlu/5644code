mu_1 = [0 0 0]';
sigma_1 = [1 0.3 -0.2; 0.3 2 0.5; -0.2 0.5 1];
pdf_1 = @(x)(mvnpdf(x, mu_1, sigma_1)); % Class conditional pdf for class 1

mu_2 = [-2 1 2]';
sigma_2 = [2 0.4 0; 0.4 2 -0.7; 0 -0.7 1];
pdf_2 = @(x)(mvnpdf(x, mu_2, sigma_2)); % Class conditional pdf for class 1

mu_3a = [-1 1 -1]';
sigma_3a = [1 0.5 0.3; 0.5 1 -0.4; 0.3 -0.4 3];
mu_3b = [1 -2 2]';
sigma_3b = [3 -0.5 0.25; -0.5 1 0.4; 0.25 0.4 1];
pdf_3 = @(x)(0.5*mvnpdf(x, mu_3a, sigma_3a) + 0.5*mvnpdf(x, mu_3b, sigma_3b)); % Class conditional pdf for class 1

priors = [0.3 0.3 0.4];
N = 1e4;
labels = discreteSample(priors,N); % Generating 10K labels
dataset = zeros(3,N); % Generating random 10K samples in accordance with the labels
for i = 1:N
    switch labels(i)
        case 1
            dataset(:,i) = mvnrnd(mu_1,sigma_1);
        case 2
            dataset(:,i) = mvnrnd(mu_2,sigma_2);
        case 3
            if rand() < 0.5
                dataset(:,i) = mvnrnd(mu_3a,sigma_3a);
            else
                dataset(:,i) = mvnrnd(mu_3b,sigma_3b);
            end
        otherwise
            disp("Sample label error, label number out of bounds.")
    end
end

%load('q2data.mat') % Uncommenting this line will let you use the same data I used in generating results for my submission

figure;
ax = gca;
ax.FontSize = 16;
X = dataset(1, labels==1);
Y = dataset(2, labels==1);
Z = dataset(3, labels==1);
scatter3(X,Y,Z,'b','filled');
hold on;
X = dataset(1, labels==2);
Y = dataset(2, labels==2);
Z = dataset(3, labels==2);
scatter3(X,Y,Z,'g','filled');
X = dataset(1, labels==3);
Y = dataset(2, labels==3);
Z = dataset(3, labels==3);
scatter3(X,Y,Z,'r','filled');
set(gca, 'FontSize', 14);
title('Dataset visualization','FontWeight','bold','FontSize',16);
legend('Class 1','Class 2','Class 3','FontSize',16);

decisions = zeros(1,N);
R_star = 0;
for k = 1:N
    X = dataset(:,k);
    Px = pdf_1(X)*priors(1) + pdf_2(X)*priors(2) + pdf_3(X)*priors(3);
    [maxp,ind] = max([pdf_1(X)*priors(1) pdf_2(X)*priors(2) pdf_3(X)*priors(3)]);
    %R_star = R_star + (1 - maxp/Px);
    R_star = R_star + (Px - maxp);
    switch ind
        case 1
            decisions(k) = 1;
        case 2
            decisions(k) = 2;
        case 3
            decisions(k) = 3;
        otherwise
            disp("Decision label error, label number out of bounds.");
    end
end

% Confusion matrix
conf_mat = [sum(decisions(labels==1)==1) sum(decisions(labels==2)==1) sum(decisions(labels==3)==1); ...
               sum(decisions(labels==1)==2) sum(decisions(labels==2)==2) sum(decisions(labels==3)==2);
               sum(decisions(labels==1)==3) sum(decisions(labels==2)==3) sum(decisions(labels==3)==3)] ./ [sum(labels==1) sum(labels==2) sum(labels==3)];           
disp(conf_mat);

% Scatter plot 0-1 loss
figure;
X = dataset(1, labels==1 & decisions==1);
Y = dataset(2, labels==1 & decisions==1);
Z = dataset(3, labels==1 & decisions==1);
scatter3(X,Y,Z,'go');
hold on;

X = dataset(1, labels==1 & ~(decisions==1));
Y = dataset(2, labels==1 & ~(decisions==1));
Z = dataset(3, labels==1 & ~(decisions==1));
scatter3(X,Y,Z,'ro');

X = dataset(1, labels==2 & (decisions==2));
Y = dataset(2, labels==2 & (decisions==2));
Z = dataset(3, labels==2 & (decisions==2));
scatter3(X,Y,Z,'gx');

X = dataset(1, labels==2 & ~(decisions==2));
Y = dataset(2, labels==2 & ~(decisions==2));
Z = dataset(3, labels==2 & ~(decisions==2));
scatter3(X,Y,Z,'rx');

X = dataset(1, labels==3 & decisions==3);
Y = dataset(2, labels==3 & decisions==3);
Z = dataset(3, labels==3 & decisions==3);
scatter3(X,Y,Z,'gd');

X = dataset(1, labels==3 & ~(decisions==3));
Y = dataset(2, labels==3 & ~(decisions==3));
Z = dataset(3, labels==3 & ~(decisions==3));
scatter3(X,Y,Z,'rd');
set(gca, 'FontSize', 14);
set(gca, 'FontSize', 14);
title('0-1 loss classification correctness','FontWeight','bold','FontSize',16);
legend('Class 1 (C)','Class 1 (I)','Class 2 (C)','Class 2 (I)','Class 3 (C)','Class 3 (I)','FontSize',16);

%% PART B

R10_1 = @(X)(pdf_2(X)*priors(2) + 10*pdf_3(X)*priors(3));
R10_2 = @(X)(pdf_1(X)*priors(1) + 10*pdf_3(X)*priors(3));
R10_3 = @(X)(pdf_1(X)*priors(1) + pdf_2(X)*priors(2));
R10_star = 0;

decisions_10 = zeros(1,N);
for k = 1:N
    X = dataset(:,k);
    Px = pdf_1(X)*priors(1) + pdf_2(X)*priors(2) + pdf_3(X)*priors(3);
    [risk,ind] = min([R10_1(X) R10_2(X) R10_3(X)]);
    %R10_star = R10_star + risk/Px;
    R10_star = R10_star + risk;
    switch ind
        case 1
            decisions_10(k) = 1;
        case 2
            decisions_10(k) = 2;
        case 3
            decisions_10(k) = 3;
        otherwise
            disp("Decision label error, label number out of bounds.");
    end
end

% Confusion matrix
conf_mat_10 = [sum(decisions_10(labels==1)==1) sum(decisions_10(labels==2)==1) sum(decisions_10(labels==3)==1); ...
               sum(decisions_10(labels==1)==2) sum(decisions_10(labels==2)==2) sum(decisions_10(labels==3)==2);
               sum(decisions_10(labels==1)==3) sum(decisions_10(labels==2)==3) sum(decisions_10(labels==3)==3)] ./ [sum(labels==1) sum(labels==2) sum(labels==3)];           
disp(conf_mat_10);

% Scatter plot \Lambda_10
figure;
X = dataset(1, labels==1 & decisions_10==1);
Y = dataset(2, labels==1 & decisions_10==1);
Z = dataset(3, labels==1 & decisions_10==1);
scatter3(X,Y,Z,'go');
hold on;

X = dataset(1, labels==1 & ~(decisions_10==1));
Y = dataset(2, labels==1 & ~(decisions_10==1));
Z = dataset(3, labels==1 & ~(decisions_10==1));
scatter3(X,Y,Z,'ro');

X = dataset(1, labels==2 & (decisions_10==2));
Y = dataset(2, labels==2 & (decisions_10==2));
Z = dataset(3, labels==2 & (decisions_10==2));
scatter3(X,Y,Z,'gx');

X = dataset(1, labels==2 & ~(decisions_10==2));
Y = dataset(2, labels==2 & ~(decisions_10==2));
Z = dataset(3, labels==2 & ~(decisions_10==2));
scatter3(X,Y,Z,'rx');

X = dataset(1, labels==3 & decisions_10==3);
Y = dataset(2, labels==3 & decisions_10==3);
Z = dataset(3, labels==3 & decisions_10==3);
scatter3(X,Y,Z,'gd');

X = dataset(1, labels==3 & ~(decisions_10==3));
Y = dataset(2, labels==3 & ~(decisions_10==3));
Z = dataset(3, labels==3 & ~(decisions_10==3));
scatter3(X,Y,Z,'rd');
set(gca, 'FontSize', 14);
title('$\Lambda_{10}$ loss function classification correctness','FontWeight','bold','FontSize',16,'Interpreter','latex');
legend('Class 1 (C)','Class 1 (I)','Class 2 (C)','Class 2 (I)','Class 3 (C)','Class 3 (I)','FontSize',16);

R100_1 = @(X)(pdf_2(X)*priors(2) + 100*pdf_3(X)*priors(3));
R100_2 = @(X)(pdf_1(X)*priors(1) + 100*pdf_3(X)*priors(3));
R100_3 = @(X)(pdf_1(X)*priors(1) + pdf_2(X)*priors(2));

decisions_100 = zeros(1,N);
R100_star = 0;
for k = 1:N
    X = dataset(:,k);
    Px = pdf_1(X)*priors(1) + pdf_2(X)*priors(2) + pdf_3(X)*priors(3);
    [risk,ind] = min([R100_1(X) R100_2(X) R100_3(X)]);
    %R100_star = R100_star + risk/Px;
    R100_star = R100_star + risk;
    switch ind
        case 1
            decisions_100(k) = 1;
        case 2
            decisions_100(k) = 2;
        case 3
            decisions_100(k) = 3;
        otherwise
            disp("Decision label error, label number out of bounds.");
    end
end

conf_mat_100 = [sum(decisions_100(labels==1)==1) sum(decisions_100(labels==2)==1) sum(decisions_100(labels==3)==1); ...
               sum(decisions_100(labels==1)==2) sum(decisions_100(labels==2)==2) sum(decisions_100(labels==3)==2);
               sum(decisions_100(labels==1)==3) sum(decisions_100(labels==2)==3) sum(decisions_100(labels==3)==3)] ./ [sum(labels==1) sum(labels==2) sum(labels==3)];
disp(conf_mat_100);

% Scatter plot \Lambda_100
figure;
X = dataset(1, labels==1 & decisions_100==1);
Y = dataset(2, labels==1 & decisions_100==1);
Z = dataset(3, labels==1 & decisions_100==1);
scatter3(X,Y,Z,'go');
hold on;

X = dataset(1, labels==1 & ~(decisions_100==1));
Y = dataset(2, labels==1 & ~(decisions_100==1));
Z = dataset(3, labels==1 & ~(decisions_100==1));
scatter3(X,Y,Z,'ro');

X = dataset(1, labels==2 & (decisions_100==2));
Y = dataset(2, labels==2 & (decisions_100==2));
Z = dataset(3, labels==2 & (decisions_100==2));
scatter3(X,Y,Z,'gx');

X = dataset(1, labels==2 & ~(decisions_100==2));
Y = dataset(2, labels==2 & ~(decisions_100==2));
Z = dataset(3, labels==2 & ~(decisions_100==2));
scatter3(X,Y,Z,'rx');

X = dataset(1, labels==3 & decisions_100==3);
Y = dataset(2, labels==3 & decisions_100==3);
Z = dataset(3, labels==3 & decisions_100==3);
scatter3(X,Y,Z,'gd');

X = dataset(1, labels==3 & ~(decisions_100==3));
Y = dataset(2, labels==3 & ~(decisions_100==3));
Z = dataset(3, labels==3 & ~(decisions_100==3));
scatter3(X,Y,Z,'rd');
set(gca, 'FontSize', 14);
title('$\Lambda_{100}$ loss function classification correctness','FontWeight','bold','FontSize',16,'Interpreter','latex');
legend('Class 1 (C)','Class 1 (I)','Class 2 (C)','Class 2 (I)','Class 3 (C)','Class 3 (I)','FontSize',16);