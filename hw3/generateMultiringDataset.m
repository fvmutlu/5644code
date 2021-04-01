function [data,labels] = generateMultiringDataset(numberOfClasses,numberOfSamples)

C = numberOfClasses;
N = numberOfSamples;
% Generates N samples from C ring-shaped 
% class-conditional pdfs with equal priors

% Randomly determine class labels for each sample
thr = linspace(0,1,C+1); % split [0,1] into C equal length intervals
u = rand(1,N); % generate N samples uniformly random in [0,1]
labels = zeros(1,N);
for l = 1:C
    ind_l = find(thr(l)<u & u<=thr(l+1));
    labels(ind_l) = repmat(l,1,length(ind_l));
end

a = [1:C].^2.5; b = repmat(1.7,1,C); % parameters of the Gamma pdf needed later
% Generate data from appropriate rings
% radius is drawn from Gamma(a,b), angle is uniform in [0,2pi]
angle = 2*pi*rand(1,N);
radius = zeros(1,N); % reserve space
for l = 1:C
    ind_l = find(labels==l);
    radius(ind_l) = gamrnd(a(l),b(l),1,length(ind_l));
end

data = [radius.*cos(angle);radius.*sin(angle)];

if 1
    colors = rand(C,3);
    figure;
    %figure(1), clf,  % Editing this line so that figures for both data sets show up
    for l = 1:C
        ind_l = find(labels==l);
        plot(data(1,ind_l),data(2,ind_l),'.','MarkerFaceColor',colors(l,:)); axis equal, hold on,
    end
end
