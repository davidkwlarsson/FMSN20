%% Project 1 Ordinary Least Squares
%Run this section by section 
%Also run this before Kriging script
load swissRainfall.mat
%% Plot the covariates for the different parameters
scatter(swissRain(:,2), sqrt(swissRain(:,1)));
xlabel('elevation')
figure
scatter(swissRain(:,3), sqrt(swissRain(:,1)))
xlabel('X-distance')
figure
scatter(swissRain(:,4), sqrt(swissRain(:,1)))
xlabel('Y-distance')

%%
Y_valid = swissRain(swissRain(:,5) == 1, :);
Y = swissRain(swissRain(:,5) == 0, :);
Y(:,1) = sqrt(Y(:,1));
Y_valid(:,1) = sqrt(Y_valid(:,1));
X = [ones(size(Y,1),1) Y(:,3) Y(:,2).^2];
beta = X\Y(:,1);
%Ordinary Least Square:
Beta_hat = ((X'*X)\X')*Y(:,1);
e_res = Y(:,1) - X*Beta_hat;

%Compute covariance of e_res:
n = size(Y,1);
p = size(X,2);
sigmae_2 = (e_res'*e_res)/(n-p);
V_beta = sigmae_2*((X'*X)\eye(size(X,2))); %Variance of beta
V_mu = sum((X*V_beta).*X,2); %Confidence
V_y = sigmae_2 + V_mu; %Prediction
%% Validation test
X_test =[ones(size(Y_valid,1),1)  Y_valid(:,3) Y_valid(:,2).^2];
Y_test = X_test*Beta_hat;
plot(Y_test,'r')
hold on
plot(Y_valid(:,1),'b')
V_mu = sum((X_test*V_beta).*X_test,2);
%plot the prediction intervall for the validations
Y_upper = Y_test + 1.96*sqrt(V_mu + sigmae_2);
Y_lower = Y_test - 1.96*sqrt(V_mu + sigmae_2);
plot(Y_upper,'--')
plot(Y_lower,'--')
%% Actual prediction
hold on
% extract covariates and reshape to images to columns
swissGrid = [swissElevation(:) swissX(:) swissY(:)];
%remove points outside of Switzerland
swissGrid = swissGrid( ~isnan(swissGrid(:,1)),:);
Xgrid = [ones(size(swissGrid,1),1) swissGrid(:,2) swissGrid(:,1).^2];
V_mu0 = sum((Xgrid*V_beta).*Xgrid,2); %Confidence
mu = nan(size(swissElevation));
%and place the predictions at the correct locations
mu( ~isnan(swissElevation) ) = Xgrid*Beta_hat;
imagesc([0 max(swissX(:))], [0 max(swissY(:))], mu,'alphadata', ~isnan(mu))
plot(swissBorder(:,1), swissBorder(:,2),'k')
scatter(Y(:,3), Y(:,4), 20, Y(:,1), 'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('sqrt of rainfall and predictions')
%% Plot the STD
hold on
STD = nan(size(swissElevation));
STD( ~isnan(swissElevation) ) = sqrt(V_mu0 + sigmae_2);
imagesc([0 max(swissX(:))], [0 max(swissY(:))], STD,'alphadata', ~isnan(STD))
scatter(Y(:,3), Y(:,4), 'r')
axis xy tight; hold off; colorbar
caxis([0 0.65])
title('Standard deviation of the estimated rainfall using OLS')
