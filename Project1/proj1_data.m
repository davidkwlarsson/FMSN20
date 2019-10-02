%set paths to course files and/or download swissRainfall.mat from the homepage

%load data
load swissRainfall.mat

% extract covariates and reshape to images to columns
swissGrid = [swissElevation(:) swissX(:) swissY(:)];
%remove points outside of Switzerland
swissGrid = swissGrid( ~isnan(swissGrid(:,1)),:);

%plot observations
figure(1)
subplot(221)
scatter(swissRain(:,3), swissRain(:,4), 20, swissRain(:,1), 'filled')
hold on
plot(swissBorder(:,1), swissBorder(:,2),'k')
axis xy tight; hold off; colorbar
title('Precip. (mm)')

%plot elevation (prediction surface + at observations sites)
subplot(222)
imagesc([0 max(swissX(:))], [0 max(swissY(:))], swissElevation, ...
        'alphadata', ~isnan(swissElevation))
hold on
plot(swissBorder(:,1), swissBorder(:,2),'k')
scatter(swissRain(:,3), swissRain(:,4), 20, swissRain(:,2), ...
        'filled','markeredgecolor','r')
%replace markeredgecolor with markeredge on older version of matlab.
axis xy tight; hold off; colorbar
title('Elevation (km)')

%plot X and Y coordinates
subplot(223)
imagesc([0 max(swissX(:))], [0 max(swissY(:))], swissX, ...
        'alphadata', ~isnan(swissX))
hold on
plot(swissBorder(:,1), swissBorder(:,2),'k')
scatter(swissRain(:,3), swissRain(:,4), 20, swissRain(:,3), ...
        'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('X-dist (km)')
subplot(224)
imagesc([0 max(swissX(:))], [0 max(swissY(:))], swissY, ...
        'alphadata', ~isnan(swissX))
hold on
plot(swissBorder(:,1), swissBorder(:,2),'k')
scatter(swissRain(:,3), swissRain(:,4), 20, swissRain(:,4), ...
        'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('Y-dist (km)')

%Extract observations and split data into obsrvation and validation set
Y = swissRain(swissRain(:,5)==0,:);
Yvalid = swissRain(swissRain(:,5)==1,:);

%transform data
Y(:,1) = sqrt(Y(:,1));
Yvalid(:,1) = sqrt(Yvalid(:,1));

%perform a linear regression on elevation
X = [ones(size(Y,1),1) Y(:,2)];
beta = X\Y(:,1);

%extract covariates (elevation) for the prediction locations
Xgrid = [ones(size(swissGrid,1),1) swissGrid(:,1)];

%recall that swissGrid contains all relevant locations (i.e. not nan values)
%we need to recreate the full matrix
mu = nan( size(swissElevation) );
%and place the predictions at the correct locations
mu( ~isnan(swissElevation) ) = Xgrid*beta;

%and plot the result
figure(2)
imagesc([0 max(swissX(:))], [0 max(swissY(:))], mu, ...
        'alphadata', ~isnan(mu))
hold on
plot(swissBorder(:,1), swissBorder(:,2),'k')
scatter(Y(:,3), Y(:,4), 20, Y(:,1), 'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('sqrt of rainfall and predictions')