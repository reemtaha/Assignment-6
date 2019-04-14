close all
clear
clc

%PCA to find the min features
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T = read(ds);
Data=T{:,4:21};
[m n]=size(Data);

correlation_x = corr(Data);
covariance_x = cov(Data);

K = 0;
Alpha=0.01;
lamda=0.001;


% Normalisation
for w=1:n
    if max(abs(Data(:,w)))~=0
        Data(:,w)=(Data(:,w)-mean((Data(:,w))))./std(Data(:,w));
        
    end
end


[eigen_vector S V] =  svd(covariance_x); %returns eigen values
% eigen_values= diag(S)';

%Calculate the K whic gives me the alpha<0.001
alpha=0.5;
while (alpha>=0.001)
    K=K+1;
    lamda1(K,:)=sum(max(S(:,1:K)));
    lamda2=sum(max(S));
    alpha=1-lamda1./lamda2;
end

reduced_data=eigen_vector(:, 1:K)'*(Data)';  % K=2
data_approx=eigen_vector(:,1:K)*reduced_data;
error_func=(1/m)*(sum(data_approx-Data').^2);


%Linear Regression
h=1;
Theta=zeros(n,1);
k1=1;
Y=T{:,3}/mean(T{:,3});
E(k1)=(1/(2*m))*sum((data_approx'*Theta-Y).^2); %cost function

while h==1
    Alpha=Alpha*1;
    Theta=Theta-(Alpha/m)*data_approx*(data_approx'*Theta-Y);
    k1=k1+1;
    E(k1)=(1/(2*m))*sum((data_approx'*Theta-Y).^2);
    
    %Regularization
    Reg(k1)=(1/(2*m))*sum((data_approx'*Theta-Y).^2)+(lamda/(2*m))*sum(Theta.^2);
    %
    if E(k1-1)-E(k1)<0;
        break
    end
    q=(E(k1-1)-E(k1))./E(k1-1);
    if q <.000001
        h=0;
    end
end


%K-Means Plotting

data=T{:,4:21};
[m n]=size(data);

k=18;
numP = m; % number of points
xMax = 16; % x between 0 and xMax
yMax = 16; % y between 0 and yMax
fprintf('k-Means will run with %d clusters and %d data points.\n',k,numP);

xP = (data(:,8))';
yP = (data(:,9))';
points = [xP; yP];


[cluster, centr] = kMeans(k, points); % my k-means



% visualize the clustering

scatter(xP,yP,200,cluster,'.');
hold on;
scatter(centr(1,:),centr(2,:),'xk','LineWidth',1.5);
axis([0 xMax 0 yMax]);
daspect([1 1 1]);
xlabel('x');
ylabel('y');
title('K-Mean Clustering');
grid on;
figure;


%K-Mean Error

max_iterations = 10;
cost = zeros(1,10);

for K=1:max_iterations
    
    
for i=1:max_iterations
    for j=1:18
   Intial_centroids(i,j)=rand;
    end
end


for i=1:max_iterations
 
indices=getClosestCentroids(Data, Intial_centroids);

 [centroids,error]=computeccentroids(Data, indices, K);
 Intial_centroids = centroids; 
 
end

costVec(K)=error;
end


plot(costVec)
axis([1,max_iterations, 1, max_iterations])
[a b]=min(costVec(2:10));     %b is the best number of clusters




%K-Mean Error of the reduced data
% 
% max_iterations1 = 10;
% cost1 = zeros(1,10);
% 
% for K1=1:max_iterations1
%     
%     
% for i1=1:max_iterations1
%     for j1=1:18
%    Intial_centroids1(i1,j1)=rand;
%     end
% end
% 
% 
% for i1=1:max_iterations1
%  
% indices1=getClosestCentroids(reduced_data, Intial_centroids1);
% 
%  [centroids1,error1]=computeccentroids(reduced_data1, indices1, K1);
%  Intial_centroids1 = centroids1; 
%  
% end
% 
% costVec1(K1)=error1;
% end
% 
% plot(costVec1)
% axis([1,max_iterations1, 1, max_iterations1])
% [a1 b1]=min(costVec1(2:10));     %b is the best number of clusters
% 
% 
