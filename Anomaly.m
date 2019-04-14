clc
clear all
close all

ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T = read(ds);
Data=T{:,4:21};
[m n]=size(Data);

data=normalize(Data);

mean_data=mean(data);
std_data=std(data);
pdf_data=zeros(1,18);

for i=1:18
    pdf_data(i)=normpdf(data(1,i), mean_data(i), std_data(i)); %Gaussian Distribution
end

if prod(pdf_data) >0.99   %Product of the pdf data if it is 1 then it is anomly and if it 0 it is not
    anomly=1;
end

if prod(pdf_data) <0.001
    anomly=0;
end


