clc; clear all; close all;

%% Porting data
data_mat = []; 
data_mat = csvread('imgdata_norm.csv');
shuffled_mat = data_mat(randperm(size(data_mat,1)),:);
csvwrite('imgdata_shuffled.csv', shuffled_mat);

%% Data Reconstruction 
nInput = 25*25;
nTarget = 1;
[m,n] = size(shuffled_mat);
input=shuffled_mat(:,[1:n-nTarget])';
target=shuffled_mat(:,[(n-nTarget+1):n])'; 

%% Neural Network
for ii=1:1:100
    %Setting
    net=newff(input,target,[30],{'logsig'},'trainlm','learngdm','mse');
    net.trainParam.epochs=1000;                   
    net.trainParam.lr=0.01;       
    net.trainParam.goal=0.0000001;
    net.trainParam.mc=0.9;                                         

    net.divideParam.trainRatio = 60/100; 
    net.divideParam.valRatio = 30/100; 
    net.divideParam.testRatio = 10/100; 

    % Training
    [net,tr]= train(net,input,target); 
%   [net,tr]= train(net,input,target,'useGPU','yes');

    % Validation Accuracy Check
    [rows,cols] = size(input);
    sample_input = zeros(rows,length(tr.valInd)); % nInput x n
    sample_target = zeros(1,length(tr.valInd)); % nTarget x n
    for i=1:1:length(tr.valInd);
        sample_input(:,i) = input(:,tr.valInd(i));
        sample_target(:,i) = target(1,tr.valInd(i));
    end
    
    output = zeros(1,length(tr.valInd));
    for j=1:1:length(tr.valInd);              
        output(:,j) = net(sample_input(:,j));
    end

    error=abs(output-sample_target);
    accuracy=(1-error)*100;
    data(1,ii) = mean(accuracy);

    figure(2)
    x = linspace(1,ii,ii);
    plot(x,data,'-ob');
    set(gca,'FontSize',10,'FontWeight','Bold','FontName','Times New Roman');
    xlabel('Iteration No');
    title('Validation Accuracy(%)');
    if (mean(accuracy)>98)  
       break
    end
end
