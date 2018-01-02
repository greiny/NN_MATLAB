clc; clear all; close all;

 %% Porting data
 img_mat = []; 
% ratio = 28/30;
% % correct data
% for i=1:536
%   img = imread(sprintf('data/correct/correct (%d).jpg',i));
%   img = imresize(img,ratio);
%   A = img(:);
%   A = vertcat(A,[1]);
%   A = reshape(A,1,[]);
%   img_mat = vertcat(img_mat,A);
% end
% % incorrect data
% for j=1:76
%   img = imread(sprintf('data/incorrect/incorrect (%d).jpg',j));
%   img = imresize(img,ratio);
%   A = img(:);
%   A = vertcat(A,[0]);
%   A = reshape(A,1,[]);
%   img_mat = vertcat(img_mat,A);
% end
% csvwrite('imgdata.csv', img_mat);

%% Data Reconstruction 
img_mat = csvread('imgdata_norm.csv');
shuffled_mat = img_mat(randperm(size(img_mat,1)),:);
csvwrite('imgdata_shuffled.csv', shuffled_mat);

%%
nInput = 25*25;
nTarget = 1;
[m,n] = size(img_mat);
input=shuffled_mat(:,[1:n-nTarget])';
target=shuffled_mat(:,[(n-nTarget+1):n])'; 

%%
for ii=1:1:100
    net=newff(input,target,[30],{'logsig'},'trainlm');
    net.trainParam.epochs=1000;                   
    net.trainParam.lr=0.01;       
    net.trainParam.goal=0.0000001;
    net.trainParam.mc=0.9;                                         

    net.divideParam.trainRatio = 60/100; 
    net.divideParam.valRatio = 30/100; 
    net.divideParam.testRatio = 10/100; 
    
    [net,tr]= train(net,input,target); 
%   [net,tr]= train(net,input,target,'useGPU','yes');   
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
