clear;close all;clc;

%Gerando pontos aleatórios
X = 0.01:0.01:10;
X = X';
Y1 = sin(2*X);
Y1 = Y1';
nValue = 0.1;
noise = nValue*randn(1, length(Y1)) - nValue/2;
Y2 = Y1 + noise;
Y2 = Y2';

trainSize = 800;
dados = [X Y2];
dadosRandom = dados(randperm(size(dados,1)),:);
train = dadosRandom(1:trainSize,1);
trainResult = dadosRandom(1:trainSize,2);
test = dadosRandom(trainSize+1:size(dados,1),1);
testResult = dadosRandom(trainSize+1:size(dados,1),2);

H = [];
bestI = [0, 100000];
for i = 1:250
    W_oculto = rand(i,1);
    %H = W_oculto * train';
    
    %H = H';
    U = W_oculto * train';
    U = U';
    H = 1 ./ (1 + exp(-U));
    bias = repmat(-1,length(train),1);
    H = [bias H];
    
    W_saida = pinv(H)*trainResult;
    
    %H2 = W_oculto * test';
    %H2 = H2';
   
    %
    U2 = W_oculto * test';
    U2 = U2';
    H2 = 1 ./ (1 + exp(-U2));
    bias2 = repmat(-1,length(test),1);
    H2 = [bias2 H2];
    
        
    Y_final = W_saida' * H2';
    Y_final = Y_final';
    e = Y_final - testResult;
    MSE = sqrt(sum(e.^2));
    if bestI(2) > MSE
        bestI = [i, MSE];
        bestY = Y_final;
    end
end
plot(test,bestY, 'r.'); hold on;
plot(test,testResult, 'b.');
