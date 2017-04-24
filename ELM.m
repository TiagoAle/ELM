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
X = train;
Y2 = trainResult;
dataRight = [X Y2];

H = [];
bestFold = [0, 100000];
for q = 1:250
   
    W_oculto = rand(q,1);
     % Criando folds
        sizeFold = size(dataRight,1)/5;
        for i = 1 : 5
            fold(i,:,:) = dataRight( sizeFold*(i-1)+1 : sizeFold*i,:);
        end
        
        mediaFold = 0;
        
        for folds = 1 : 5
            removedFold = folds;
            
            % Usando folds escolhidos para treino
            H = [];
            Yfold = [];
            Xfold = [];
            xTestFold = [];
            yTestFold = [];
            HR = [];
            
            testFold = fold(removedFold,:,1);
            yTestFold = fold(removedFold,:,2);
            
            for j = 1 : size(fold,1)
                %fold removido não é adicionado aos vetores de treinamento
                if j ~= removedFold
                    Yfold = [Yfold fold(j,:,2)];
                    Xfold = [Xfold fold(j,:,1)];
                end
            end
        end
        
        U = W_oculto * Xfold;
        H = 1 ./ (1 + exp(-U));
        H = H';
        bias = repmat(-1,size(H(:,1)),1);
        H = [bias H];
        W_saida = pinv(H)*Yfold';

        U2 = W_oculto * testFold;
        H2 = 1 ./ (1 + exp(-U2));
        H2 = H2';
        bias2 = repmat(-1,size(H2(:,1)),1);
        H2 = [bias2 H2];
        
        
        Y_final = W_saida' * H2';
        Y_final = Y_final';
        e = Y_final - yTestFold';
        MSE = sqrt(sum(e.^2));
        if bestFold(2) > MSE
            bestFold = [q, MSE];
        end
end
W_test = rand(bestFold(1),1);

U = W_test * train';
H = 1 ./ (1 + exp(-U));
H = H';
bias = repmat(-1,size(H(:,1)),1);
H = [bias H];
W_saidaTest = pinv(H)*trainResult;

U2 = W_test * test';
H2 = 1 ./ (1 + exp(-U2));
H2 = H2';
bias2 = repmat(-1,size(H2(:,1)),1);
H2 = [bias2 H2];


Y_finalTest = W_saidaTest' * H2';
Y_finalTest = Y_finalTest';
e = Y_finalTest - testResult;
MSE = sqrt(sum(e.^2));

plot(test,Y_finalTest, 'r*'); 
hold on;
plot(test,testResult, 'b*');