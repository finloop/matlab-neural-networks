clear
clc
load('OptDigits5');
X = OptDigits5(:,1:end-1);
Y = OptDigits5(:,end)+1;
K = 10;
step = 10;

Y_C = full(ind2vec(Y'));

[rows, cols] = size(X);
N1 = 2 * cols + 16; % Zaokrąglić do najbliższej podzielnej przez 10 - 130
%N2 = N1 / 2;
nclasses = length(unique(Y));

% TODO
% Zrobić wykres (Acc, N1)
% Zrobić wykres (Acc, neurons)

% Funkcje do trenowania
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];


% Wygenerowanie indeksów do CV
% Wymagane jest zadanie wartości dla K.
iterations = N1/step + 2; % Od 10 do 190

for tfun = training_functions
    for afun = activation_functions
        ACC=zeros(iterations, nclasses); % Acc per step     
        STD=zeros(iterations, nclasses); % Std per step

        RECALL=zeros(iterations, nclasses); % Acc per step     
        RECALL_STD=zeros(iterations, nclasses); % Std per step
        disp(['Starting job. Iterations to do: ', num2str(length(ACC))])
        for neurons=1:iterations
            indeksy = crossvalind('Kfold', Y, K);
            Acc_CV = zeros(K, nclasses);
            Recall_CV = zeros(K, nclasses);
            disp(['Starting iteration : ', num2str(neurons),'/' , num2str(length(ACC))])
            for k = 1:K
                % Indeksy do walidacji
                test_ind = (indeksy == k);
                % Indeksy do uczenia
                train_ind = ~test_ind;

                % Rekordy do walidacji
                X_Test = X(test_ind,:)';
                Y_Test = Y_C(:,test_ind);
                % Rekordy do uczenia
                X_Train = X(train_ind,:)';
                Y_Train = Y_C(:,train_ind);

                % Projekt i uczenie MLP: +- 2 * cechy + 1
                net = feedforwardnet(neurons * step);
                net.trainParam.showWindow = false;
                net.layers{1}.transferFcn = afun;
                net.trainFcn = tfun;
                net.output.processFcns = {'mapminmax'};
                net.input.processFcns = {'mapminmax'};
                net = train(net, X_Train, Y_Train, 'useGPU', 'yes');

                % Walidacja sieci
                Y_Out = net(X_Test, 'useGPU', 'yes');
                Y_Out_Final = zeros(nclasses, length(Y_Out));
                % Dokładnośc sieci
                for j = 1:length(Y_Out)
                    [max_el, ind] = max(Y_Out(:,j));
                    Y_Out_Final(ind,j) = 1;
                end
                % TP / P
                % Czułość / Recall
                Recall_CV(k, :) = sum(Y_Out_Final .* Y_Test, 2) ./ sum(Y_Test, 2);

                % Dokładność / Accuracy
                Acc_CV(k, :) = sum(Y_Out_Final .* Y_Test, 2) ./ sum(Y_Out_Final, 2); 
            end
            ACC(neurons, :) = mean(Acc_CV, 2);
            STD(neurons, :) = std(Acc_CV, 0, 2);

            RECALL(neurons, :) = mean(Recall_CV, 2);
            RECALL_STD(neurons, :) = std(Recall_CV, 0, 2);
        end
        save(tfun +"_"+ afun + ".mat");
    end
end