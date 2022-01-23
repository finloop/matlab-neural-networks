clear;
load("OptDigits5.mat")
X = OptDigits5(:,1:end-1); 
Y = OptDigits5(:,end);

classes = unique(Y);
i = 1;
RES = table();
for class = classes'
    ind_1 = find(Y == class);
    X1 = X(ind_1,:);
    k1 = round(0.4 * length(ind_1)); % 40 % klas
    [k_1,Klastry_1] = kmeans(X1, k1, 'Distance', 'sqeuclidean');
    RES = [RES; array2table([Klastry_1 i*ones(k1, 1)])]; 
    i = i + 1;
end

XY = table2array(RES);
save('XYNEW.mat')
% Dla nowych danych odpalic najlepszy algorytm

X = XY(:,1:end-1);
Y = XY(:,end);

kernels = ["gaussian"];
sigmas = [0.5 5 20];
Cs = [0.1 10 1000];

RESULTS = table();

for kernel=kernels
    for sigma=sigmas
        for C=Cs
            RESULTS = [RESULTS; ...
                cell2table(mytrainsvm(X, Y, kernel, sigma, 'SMO', C, 2))];
        end
    end
end

kernels = ["polynomial"];
ds = [2 3];
Cs = [0.1 10 1000];

for kernel=kernels
    for d=ds
        for C=Cs
            RESULTS = [RESULTS; ...
                cell2table(mytrainsvm(X, Y, kernel, 2, 'SMO', C, d))];
        end
    end
end

RESULTS.Properties.VariableNames = ["SVM" "acc" "std(acc)" "precision" "recall" "F1 score"];
writetable(sortrows(RESULTS, [5 2 4 6], 'descend'), "res.csv")


function clf_score = mytrainsvm(X, Y, kernel, kernel_scale, solver, C, d) 
    L = length(Y);
    classes = unique(Y);
    nclasses = length(classes);
    CLASS_ACC = zeros(nclasses, 1);
    CLASS_STD = zeros(nclasses, 1);
    CLASS_PREC = zeros(nclasses, 1);
    CLASS_RECALL = zeros(nclasses, 1);
    CLASS_F1 = zeros(nclasses, 1);
    
    % Dla każdej z klas
    for nclass=1:nclasses
        % Pobieram dane wyłącznie należące do danej klasy
        Y_class = zeros(L, 1);
        class_ind = find(Y == nclass);
        Y_class(class_ind) = 1;
    
        
        % Wygenerowanie indeksów do CV
        indeksy_cv = crossvalind('Kfold', Y_class, 10);
        acc = zeros(10,1);
        prec = zeros(10, 1);
        recall = zeros(10, 1);
        f1 = zeros(10, 1);
    
        % Walidacja krzyżowa 10 razy
        for k = 1 : 10
            % Indeksy do walidacji i uczenia
            cv_test_ind = (indeksy_cv == k);
            cv_train_ind = ~cv_test_ind;
                % Rekordy do walidacji i uczenia
            X_Test = X(cv_test_ind,:); 
            Y_Test = Y_class(cv_test_ind, :);
            X_Train = X(cv_train_ind,:); 
            Y_Train = Y_class(cv_train_ind, :); % Uczenie i walidacja SVM:
            
            if kernel == "gaussian"
                SVM{k, nclass} = fitcsvm(X_Train, Y_Train, 'Standardize', true, ...
                                'KernelFunction',kernel, ...
                                'KernelScale', kernel_scale, ...
                                'Solver', solver, ...
                                'BoxConstraint', C);
            else
                SVM{k, nclass} = fitcsvm(X_Train, Y_Train, 'Standardize', true, ...
                                'KernelFunction',kernel, ...
                                'PolynomialOrder', d, ...
                                'Solver', solver, ...
                                'Solver', solver, ...
                                'BoxConstraint', C);
            end

            Label = predict(SVM{k, nclass}, X_Test);
            
            % Dokładnosc (CV) SVM
            acc(k) = sum(Label == Y_Test)/length(Y_Test);
            
            TP = sum(Label == 1);
            FP = sum(Label(Y_Test == 0) == 1);
            FN = sum(Label(Y_Test == 1) == 0);
            
            prec(k) = TP/(TP + FP); 
            if isnan(prec(k)) 
                prec(k) = 0;
            end
    
            recall(k) = TP/(TP + FN);
            if isnan(recall(k)) 
                recall(k) = 0;
            end
    
            f1(k) = 2 * prec(k)*recall(k)/( prec(k) + recall(k) );
            if isnan(f1(k)) 
                f1(k) = 0;
            end
        end
        CLASS_ACC(nclass, :) = mean(acc);
        CLASS_STD(nclass, :) = std(acc);
        CLASS_PREC(nclass, :) = mean(prec);
        CLASS_RECALL(nclass, :) = mean(recall);
        CLASS_F1(nclass, :) = mean(f1);
    end
    if kernel == "gaussian"
        clf_score = {kernel + "-" + string(kernel_scale) + '-' + string(C) ...
            mean(CLASS_ACC) ...
            mean(CLASS_STD) ...
            mean(CLASS_PREC) ...
            mean(CLASS_RECALL) ...
            mean(CLASS_F1) ...
        };
    else
        clf_score = {kernel + "-" + string(d) + '-' + string(C) ...
            mean(CLASS_ACC) ...
            mean(CLASS_STD) ...
            mean(CLASS_PREC) ...
            mean(CLASS_RECALL) ...
            mean(CLASS_F1) ...
        };
    end
    
end
