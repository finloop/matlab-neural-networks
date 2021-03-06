clear;
load('OptDigits5');
X = OptDigits5(:,1:end-1);
Y = OptDigits5(:,end)+1;

L = length(Y);
classes = unique(Y);
nclasses = length(classes);
CLASS_ACC = zeros(nclasses, 1);
CLASS_STD = zeros(nclasses, 1);
CLASS_PREC = zeros(nclasses, 1);
CLASS_RECALL = zeros(nclasses, 1);
CLASS_F1 = zeros(nclasses, 1);

standardize = false;

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
        
        SVM{k, nclass} = fitcsvm(X_Train, Y_Train, 'Standardize', standardize, ...
                        'KernelFunction',"polynomial", ...
                        'PolynomialOrder', 2, ...
                        'Solver', "SMO", ...
                        'BoxConstraint', 0.1);

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

res = table((1:10)',round(CLASS_ACC, 4), round(CLASS_STD, 4), round(CLASS_PREC, 4), round(CLASS_RECALL, 4),round(CLASS_F1, 4));
fin = table(0,round(mean(CLASS_ACC),4),round(mean(CLASS_STD),4),round(mean(CLASS_PREC),4),round(mean(CLASS_RECALL),4),round(mean(CLASS_F1),4));
res = [res; fin];
res.Properties.VariableNames = ["Klasa" "acc" "std(acc)" "prec" "recall" "F1"];
writetable(res, "svm-" + string(standardize) + ".csv")



%errorbar(CLASS_ACC, CLASS_STD, 'LineWidth', 2)
%title(["Wykres dokładności wraz z odchyleniem" "z walidacji krzyżowej dla standardize=" + string(standardize)])
%ylim([0.5 1])
%xlabel("Klasa")
%ylabel("Dokładność")
%saveas(gcf, 'img/acc-best.png')
%close(gcf);

%plot(CLASS_PREC, 'LineWidth', 2)
%title(["Wykres precyzji" "dla standardize=" + string(standardize)])
%ylim([0.5 1])
%xlabel("Klasa")
%ylabel("Precyzja")
%saveas(gcf, 'img/prec-best.png')
%close(gcf);
