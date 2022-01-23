load("OptDigits5.mat")

X = OptDigits5(:,1:end-1); 
Y = OptDigits5(:,end);
% Parametry CART
prune = 'off';
max_num_splits = [2:2:30];
split_criterions = ["deviance" "gdi"];
min_leaf_sizes = [2 4];
REP = 50;

RESULTS = table();

for max_num_split = max_num_splits
    for split_criterion = split_criterions
        for min_leaf_size = min_leaf_sizes
            REP_ACC = zeros(length(REP),1);
            for i = 1:REP
                % Wygenerowanie indeksów do CV
                indeksy_cv = crossvalind('Kfold', Y, 10);
                Accuracy_CV = zeros(10,1);
                for k = 1 : 10
                   % Indeksy do walidacji i uczenia
                   cv_test_ind = (indeksy_cv == k);
                   cv_train_ind = ~cv_test_ind; 
                   % Rekordy do walidacji i uczenia
                   X_Test = X(cv_test_ind,:); Y_Test = Y(cv_test_ind);
                   X_Train = X(cv_train_ind,:); Y_Train = Y(cv_train_ind);
                   % Uczenie i walidacja CART:
                   C_Tree{k} = fitctree(X_Train,Y_Train, ...
                       'SplitCriterion', split_criterion, ...
                       'Prune', prune, ...
                       'MinLeafSize', min_leaf_size, ...
                       'MaxNumSplits', max_num_split);
                   Label = predict(C_Tree{k}, X_Test);
                   % Dokładnosc (CV) dla CART
                   Accuracy_CV(k) = sum(Label == Y_Test)/length(Y_Test);
                end
                REP_ACC(i) = mean(Accuracy_CV);
                %[max,ind_max] = max(Accuracy_CV);
                %ind = ind_max(1);
            end % end_rep
            % SAVE DATA TO TABLE
            label = split_criterion + "-" +  string(min_leaf_size);
            RESULTS = [RESULTS; cell2table({max_num_split, label, round(mean(REP_ACC),4)})];

        end % end min_leaf_size
    end % end split_criterions
end % end max_num_split

save("RES.mat")
% Wykresy:
% X: ManNumSplits, Y: Acc, dla kryteriów: [gdi, deviance] oran
% [MinLeafSize(2 lub 4)]

sp = split(RESULTS.Var2, "-");
criterion = array2table(sp(:, 1));
criterion.Properties.VariableNames = {'SplitCriterion'};
min_leaf_size = array2table(str2double(sp(:, 2)));
min_leaf_size.Properties.VariableNames = {'MinLeafSize'};
data = [RESULTS(:, [1 3]) criterion min_leaf_size];
data.Properties.VariableNames = {'MaxNumSplits' 'Acc' 'SplitCriterion' 'MinLeafSize'};

stackedplot(data(data.SplitCriterion == 'gdi', :))
title('Wpływ parametrów dla SplitCriterion gdi')
saveas(gcf, 'img/gdi.png')
close(gcf)

stackedplot(data(data.SplitCriterion == 'deviance', :))
title('Wpływ parametrów dla SplitCriterion deviance')
saveas(gcf, 'img/deviance.png')
close(gcf)