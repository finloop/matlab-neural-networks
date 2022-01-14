min_leaf_size = 4;
max_num_split = 30;
split = 'deviance'; prune = 'off';
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
 C_Tree{k} = fitctree(X_Train,Y_Train,'SplitCriterion',split,'Prune',prune, ...
                        'MinLeafSize', min_leaf_size, ...
                       'MaxNumSplits', max_num_split);
 Label = predict(C_Tree{k}, X_Test);
 % Dokładnosc (CV) dla CART
 Accuracy_CV(k) = sum(Label == Y_Test)/length(Y_Test);
end
Avr_Accuracy = mean(Accuracy_CV);
[max,ind_max] = max(Accuracy_CV);
ind = ind_max(1);
% Podanie reguł i narysowanie struktury
view(C_Tree{ind});
view(C_Tree{ind}, 'mode', 'graph');