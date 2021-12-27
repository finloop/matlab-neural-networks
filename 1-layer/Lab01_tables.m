training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

temp = table();

for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        names = strings(16, 1);
        for n=10:10:160
            names(n/10) = (tfun + "-" + afun + "-") + string(n);
        end
        t = table(names, round(mean(ACC, 2), 3), round(mean(STD, 2), 4), round(mean(RECALL, 2), 3), round(mean(RECALL_STD, 2), 4));
        temp = [temp; t];
    end
end
temp.Properties.VariableNames = ["Network" "acc" "std(acc)" "recall" "std(recall)"];
writetable(sortrows(temp, [4 2 5 3], 'descend'), "comp.csv")
