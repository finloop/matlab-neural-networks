clear
clc
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

L1_NEURONS = [20 40 60 120];
L2_NEURONS = [20 40 60 80];

NET_SIZES = zeros(length(L1_NEURONS)*length(L2_NEURONS), 2);

for L1=1:length(L1_NEURONS)
    for L2=1:length(L2_NEURONS)
        NET_SIZES((L1-1)*length(L1_NEURONS)+L2, :) = [L1_NEURONS(L1) L2_NEURONS(L2)];
    end
end

temp = table();

for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        names = strings(16, 1);
        for n=1:16
            names(n) = (tfun + "-" + afun + "-") + mat2str(NET_SIZES(n, :));
        end
        t = table(names, round(mean(ACC, 2), 3), round(mean(STD, 2), 4), round(mean(RECALL, 2), 3), round(mean(RECALL_STD, 2), 4));
        temp = [temp; t];
    end
end
temp.Properties.VariableNames = ["Network" "acc" "std(acc)" "recall" "std(recall)"];
writetable(sortrows(temp, [4 2 5 3], 'descend'), "comp.csv")
