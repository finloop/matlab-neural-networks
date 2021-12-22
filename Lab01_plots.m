%%%%%%%%%%%%%%%%%%% Wykres wszystkie algorytmy %%%%%%%%%%%%%%%%%%%%%%%%%%%% 

clear
clc
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

hold off
i = 1;
cont = {};
for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        [M, I] = max(mean(ACC, 2))

        plot(10:10:160,mean(ACC, 2), 'LineWidth', 2);
        ylim([0.5 1]);
        cont{i} = tfun +"--" + afun;

        hold on
        i = i +1;
        clear("ACC")
    end
end

plot(I*10, M, '.', 'MarkerSize', 20)
cont{length(cont)+1}= "Punkt z największą precyzją";
title(["Wykres precyzji od ilości neuronów dla rożnych", ...
    "algorytmów uczących i funkcji aktywacji"])
xlabel("Ilość neuronów w warstwie ukrytej")
ylabel("Prezyzja")
legend(cont, 'Location', 'southwest')
saveas(gcf, 'img/precision-all.png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


