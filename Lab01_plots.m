clear
clc

% 
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

hold off
i = 1;
cont = {};
for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        
        plot(10:10:160,mean(ACC, 2));
        ylim([0 1]);
        cont{i} = "Train:" + tfun +" Aktyw:" + afun;

        hold on
        i = i +1;
        clear("ACC")
    end
end

title("Wykres precyzji od ilości neuronów dla rożnych algorytmów uczących i funkcji aktywacji")
xlabel("Ilość neuronów w warstwie ukrytej")
ylabel("Prezyzja")
legend(cont, 'Location', 'southwest')