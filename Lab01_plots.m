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
title(["Wykres dokładności od ilości neuronów dla rożnych", ...
    "algorytmów uczących i funkcji aktywacji"])
xlabel("Ilość neuronów w warstwie ukrytej")
ylabel("Prezyzja")
legend(cont, 'Location', 'southwest')
saveas(gcf, 'img/precision-all.png')
close(gcf)

%%%%%%%%%%%%% Wykresy dokładności per klasa dla algorytmów %%%%%%%%%%%%%%%%%%%
clear
clc
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

i = 1;
cont = {};
lab = string(17);

for j=[0:16]
    lab(j+1) = num2str(j*20);
end

for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        plot(ACC, 'LineWidth', 2);
        xticklabels(lab);
        title(["Precyzja dla każdej z klas dla kombinacji", ...
            (tfun +"--"+ afun)])
        ylim([0.5 1]);
        for j=[1:12]
            cont{j} = num2str(j);
        end
        
        leg = legend(cont,'Location', 'southeast');
        title(leg, "Klasy");
        leg.Title.Visible = 'on';
        xlabel("Ilość neuronów w warstwie ukrytej")
        ylabel("Prezyzja")
        i = i +1;
        clear("ACC")
        saveas(gcf, "img/precision-per-class-" + tfun +"-"+ afun + ".png")
        close(gcf) 
    end
end



