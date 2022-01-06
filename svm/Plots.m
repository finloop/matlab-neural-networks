load("svm.mat")

%RESULTS(contains(RESULTS.SVM, 'gaussian'),:)
a = array2table(split(RESULTS.SVM, '-'));

a.Var2 = str2double(array2table(split(RESULTS.SVM, '-')).Var2);
a.Var3 = log(str2double(array2table(split(RESULTS.SVM, '-')).Var3));
a.Properties.VariableNames = {'Kernel' 'sigma' 'C (log)'}

gauss = [a(contains(RESULTS.SVM, 'gaussian'), :)  RESULTS(contains(RESULTS.SVM, 'gaussian'), 2:6)];
stackedplot(gauss(:, [2 3 4 6]))
title(["Wyniki jądra gaussian w zależności od sigma i C (log)"])
saveas(gcf, 'img/gaussian.png')
close(gcf)

b = array2table(split(RESULTS.SVM, '-'));

b.Var2 = str2double(array2table(split(RESULTS.SVM, '-')).Var2);
b.Var3 = log(str2double(array2table(split(RESULTS.SVM, '-')).Var3));
b.Properties.VariableNames = {'Kernel' 'd' 'C (log)'};

poly = [b(contains(RESULTS.SVM, 'polynomial'), :)  RESULTS(contains(RESULTS.SVM, 'polynomial'), 2:6)];
stackedplot(poly(:, [2 3 4 6]))
title(["Wyniki jądra polynomial w zależności od d i C (log)"])
saveas(gcf, 'img/polynomial.png')
close(gcf)