
load OISdata.mat
TT = length(times);

errorsLOOCV      = zeros(TT,1);
predictionsLOOCV = cell(TT,1);

for i = 1:TT
    skipIndex = i;              % define it in workspace
    run('Kalman_fast.m');  
    
    
    % Extract only the first 27 elements of predictedPrice (if relevant to actualPrice)
    predictedPrice = fAll(:, i); 
    actualPrice    = priceAll{i};
    predictedPriceSubset = predictedPrice(1:length(actualPrice));

    errVal = mean(abs(predictedPriceSubset - actualPrice), "all");
     
    errorsLOOCV(i)      = errVal;
    predictionsLOOCV{i} = predictedPrice;
    
    clearvars -except i TT times priceAll errorsLOOCV predictionsLOOCV
end

meanError = mean(errorsLOOCV);
disp(['Mean LOOCV error = ', num2str(meanError)]);
