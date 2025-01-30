measurementPath = '.\measurement';

addpath(measurementPath)

p = 100;
currency = 'EUR'; 
mi = marketInfo(currency);

fileName = currency;

[pl, times, ric, assetID, assetType, assetTenor, currencies] = populatePortfolioWithData(fileName, mi.currency, mi.currencyTimeZone, mi.iborTimeZone);

indOIS = find(strcmp(mi.onTenor, assetTenor) &  pl.atOISG == assetType);
indIBORON = find(strcmp(mi.onTenor, assetTenor) & pl.atIBOR == assetType);

isHoliday = mexPortfolio('isHoliday', mi.onCal, floor(times), floor(times)); % Note that projected holidays may change over time (holidays are added and removed, hence the date when the calendar is defined is important)
times(isHoliday==1) = []; % Removes all holidays

if (currency == 'EUR')
  nY = 10; % Number of years the curves span
  times = times(times>= datenum(2005,08,15));

  load EUR100;
  outOfSampleStartDate = datenum(2020,1,1);

elseif (currency == 'SEK')
  nY = 10; % Number of years the curves span
  times = times(times>= datenum(2012,10,22)); % OIS 2-10Y start on this date
elseif (currency == 'USD')
  nY = 10; % Number of years the curves span
  times = times(times>= datenum(2012,03,30)); % OIS 3-10Y start on this date
end

firstDatesPCA = firstDates;
set(0, 'DefaultFigureVisible', 'off');
[fH, piH, lastDatesPCA, nIn, ef, Ef, fTotVar, ePi, EPi, piTotVar] = pcaIRS(times, fH, piH, firstDates, lastDates, outOfSampleStartDate);
set(0, 'DefaultFigureVisible', 'on'); % Restore visibility

nH = min(lastDatesPCA-firstDates);
C = cov(fH(:,1:nH));%-fH(:,1:nH-1));

nEigs = 6;

[V,D] = eigs(C, nEigs);
[ef,ind] = sort(diag(D),1, 'descend');
Ef = V(:,ind);

nD = nY*365+10;

indAll = [indIBORON ; indOIS];
ricAll = [ric(indIBORON) ric(indOIS)];
assetTypeAll = [assetType(indIBORON) ; assetType(indOIS)];


%declare size of variables later used
firstDates = ones(length(times), 1)*inf;
tradeDates = ones(length(times), 1)*inf;
lastDates = ones(length(times), 1)*inf;

indInstrAll = cell(length(times), 1);
priceAll = cell(length(times), 1);
atAll = cell(length(times), 1);
tcAll = cell(length(times), 1);
oAll = cell(length(times), 1);
oIndAll = cell(length(times), 1);
usedInstr = false(1, length(indAll));
maturityAll = cell(length(times), 1);

eonioa = cell(length(times), 1);

CAll = cell(length(times), 1);

I_zAll = cell(length(times), 1);
I_xAll = cell(length(times)-1, 1); %OBS T-1 pga används för att gå mellan states
n_cAll = cell(length(times), 1);
year_fracAll = cell(length(times), 1);
FAll = cell(length(times), 1); %definierat som det som tar oss TILL state t



%%%%%%%%%%%--- martin kod tillagdt

% Read the data into a table
data_policy_meetings = readtable('ECB_Interest_Decisions.xlsx');

% HÃ¤mta ner och bearbeta ECBs mÃ¶ten och rÃ¤ntefÃ¶rÃ¤ndringar
pol_met = readmatrix('ECB_Interest_Decisions.xlsx', 'DataRange', 'A3:D7811');
pol_met(:,1) = pol_met(:,1) + 693960;


D = 0.95 * eye(27);
n_u = 27;
n_c = 0;

C_prior = 0;

TT = length(times); % antalet tidpunkter som man vill köra

for k=1:length(times(1:TT)) 
    lastInterestDatesHvec = ones(1, length(indOIS))*inf;
    tradeDate = floor(times(k));

    % IBOR ON
    if (~isempty(indIBORON))
    % Retrieve data
    %     [timeData, data] = mexPortfolio('getValues', assetID(indIBORON), times(k), mi.iborTimeZone, 'LAST');
    
    % RFR rate is set at 08:00:00 the next day, ensure that previous value is used
    rfrTimeLimit = datenum(year(times(k)),month(times(k)),day(times(k)),7, 59, 59);
    [timeData, data] = mexPortfolio('getValues', assetID(indIBORON), rfrTimeLimit, mi.iborTimeZone, 'LAST');
    % fprintf('%s: %d\n', datestr(timeData), data);
    rfrTime = datenum(year(timeData), month(timeData), day(timeData));
    % fprintf('%s: %s -> %s\n', datestr(times(k)), datestr(rfrTimeLimit), datestr(rfrTime));
    % Create IBOR
    [iborOnID] = mexPortfolio('initFromGenericInterest', assetID(indIBORON), 1, data, rfrTime, 0);
    iborOnPrices = [data ; data ; Inf];
    iborOnMarketQuoteP = data;
    else
    iborOnPrices = zeros(3,0);
    iborOnID = [];
    iborOnMarketQuoteP = [];
    end

    oisID = zeros(size(indOIS));
    oisPrices = ones(3,length(indOIS))*Inf;
    oisMarketQuoteP = ones(1,length(indOIS))*Inf;

    for j=1:length(indOIS)
        % Retrieve data
        [timeData, data] = mexPortfolio('getValues', assetID(indOIS(j)), times(k), mi.currencyTimeZone, {'BID', 'ASK'});
        if (timeData < times(k)-1) % More than 24 hours old, do not use
          data = [NaN ; NaN];
        end
        maturityDate = mexPortfolio('maturityDate', assetID(indOIS(j)), tradeDate);
        if (maturityDate-tradeDate > nD) % Too long time to maturity, skip asset
          data = [NaN ; NaN];
        end
        oisPrices(1:2,j) = data;
        if (sum(isnan(data))==0)
          mid = mean(data);
          % Create OIS
          [oisID(j)] = mexPortfolio('initFromGenericInterest', assetID(indOIS(j)), 1, mid, tradeDate, 1);
        else
          mid = 0;
          oisID(j) = 1E19; % Removes asset in createInstrumentsClasses
        end
        oisMarketQuoteP(j) = mid;
    end
    

   eonioa{k} = iborOnPrices(1);
    
   instrID = [oisID];
   prices = [oisPrices];
   marketQuoteP = [ oisMarketQuoteP];
   ricInstr = [ric(indOIS)];
   indInstr = 1:length(instrID);

    

   conType = ones(size(instrID))*6; % Unique mid price
   [instr,removeAsset,flg] = createInstrumentsClasses(pl, tradeDate, ricInstr, instrID, prices, conType);

   instrID(removeAsset) = [];
   marketQuoteP(removeAsset) = [];
   indInstr(removeAsset) = [];
   indGeneric = indAll(~removeAsset);
  
   firstDate = tradeDate;
   for j=1:length(instrID)
     firstDate = min(firstDate, instr.data{j}.settlementDate);
   end
   indInstrAll{k} = indInstr;

   nExtrapolate = tradeDate-firstDate;
   E = [repmat(Ef(1,:),nExtrapolate, 1) ; Ef];
   intE = [zeros(1, size(E,2)) ; cumsum(E,1)]/365;
   %summerar ihop elemen i egenvektorerna, används i lilla o, se ekvation
   %(4.3) i rapporten

   Eeig = E;

   period = 365 * 2; % Vi kollar antal möten för 2 år framåt från dag k
   C =[];
   cur_pol_met = pol_met(k:end,:);
   num_meet = NumberOfMeetings(cur_pol_met, 1, period);
   index_C = 0;
   for j = 1:num_meet
        index = find(cur_pol_met(:,4),1,'first');
        cur_pol_met = cur_pol_met(index+1:end,:);
        C =[C [zeros(index+index_C-1, 1);ones(length(E)-index-index_C,1)]];
        index_C = index_C + index;
   end
   C = [repmat(C(1,:),nExtrapolate, 1) ; C];

    %-------------------------------nytt

    C = [ones(size(C,1),1) C];
    % en "basrad". 

    % ----------------------nytt

   
   intC = [zeros(1, size(C,2)) ; cumsum(C,1)]/365;
 
    o = [];
    tc = [];
    maturity = zeros(length(oisID),1);
    oInd = zeros(length(instrID)+1,1);
    oInd(1) = 1;

    g = [];
    grad_g =[];
    
     %hehe
    %x_s = [x_p; x_c];
    %x = [x_s; x_u];

    n_c_prior = n_c;
    n_c = size(C,2);
    n_p = size(ef,1);
    if k ~= 1
        %n_c = size(x_c,1);
            
        if n_c == n_c_prior
            I_x = eye(n_c);
        elseif n_c_prior > n_c
            I_x = eye(n_c);
            I_x = [ [ 1; zeros(n_c-1,1) ], I_x];
        elseif n_c_prior < n_c
            I_x = eye(n_c_prior);
            I_x =[I_x; [0, zeros(1,n_c_prior-1)]];
        end  
         I_x = blkdiag(eye(n_p),I_x);
         F = blkdiag(I_x,D);
    end
          
    
    %w = zeros(size(x));
    %v = zeros(size(x_u));
    I_z = zeros(length(instrID),n_u);
    % I_z ska bara vara kvadratisk om alla instrument är med

    %mu_w = zeros(size(w));
    %mu_v = zeros(size(v));
    %mu_x = zeros(size(x));

    %sigma_w = diag(ones(size(w))/20);
    %sigma_v = diag(ones(size(v))/20);
    %sigma_x = diag(ones(size(x))/20);
    
    %sigma är kovariansmatris, dessa ska skattas sen.


   
    o_inst = cell(length(instrID),1);
    year_frac = cell(length(instrID),1);
    for j=1:length(instrID)
        jj = instrID(j);
        usedInstr(jj) = true;
        T0 = instr.data{j}.settlementDate - firstDate + 1;
        maturity(j) = instr.data{j}.maturityDate;
        T = instr.data{j}.cfDatesFix - firstDate + 1;
        % hittar T0 och T dagen och använder dem som index.
        o_tmp = [-intE(T0,:), -intC(T0,:) ; -intE(T,:), -intC(T,:)];
        o_inst{j} = o_tmp;
        year_frac{j} = diff([T0;T])/365;
        %o_x = o_tmp * x_s; %multiplicerar o för aktuellt instrument med state variabler
      
        I_z(j, jj-instrID(1)+1) = 1;

        

        %tc = [tc ; NaN ; instr.data{j}.dtFix];

    end

    maturityAll{k} = maturity;
    priceAll{k} = marketQuoteP';
    %atAll{k} = assetTypeAll(indInstr);
    %tcAll{k} = tc;
    oAll{k} = o_inst;
    I_zAll{k} = I_z;
    if k ~=1 
        I_xAll{k} = I_x;
        FAll{k} = blkdiag(I_x, D);
    end
    n_cAll{k} = n_c;
    year_fracAll{k} = year_frac;

    if size(C_prior,2) ~= size(C,2)
        CAll{k} = C;
    else
        CAll{k} = NaN;
    end

    C_prior = C;
    

    %oIndAll{k} = oInd;
    %lastDate = mexPortfolio('lastDate', instrID);
 
    %firstDates(k) = firstDate;
    %tradeDates(k) = tradeDate;
    %lastDates(k) = lastDate;

end

yieldH = yieldH(:,2:28);
zH = zH(:,2:28);
fH = fH(:, 1:3661);

I_xAll{1} = I_xAll{2};

save OISdata fH zH yieldH CAll D Eeig ef eonioa FAll I_xAll I_zAll n_cAll oAll priceAll year_fracAll TT times indInstrAll