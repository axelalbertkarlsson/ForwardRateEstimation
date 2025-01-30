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
C = cov(fH(:,1:nH));

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

%Initializing variables

% initierar variabler till lite värden endast för att testa.
x_p = ones(6,1)/10;
x_p(1) = 0.8;
x_u = zeros(27,1); % ändra denna så att den är max size av instrID (på något smart sätt - dvs minskar inte om vi inte får in någon observation). 
% Vi antar att man kommer skicka med x_s för föregående tidpunkt som då
% blir x_c_prior
x_c = zeros(size(C,2),1); %hehe
x_s = [x_p; x_c];
x = [x_s; x_u];


D = 1 * eye(27);
n_u = 27;
n_c = 0;

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

    %C = [ones(size(C,1),1) C];
    % en "basrad". 

    % ----------------------nytt

   CAll{k} = C;
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
    n_p = size(x_p,1);
    if k ~= 1
        %n_c = size(x_c,1);
            
        if n_c == n_c_prior
            I_x = eye(n_c);
        elseif n_c_prior > n_c
            I_x = eye(n_c);
            I_x = [ [ 0; zeros(n_c-1,1) ], I_x];
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

    

    %oIndAll{k} = oInd;
    %lastDate = mexPortfolio('lastDate', instrID);
 
    %firstDates(k) = firstDate;
    %tradeDates(k) = tradeDate;
    %lastDates(k) = lastDate;

end

% k är antalet iterationer
xAll = cell(length(times), 1);
wAll = cell(length(times), 1);
vAll = cell(length(times), 1);
A_overlineAll = cell(length(times), 1);
P_overlineAll = cell(length(times), 1);
a_overlineAll = cell(length(times), 1);
delta_wAll = cell(length(times), 1);
delta_vAll = cell(length(times), 1);
HAll = cell(length(times), 1);
gAll = cell(length(times), 1);



I_x = blkdiag(eye(n_p),eye(n_cAll{1}));
FAll{1} = blkdiag(I_x,D);

price_Err = cell(length(times), 1);
norm_Err = zeros(length(times), 1);
PAll = cell(length(times), 1);
PpredictAll = cell(length(times), 1);
PTAll = cell(length(times), 1);

mu_xAll = cell(length(times), 1);


%ansätt en lösning för ALLA tidsprioder


%load x_start3000.mat xAll
for t = 1:length(times(1:TT))
    %x_p = zeros(6,1);%ones(6,1)/100;
    %x_p(1) = eonioa{t}/Eeig(1,1)/30*0.9;
    x_p = Eeig' * fH(t,1:3661)';
    x_u = zeros(27,1); 
    test = n_cAll{t};
    x_c = zeros(n_cAll{t},1); %hehe
    %x_c(1) = eonioa{t};
    x_s = [x_p; x_c];
    %x = xAll{t};
    x = [x_s; x_u];

    %x = x + rand(length(x),1)/10 - 0.05;
    n_z = size(priceAll{t},1);
    v = rand(n_z,1)/10000;
    w = rand(size(x))/10000;


    xAll{t} = x;
    wAll{t} = w;
    vAll{t} = v;

    %gör en riktig skattning av P
    %x_s = x(1:end-27);
    tmp = diag(ones(size(x)))/100;
    tmp(1,1) = ef(1);
    tmp(2,2) = ef(2);
    tmp(3,3) = ef(3);
    tmp(4,4) = ef(4);
    tmp(5,5) = ef(5);
    tmp(6,6) = ef(6);
    %x_u = x(end-26:end);
    %tmp2 = diag(ones(size(x_u)))/100;

    PTAll{t} = tmp;
end

%OBS ska göra om ansättning av kovariansmatriser. Dessa ska skattas.
w= wAll{1};
v = vAll{1};
Qw = diag(ones(size(w)))/1000;
Rv = diag(ones(size(v)))/1000000;

x0 = xAll{1};


mean_price_Err = zeros(TT, 20);
c_sumAll = zeros(20,1);
for k = 1:20
    
    F = FAll{1};
    mu_xAll{1} = F*x0;
    for t = 2:length(times(1:TT))
        x = xAll{t-1};
        F = FAll{t};
        mu_x = F*x;

        %mu_x(end-26:end) = zeros(27,1);

        
        mu_xAll{t} = mu_x;
        % n = size(x,1);
        % 
        % if n == 57
        %     x_mean = [x_mean, x];
        % elseif n < 57
        %     tmp = [x; x(end+n-57+1:end)];
        %     x_mean = [x_mean, tmp];
        % else
        %     tmp = x(1:end-n+57);
        %     x_mean = [x_mean, tmp];
        % end
    end 
    %mu_x = mean(x_mean, 2);

    %EM skattning av R och Q
    cov_w = 0;
    cov_v = 0;
    missed_z = 0;
    for t = length(times(1:TT)):-1:1
        x = xAll{t};
        if t == 1
            x_prior = x0;
        else
            x_prior = xAll{t-1};
        end
        x_s = x(1:end-27);
        x_u = x(end-26:end);
        z = priceAll{t};
        I_z = I_zAll{t};
        grad_g = [];
        g = [];
        %d_post = 0;
        %q_post = 0;
        %Q_post = 0;
        o_inst = oAll{t};

        for j=1:length(o_inst)
            
            o_tmp = o_inst{j};
            dTtmp = year_fracAll{t};
            deltaT = dTtmp{j};
            
            o_x =  o_tmp* x_s;
            g = [g; (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT)];  
            grad_g = [grad_g ; (o_tmp(1) * exp(o_x(1)) - o_tmp(end)* exp(o_x(end))) / (exp(o_x(2:end))' * deltaT) - (exp(o_x(1)) - exp(o_x(end)) ) * exp(o_x(2:end))' * diag(deltaT) *o_tmp(2:end,:) / (exp(o_x(2:end))' * deltaT)^2;];
        end
        H = [grad_g I_z];
        z_hat = g + I_z*x_u;
        


        if t~=TT
            x_post = xAll{t+1};
            F = FAll{t+1};
            tmp = (x_post - F * x)*(x_post - F * x)';
            n = size(tmp, 1);
            if n < 58
                tmp = blkdiag(tmp, tmp(end+n-58+1:end,end+n-58+1:end));
            else
                tmp = tmp(1:58,1:58);
            end
            cov_w = cov_w + tmp;
        end
        if size(z,1) == 27
            cov_v = cov_v + (z-z_hat)*(z-z_hat)';
        else 
            missed_z = missed_z + 1;
        end

        HAll{t} = H;
        gAll{t} = g;
        mean_price_Err(t,k) = mean(abs(g-priceAll{t}));

    end

    
    
    %-----------------------------------Här kan man skatta kovariansmatriser---------------------------
    
    % if k~=1
    %     %Rv = cov_v /(TT-missed_z) + eye(size(cov_v,1))/100000;
    %     %Qw = cov_w /(TT-1) + eye(size(cov_w,1))/100000;
    %     %Predict and update of covariance matrix
    %     P = PTAll{1};%diag(ones(size(x)))/100;
    %     for t = 1:length(times(1:TT))
    %         F = FAll{t};
    %         H = HAll{t};
    %         x = xAll{t};
    %         v = vAll{t};
    %         n = size(x,1);
    %         nQ = size(Qw,1);
    %         if n > nQ
    %             sigma_w = blkdiag(Qw, Qw(end-n+nQ+1:end,end-n+nQ+1:end));
    %         else
    %             sigma_w = Qw(1:n, 1:n);
    %         end
    %         nz = size(v,1);
    %         if nz > size(Rv,1)
    %             sigma_v = blkdiag(Rv, Rv(end-nz+27+1:end,end-nz+27+1:end));         
    %         else
    %             sigma_v = Rv(1:nz, 1:nz);
    %         end
    % 
    %         P_predict = F * P * F' + sigma_w;
    %         K = P_predict * H' / (H * P_predict * H' + sigma_v);
    %         P = (diag(ones(size(x))) - K * H) * P_predict;
    %         PAll{t} = P;
    %         PpredictAll{t} = P_predict;
    % 
    %     end
    % 
    %     %smoother pass
    % 
    %     PTAll{TT} = PAll{TT};
    %     for t = length(times(1:TT))-1:-1:1
    %         F = FAll{t+1};
    %         P = PAll{t};
    %         P_predict = PpredictAll{t+1};
    %         J = P * F' / P_predict;
    %         PTAll{t} = P + J * (PTAll{t+1} - PpredictAll{t+1}) * J';
    % 
    %     end
    % end

    Q_post = 0;
    q_post = 0;
    d_post = 0;
    
    %räknar ut parametrar som beror av kovariansmatriser
    c_sum = 0;
    for t = length(times(1:TT)):-1:1
        F = FAll{t};
        H = HAll{t};
        z = priceAll{t};
        g = gAll{t};
        x = xAll{t};
        I_z = I_zAll{t};
        if t == 1
            x_prior = x0;
        else
            x_prior = xAll{t-1};
        end
        x_s = x(1:end-27);
        x_u = x(end-26:end);
        v = vAll{t};
        w = wAll{t};
        
    
        mu_w = zeros(size(w));
        mu_v = zeros(size(v));
        P = PTAll{t};

        n = size(x,1);
        nz = size(v,1);
        nQ = size(Qw,1);
        if n > nQ
            %tmp = [mu_x; mu_x(end-n+57+1:end)];
            sigma_w = blkdiag(Qw, Qw(end-n+nQ+1:end,end-n+nQ+1:end));
        else
            %tmp = mu_x(1:n);
            sigma_w = Qw(1:n, 1:n); 
        end

        if nz > 27
            sigma_v = blkdiag(Rv, Rv(end-nz+27+1:end,end-nz+27+1:end));         
        else
            sigma_v = Rv(1:nz, 1:nz);
        end

        mu_x = mu_xAll{t};
        [c_w , a, A] = loglikelihood(w, mu_w, sigma_w, 'normal', 0);
        [c_v , b, B] = loglikelihood(v, mu_v, sigma_v, 'normal', 0);
        [c_x , e, E] = loglikelihood(x, mu_x, P, 'normal', 0);

        c = c_x + c_v + c_w;
        
        c_sum = c_sum + c;
        delta_v = v - z + g + I_z * x_u ; %alla vektorer har olika dimensioner?
        delta_w = F * x_prior + w - x;
        
        %eq 4.60
        c_overline = c - a'*delta_w + 1/2 * delta_w' * A * delta_w - b'*delta_v + 1/2 * delta_v' * B * delta_v + d_post;
        
        a_overline = a - H'  * b  + e - A * delta_w  + H' * B * delta_v + q_post;  

        r_overline = -F'*a + F' * A * delta_w;

        A_overline = A + E + H' * B * H + Q_post;

        R_overline = F' * A * F;

        P_overline = F' * A;

        d = c_overline - 1/2 * a_overline' * (A_overline \ a_overline);
        q = r_overline + P_overline * (A_overline \ a_overline);
        Q = R_overline - P_overline * (A_overline \ P_overline');

        delta_vAll{t} = delta_v;
        delta_wAll{t} = delta_w;
        
        P_overlineAll{t} = P_overline;
        A_overlineAll{t} = A_overline;
        a_overlineAll{t} = a_overline;





        Q_post = Q;
        q_post = q;
        d_post = d;
    end
    c_sumAll(k) = c_sum;
    
    P = PTAll{1};
    mu_x = mu_xAll{1};
   
    [c_x , e, E] = loglikelihood(x0, mu_x, P, 'normal', 0);
    A = E + Q;
    a = e + q;

    
    mean(mean_price_Err(:,k))

    x_prior_update = - A \ a;
    x0 = x0 +  x_prior_update;
    
    f_0 = c_x + e' * x_prior_update + x_prior_update' * E* x_prior_update + d + q' * x_prior_update + 1/2 * x_prior_update' * Q * x_prior_update;
    
    if k~=1
       if f_old - f_0 > 0
            k
            break
        end
    %f_old - f_0;
    end

    f_old = f_0;
    
    for t = 1:length(times(1:TT))
        A_overline = A_overlineAll{t};
        P_overline = P_overlineAll{t};
        a_overline = a_overlineAll{t};
        F = FAll{t};
        H = HAll{t};
        delta_w = delta_wAll{t};
        delta_v = delta_vAll{t};
        
        x_update = A_overline \ P_overline' * x_prior_update - A_overline \ a_overline;

        w_update = x_update - F * x_prior_update - delta_w;
        v_update = - H * x_update - delta_v;


        if size(xAll{t},1) ~= size(x_update,1)
            disp("fel på x")
        end
        if size(wAll{t},1) ~= size(w_update,1)
            disp("fel på w")
        end
        if size(vAll{t},1) ~= size(v_update,1)
            disp("fel på v")
        end
        x = xAll{t} +x_update;
        w = wAll{t} +w_update;
        v = vAll{t} +v_update;
        x_prior_update = x_update;
        xAll{t} = x;
        wAll{t} = w;
        vAll{t} = v;

    end

    

    
end

fAll = zeros(size(C,1)+1, TT);


for t=1:length(times(1:TT))
    C = CAll{t};
    n = size(C,2);
    C = [C;ones(1,n)];
    x = xAll{t};
    x_s = x(1:end-27);
    
    f = [Eeig C] * x_s;

    fAll(:,t) = f;




    

    %x_s(7) = x_s(7) + 0.01;

    %f = [Eeig C] * x_s;
    %figure(4)
    %plot(f);
    
    g = [];
    o_inst = oAll{t};
    for j=1:length(o_inst)
        o_inst = oAll{t};
        o_tmp = o_inst{j};
        dTtmp = year_fracAll{t};
        deltaT = dTtmp{j};
        
        o_x =  o_tmp* x_s;
        g = [g; (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT)];  
    end

    %x_p = ones(6,1)/10;
    %x_p(1) = 0.8;

    %f = Eeig * x_p;

    mean_price_Err(t,20) = mean(abs(g-priceAll{t}));
    

    

end
tmp1 = times(1:TT);
tmp2 = (1:size(fAll,1))/365;
tmp3 = fAll(1:end,:);


plot3DCurve(tmp1,tmp2 ,tmp3 );





  