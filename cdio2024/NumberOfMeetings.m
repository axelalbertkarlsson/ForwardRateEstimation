function c=NumberOfMeetings(cur_pol_met, k, period)
    c = 0; % number of policy meetings
    for i=1:size(cur_pol_met(k:k+period,1))
        if(cur_pol_met(i, 4) == 1)
            c = c + 1;
        end
    end
end
