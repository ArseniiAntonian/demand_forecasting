def KPP(test):
    HIT = 0
    MISSED = 0
    for i in range(len(test['Freight_Price'])-1):
        if (test['yhat_exp'][i+1]-test['yhat_exp'][i]) * (test['Freight_Price'][i+1] - test['Freight_Price'][i])>=0:
            HIT+=1
        else:
            MISSED+=1
    return HIT/(HIT+MISSED)

