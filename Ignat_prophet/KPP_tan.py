# def KPP(test):
#     SumTan = 0
#     a = []
#     counter = 0
#     for i in range(len(test['Freight_Price'])-1):
#         SumTan+= abs((test['yhat_exp'][i+1]-test['yhat_exp'][i]) - (test['Freight_Price'][i+1] - test['Freight_Price'][i]))
#         a.append(abs((test['yhat_exp'][i+1]-test['yhat_exp'][i]) - (test['Freight_Price'][i+1] - test['Freight_Price'][i])))
#         counter+=1
#     return SumTan/counter,min(a),max(a)
def KPP(test,pred):
    SumTan = 0
    a = []
    counter = 0
    for i in range(len(test)-1):
        SumTan+= abs((pred[i+1]-pred[i]) - (test[i+1] - test[i]))
        a.append( abs((pred[i+1]-pred[i]) - (test[i+1] - test[i])))
        counter+=1
    return SumTan/counter,min(a),max(a)