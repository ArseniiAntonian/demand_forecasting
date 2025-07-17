# def KPP(test,pred):
#     HIT = 0
#     MISSED = 0
#     for i in range(len(test)-1):
#         if (test['yhat_exp'][i+1]-test['yhat_exp'][i]) * (test['Freight_Price'][i+1] - test['Freight_Price'][i])>=0:
#             HIT+=1
#         else:
#             MISSED+=1
#     return HIT/(HIT+MISSED)

# def KPP(test,pred):
#     HIT = 0
#     MISSED = 0
#     for i in range(len(test)-1):
#         if (pred[i+1]-pred[i]) * (test[i+1] - test[i])>0:
#             HIT+=1
#         else:
#             MISSED+=1
#     return HIT/(HIT+MISSED)
def KPP(test, pred):
    import numpy as np
    # Приведём обе серии к numpy-массивам, чтобы обращаться по позиции
    true = np.asarray(test)
    pred = np.asarray(pred)

    n = len(true) - 1
    count = 0
    for i in range(n):
        if (pred[i+1] - pred[i]) * (true[i+1] - true[i]) > 0:
            count += 1
    return count / n
