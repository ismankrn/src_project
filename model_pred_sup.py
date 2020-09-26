import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import load_model

def pred_sup(ord):

    test_set = pd.read_csv("./uploads/test_set.csv")
    X_test = test_set.iloc[:,1:]

    order = pd.read_csv("./uploads/{}".format(ord))
    order_col = order.columns.values.tolist()
    imp_ord = SimpleImputer(strategy='most_frequent').fit(order)
    order = imp_ord.transform(order)
    order = pd.DataFrame(order, columns=order_col)

    selector = joblib.load("./pickle/sup_selector.p")
    X_test = selector.transform(X_test)

    scaler = joblib.load("./pickle/sup_scaler.p")
    X_test = scaler.transform(X_test)

    model = load_model('./pickle/my_model.h5')

    y_pred_test = model.predict_classes(X_test)

    order["anomaly_label"] = y_pred_test

    order.to_csv("./outputs/supervised_output.csv", index=None)


# In[ ]:
