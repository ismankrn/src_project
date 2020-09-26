import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

def pred_uns(ord):
    test_set = pd.read_csv("./uploads/test_set.csv")

    order = pd.read_csv("./uploads/{}".format(ord))
    order_col = order.columns.values.tolist()
    imp_ord = SimpleImputer(strategy='most_frequent').fit(order)
    order = imp_ord.transform(order)
    order = pd.DataFrame(order, columns=order_col)

    X_test = test_set.iloc[:,1:]

    selector = joblib.load("./pickle/uns_selector.p")
    X_test = selector.transform(X_test)

    scaler = joblib.load("./pickle/uns_scaler.p")
    X_test = scaler.transform(X_test)

    pca = joblib.load("./pickle/pca.p")
    X_test_pca = pca.transform(X_test)

    kmeans = joblib.load("./pickle/kmeans.p")
    X_label = kmeans.predict(X_test_pca)

    order["anomaly_label"] = X_label

    order.to_csv("./outputs/unsupervised_output.csv", index=None)
