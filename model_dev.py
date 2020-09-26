# import library
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# function to split dates and times
def split_date(list_inp, dt_splitter="/"):
    dts = []; mos = []; yrs = []
    hrs = []; mis = []
    for inp in list_inp:
        if len(inp) > 0:
            dates, time = inp.split()
            dt, mo, yr = dates.split(dt_splitter)
            tms = time.split(":")
            dts.append(dt); mos.append(mo); yrs.append(yr)
            hrs.append(tms[0]); mis.append(tms[1])
        else:
            dts.append(0); mos.append(0); yrs.append(0)
            hrs.append(0); mis.append(0)
    return dts, mos, yrs, hrs, mis


# In[46]:

def model_dev(ord, ord_prod):

    # load data
    order = pd.read_csv("./uploads/{}".format(ord))
    order_prod = pd.read_csv("./uploads/{}".format(ord_prod))

    order_col = order.columns.values.tolist()
    order_prod_col = order_prod.columns.values.tolist()

    imp_ord = SimpleImputer(strategy='most_frequent').fit(order)
    order = imp_ord.transform(order)
    order = pd.DataFrame(order, columns=order_col)

    imp_ord_prod = SimpleImputer(strategy='most_frequent').fit(order_prod)
    order_prod = imp_ord_prod.transform(order_prod)
    order_prod = pd.DataFrame(order_prod, columns=order_prod_col)

    # copy to new df
    order_feat = order.iloc[:,:]

    # remove column buyer_type due to homogen values
    order_feat.drop(["buyer_type"], axis=1, inplace=True)

    # define the column label as a list
    order_label = order.columns.to_list()
    order_prod_label = order_prod.columns.to_list()

    # lowercase
    for i in [2, 3]:
        order_prod["{}".format(order_prod_label[i])] = order_prod["{}".format(order_prod_label[i])].apply(lambda x: x.lower())

    # label encoder
    le_list = joblib.load("./pickle/le_list.p")
    for i in [3, 4, 6]:
        le = le_list["ord_{}".format(order_label[i])]
        order_feat["{}".format(order_label[i])] = le.transform(order_feat["{}".format(order_label[i])])

    # split the information of dates and times of book time
    time_param = ["dt", "mo", "yr", "hr", "mi"]
    tmp = split_date(order_feat["book_time"])
    for i in range(len(time_param)):
        order_feat["bt_ord_{}".format(time_param[i])] = tmp[i]
    # drop book_time columns
    order_feat.drop(['book_time'], axis=1, inplace=True)
    order_feat.head()

    # split the information of dates and times of last status time
    tmp = split_date(order_feat["last_status_time"])
    for i in range(len(time_param)):
        order_feat["lst_ord_{}".format(time_param[i])] = tmp[i]
    # drop last_status_time columns
    order_feat.drop(['last_status_time'], axis=1, inplace=True)
    order_feat.head()

    brand_list, packaging_list = joblib.load("./pickle/bp_list.p")
    for lst in [brand_list, packaging_list]:
        for item in lst:
            order_feat[item] = 0

    # set id as index
    order_feat.set_index("id", inplace=True)
    # add new column
    order_feat["pack_am_sum"] = 0
    order_feat["pack_am_avg"] = 0
    order_feat["am_sum"] = 0
    order_feat["am_avg"] = 0
    order_feat["price_sum"] = 0
    order_feat["price_avg"] = 0
    order_feat["ord_prod_ls"] = ""
    order_feat["ord_prod_bt"] = ""
    order_feat["ord_prod_lst"] = ""

    # input data from data_product
    ids_ = order_feat.index.values.tolist()
    for id_ in ids_:
        tmp_df = order_prod[order_prod["id"] == id_]
        if tmp_df.shape[0] > 0:
            # brand
            brand_lst = tmp_df["brand"].values.tolist()
            brand_unq = list(set(brand_lst))
            for br_unq in brand_unq:
                if br_unq in brand_list:
                    order_feat.loc[id_, br_unq] = brand_lst.count(br_unq)
            # packaging
            pack_lst = tmp_df["packaging"].values.tolist()
            pack_unq = list(set(pack_lst))
            for pc_unq in pack_unq:
                if pc_unq in packaging_list:
                    order_feat.loc[id_, pc_unq] = pack_lst.count(pc_unq)
            # packaging_amount
            pack_am_lst = tmp_df["packaging_amount"].values.tolist()
            order_feat.loc[id_, "pack_am_sum"] = np.sum(pack_am_lst)
            order_feat.loc[id_, "pack_am_avg"] = np.average(pack_am_lst)
            # amount
            am_lst = tmp_df["amount"].values.tolist()
            order_feat.loc[id_, "am_sum"] = np.sum(am_lst)
            order_feat.loc[id_, "am_avg"] = np.average(am_lst)
            # amount
            pr_lst = tmp_df["price"].values.tolist()
            order_feat.loc[id_, "price_sum"] = np.sum(pr_lst)
            order_feat.loc[id_, "price_avg"] = np.average(pr_lst)
            # last_status
            ls_lst = tmp_df["last_status"].values.tolist()
            order_feat.loc[id_, "ord_prod_ls"] = ls_lst[0]
            # book_time
            bt_lst = tmp_df["book_time"].values.tolist()
            order_feat.loc[id_, "ord_prod_bt"] = bt_lst[0]
            # last_status_time
            lst_lst = tmp_df["last_status_time"].values.tolist()
            order_feat.loc[id_, "ord_prod_lst"] = lst_lst[0]

    # label encoder
    le = le_list["ord_prod_ls"]
    order_feat["ord_prod_ls"] = le.transform(order_feat["ord_prod_ls"])

    # split the information of dates and times of ord_prod book time
    tmp = split_date(order_feat["ord_prod_bt"], dt_splitter="-")
    for i in range(len(time_param)):
        order_feat["bt_ord_prod_{}".format(time_param[i])] = tmp[i]
    order_feat.drop(['ord_prod_bt'], axis=1, inplace=True)

    # split the information of dates and times of ord_prod last_status_time
    tmp = split_date(order_feat["ord_prod_lst"], dt_splitter="-")
    for i in range(len(time_param)):
        order_feat["lst_ord_prod_{}".format(time_param[i])] = tmp[i]
    order_feat.drop(['ord_prod_lst'], axis=1, inplace=True)

    # reset index
    order_feat.reset_index(inplace=True)
    # order_feat.to_csv("./uploads/test_set.csv", index=None)

    # supervised
    X_test = order_feat.iloc[:,1:]

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

    # unsupervised
    X_test = order_feat.iloc[:,1:]

    order = pd.read_csv("./uploads/{}".format(ord))
    order_col = order.columns.values.tolist()
    imp_ord = SimpleImputer(strategy='most_frequent').fit(order)
    order = imp_ord.transform(order)
    order = pd.DataFrame(order, columns=order_col)

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
