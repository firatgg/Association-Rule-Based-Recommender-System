import pandas as pd
# noinspection PyUnresolvedReferences
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
df_ = pd.read_csv("C:/projects/pythonProject/data_sets/armut_data.csv")
df = df_.copy()
df.head()
df.info()
df.dtypes
df.shape

df["Service"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df["New_Date"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["New_Date"].dt.strftime('%Y-%m')
df["SepetID"] = df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)

# ServiceID represents a different service for each CategoryID.
# A new variable is created to represent these services by combining ServiceID and CategoryID with "_".
# The data set consists of the date and time the services are received, there is no basket definition (invoice, etc.).
# Association Rule In order to apply Learning, a basket (invoice etc.) definition must be created.
# Here, the basket definition is the services each customer receives monthly.
# For example; 9_4, 46_4 services received by the customer with id 7256 in the 8th month of 2017 refers to one basket;
# 9_4, 38_4 services received in the 10th month of 2017 refers to another basket.
# Baskets must be identified with a unique ID.
# For this, firstly a new date variable containing only year and month is created.
# UserID and the newly created date variable are concatenated with "_" and assigned to a new variable named ID.

df.isnull().sum()
df.describe().T
df.head()
df2 = df.groupby(['SepetID', 'Service'])['Service'].count().unstack().fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)
df2.head()


frequent_itemsets = apriori(df2,
                            min_support=0.01,
                            use_colnames=True)

rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.01)
sorted_rules = rules.sort_values("lift", ascending=False)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

# Use the arl_recommender function to recommend a service to a user who last received service 2_0.

arl_recommender(rules, '2_0', 4)