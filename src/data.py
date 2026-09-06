import pandas as pd
import math

df = pd.read_csv("../data/ml-100k/u.data", sep="\t", header=None)
df.columns = ["user_id", "item_id", "rating", "timestamp"]

pivoted = df.pivot_table(index="user_id", columns="item_id", values="rating")
sorted_df = df.sort_values(by=["user_id", "timestamp"])
grouped = sorted_df.groupby("user_id")

train_parts = []
test_parts = []

for user_id, user_data in grouped:
    number_of_ratings = len(user_data)
    cut_off = math.floor(0.8 * number_of_ratings)

    train_parts.append(user_data[:cut_off])
    test_parts.append(user_data[cut_off:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)
