import pandas as pd
import numpy as np
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

train_pivot = train_df.pivot_table(index="user_id", columns="item_id", values="rating")
dim_rows, dim_columns = train_pivot.shape
k = 10
U_matrix = np.random.rand(dim_rows, k) # each row represents one user (943 rows)
V_matrix = np.random.rand(dim_columns, k) # each row represents a movie (1613 rows)

movie_to_index = {} # creating a mapping between movie_id and V row index

for index, movie_id in enumerate(train_pivot.columns):
    movie_to_index[movie_id] = index


learning_rate = 0.01
def epoch(U_matrix, V_matrix, train_df, learning_rate):
    for row in train_df.itertuples():
        i, j = (row.user_id, row.item_id)

        u_i = U_matrix[i-1, :]
        v_j = V_matrix[movie_to_index[j], :]

        r_ij = row.rating
        error_ij = r_ij - np.dot(u_i, v_j)

        U_matrix[i-1, :] = u_i + learning_rate * error_ij * v_j
        V_matrix[movie_to_index[j], :] = v_j + learning_rate * error_ij * u_i
    return (U_matrix, V_matrix)
