import pandas as pd
import numpy as np
import math

df = pd.read_csv("../data/ml-100k/u.data", sep="\t", header=None)
df.columns = ["user_id", "item_id", "rating", "timestamp"]

pivoted = df.pivot_table(index="user_id", columns="item_id", values="rating")
sorted_df = df.sort_values(by=["user_id", "timestamp"])
grouped = sorted_df.groupby("user_id")

train_parts = []
validate_parts = []
test_parts = []

for user_id, user_data in grouped:
    number_of_ratings = len(user_data)
    cut_off_train = math.floor(0.7 * number_of_ratings)
    cut_off_validate = math.floor(0.8 * number_of_ratings)

    train_parts.append(user_data[:cut_off_train])
    validate_parts.append(user_data[cut_off_train:cut_off_validate])
    test_parts.append(user_data[cut_off_validate:])

train_df = pd.concat(train_parts)
validate_df = pd.concat(validate_parts)
test_df = pd.concat(test_parts)

train_pivot = train_df.pivot_table(index="user_id", columns="item_id", values="rating")
dim_rows, dim_columns = train_pivot.shape
k = 10
U_matrix = np.random.normal(0, 0.1, size=(dim_rows, k)) # each row represents one user (943 rows)
V_matrix = np.random.normal(0, 0.1, size=(dim_columns, k)) # each row represents a movie (1613 rows)

movie_to_index = {} # creating a mapping between movie_id and V row index (since missing some movies)
for index, movie_id in enumerate(train_pivot.columns):
    movie_to_index[movie_id] = index

global_average = train_df["rating"].mean()
bias_user = np.zeros((dim_rows,))
bias_movies = np.zeros((dim_columns,))

learning_rate = 0.01
def epoch(U_matrix, V_matrix, train_df, learning_rate, global_average, bias_user, bias_movies, regularisation_strength):
    for row in train_df.itertuples():
        i, j = (row.user_id, row.item_id) # i for user, j for movie

        # parameters
        u_i = U_matrix[i-1, :].copy()
        v_j = V_matrix[movie_to_index[j], :].copy()
        b_i = bias_user[i-1, ]
        b_j = bias_movies[movie_to_index[j], ]
        mu = global_average

        # error calculation
        r_ij = row.rating
        error_ij = r_ij - (mu + b_i + b_j + np.dot(u_i, v_j))

        # updating parameters
        U_matrix[i-1, :] = u_i + learning_rate * (error_ij * v_j - regularisation_strength * u_i)
        V_matrix[movie_to_index[j], :] = v_j + learning_rate * (error_ij * u_i - regularisation_strength * v_j)
        bias_user[i-1, ] = b_i + learning_rate * (error_ij - regularisation_strength * b_i)
        bias_movies[movie_to_index[j], ] = b_j + learning_rate * (error_ij - regularisation_strength * b_j)
    return (U_matrix, V_matrix, bias_user, bias_movies)

def RMSE(U, V, ratings, global_average, bias_user, bias_movies):
    squared_error_sum = 0
    for row in ratings.itertuples():
        i, j = (row.user_id, row.item_id)

        # parameters
        u_i = U[i-1, :]
        v_j = V[movie_to_index[j], :]
        b_i = bias_user[i-1, ]
        b_j = bias_movies[movie_to_index[j], ]

        # error calculation
        r_ij = row.rating
        error_ij = r_ij - (global_average + b_i + b_j + np.dot(u_i, v_j))
        squared_error_sum += error_ij**2
    return np.sqrt(1/len(ratings) * squared_error_sum)

test_df_filtered = test_df[test_df["item_id"].isin(movie_to_index)]
validate_df_filtered = validate_df[validate_df["item_id"].isin(movie_to_index)]
# for x in range(10):
#     print(f"Training: {RMSE(U_matrix, V_matrix, train_df, global_average, bias_user, bias_movies)}")
#     print(f"Testing: {RMSE(U_matrix, V_matrix, test_df_filtered, global_average, bias_user, bias_movies)}")
#     epoch(U_matrix, V_matrix, train_df, learning_rate, global_average, bias_user, bias_movies)

reg_strengths = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
results = []

for x in reg_strengths:
    np.random.seed(0)
    U_matrix = np.random.normal(0, 0.1, size=(dim_rows, k)) # each row represents one user (943 rows)
    V_matrix = np.random.normal(0, 0.1, size=(dim_columns, k)) # each row represents a movie (1613 rows)
    bias_user = np.zeros((dim_rows,))
    bias_movies = np.zeros((dim_columns,))

    best_RMSE = np.inf
    best_epoch = 0

    for i in range(80):
        epoch(U_matrix, V_matrix, train_df, learning_rate, global_average, bias_user, bias_movies, x)
        train_RMSE = RMSE(U_matrix, V_matrix, train_df, global_average, bias_user, bias_movies)
        validate_RMSE = RMSE(U_matrix, V_matrix, validate_df_filtered, global_average, bias_user, bias_movies)

        if(validate_RMSE < best_RMSE):
            best_RMSE = validate_RMSE
            best_epoch = i + 1

    results.append({
        "regularisation_strength": x,
        "best_RMSE": best_RMSE,
        "best_epoch": best_epoch
    })

results_df = pd.DataFrame(results)
print(results_df)
