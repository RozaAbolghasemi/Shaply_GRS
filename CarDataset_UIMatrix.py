import numpy as np
import csv
import random

# Read pairwise data from the CSV file, skipping the header
pairwise_data = []
with open("./Data/CarDataset/prefs1.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        pairwise_data.append(list(map(int, row)))

# Extract unique user IDs and item IDs
user_ids = set(row[0] for row in pairwise_data)
item_ids = set(row[1] for row in pairwise_data) | set(row[2] for row in pairwise_data)

# Create a dictionary to map item IDs to column indices in the matrix
item_to_index = {item_id: index for index, item_id in enumerate(sorted(item_ids))}

# Initialize a dictionary to store user-item matrices
user_item_matrices = {}

# Iterate over all users to generate their user-item matrices
for user_id in user_ids:
    # Create the user-item matrix
    user_item_matrix = np.zeros((len(item_ids), len(item_ids)))

    # Populate the matrix based on the pairwise data for the current user
    for row in pairwise_data:
        current_user_id, item1_id, item2_id, is_control = row
        if current_user_id != user_id:
            continue  # Skip rows not corresponding to the current user
        item1_index = item_to_index[item1_id]
        item2_index = item_to_index[item2_id]
        if is_control == 0:
            user_item_matrix[item1_index, item2_index] = 1
            user_item_matrix[item2_index, item1_index] = 0
        elif is_control == 1:
            user_item_matrix[item1_index, item2_index] = 0
            user_item_matrix[item2_index, item1_index] = 1

    # Set diagonal values to 0.5
    np.fill_diagonal(user_item_matrix, 0.5)

    # Store the user-item matrix in the dictionary using the user ID as the key
    user_item_matrices[user_id] = user_item_matrix
    #print( user_item_matrices[user_id])

    # Initialize a dictionary to store user ratings
    user_ratings_real = {}

    # Calculate average ratings for each user
    for user_id, user_item_matrix in user_item_matrices.items():
        avg_ratings = np.mean(user_item_matrix, axis=1)
        user_ratings_real[f'User{user_id}'] = avg_ratings.tolist()

print(user_ratings_real)

# Shuffle the list of user IDs
shuffled_user_ids = list(user_ratings_real.keys())
random.shuffle(shuffled_user_ids)

# Define the sizes of each group
group_sizes = [3,4,5,6,7]  # Adjust this list to set the desired sizes for each group

# Initialize a list to store the groups
groups = []

# Divide the shuffled user IDs into groups with the specified sizes
start_index = 0
for size in group_sizes:
    group = shuffled_user_ids[start_index: start_index + size]
    groups.append(group)
    start_index += size
# Initialize dictionary to store grouped user ratings
user_ratings = {}
# Populate grouped user ratings
for i, group in enumerate(groups, start=1):
    user_ratings[f'Group{i}'] = {user: user_ratings_real[user] for user in group}

print(user_ratings)

# Calculate the mean for each item in each group
group_means = {}
for group, users in user_ratings.items():
    num_users = len(users)
    group_means[group] = np.mean(list(users.values()), axis=0)

# Convert the mean ratings to tuples
ground_truth_group_ratings = {group: tuple(mean_ratings) for group, mean_ratings in group_means.items()}
print('ground_truth_group_ratings=',ground_truth_group_ratings)

