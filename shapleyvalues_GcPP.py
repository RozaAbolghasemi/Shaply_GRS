from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
#from randata import *
import time

# Function to select the top 2 subsets for each group based on Shapley values
def select_top_subsets(shapley_values, top_n=15):
    top_subsets = {}
    for group_name, values in shapley_values.items():
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        top_subsets[group_name] = sorted_values[:top_n]
    return top_subsets

def calculate_shapley_values(user_ratings, ground_truth_group_ratings, group_names):
    shapley_values = {}

    for group_name in group_names:
        #print(f"Group name: {group_name}")
        users = list(user_ratings[group_name].keys())
        all_subsets = generate_all_subsets(users)
        n = len(all_subsets)  # Number of subsets in F
        shapley_values[group_name] = {}

        for subset in all_subsets:
            shapley_values[group_name][subset] = calculate_shapley_value(
                subset, users, user_ratings[group_name], ground_truth_group_ratings, n, group_name
            )

    return shapley_values


def calculate_shapley_value(subset, users, user_ratings, ground_truth_ratings, n, group_name):
    shapley_value = 0
    #print("subset",subset)

    subset_users = set(subset)
    complement_users = set(users) - subset_users
    complement_subsets = generate_all_subsets(complement_users)
    l = len(complement_subsets)

    for complement_subset in complement_subsets:
        full_subset = subset_users.union(complement_subset)
        #print("complement_subset",complement_subset)
        #print("full_subset",full_subset)
        coalition_value = calculate_coalition_value(full_subset, users, user_ratings, ground_truth_ratings, group_name)
        complement_value = calculate_coalition_value(complement_subset, users, user_ratings, ground_truth_ratings,
                                                     group_name)
        shapley_value += coalition_value - complement_value
        shapley_value /= l
    return shapley_value


def calculate_coalition_value(subset, users, user_ratings, ground_truth_ratings, group_name):

    subset_ratings = {user: user_ratings[user] for user in subset}
    coalition_ratings = combine_user_ratings(subset_ratings)
    ground_truth_ratings = ground_truth_ratings[group_name]
    similarity = cosine_similarity([coalition_ratings], [ground_truth_ratings])
    #print("similarity",similarity)
    coalition_value = similarity[0][0]
    #print("coalition_value",coalition_value)

    return coalition_value


def combine_user_ratings(user_ratings):
    return np.mean(list(user_ratings.values()), axis=0)


def generate_all_subsets(users):
    all_subsets = []
    for i in range(1, len(users) + 1):
        all_subsets.extend(combinations(users, i))
    return all_subsets


group_names= ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
user_ratings = {'Group1': {'User42': [0.45, 0.25, 0.15, 0.15, 0.65, 0.85, 0.45, 0.85, 0.45, 0.75], 'User60': [0.65, 0.35, 0.15, 0.55, 0.85, 0.75, 0.15, 0.65, 0.15, 0.65]}, 'Group2': {'User13': [0.55, 0.45, 0.15, 0.15, 0.35, 0.75, 0.55, 0.95, 0.35, 0.75], 'User28': [0.75, 0.25, 0.35, 0.45, 0.95, 0.65, 0.25, 0.65, 0.15, 0.55], 'User48': [0.35, 0.75, 0.15, 0.25, 0.45, 0.75, 0.55, 0.85, 0.15, 0.55], 'User52': [0.15, 0.85, 0.25, 0.45, 0.15, 0.45, 0.75, 0.45, 0.85, 0.65]}, 'Group3': {'User18': [0.55, 0.55, 0.15, 0.15, 0.45, 0.75, 0.65, 0.95, 0.35, 0.45], 'User21': [0.55, 0.35, 0.75, 0.55, 0.55, 0.25, 0.25, 0.05, 0.85, 0.85], 'User22': [0.55, 0.45, 0.15, 0.15, 0.55, 0.65, 0.75, 0.95, 0.35, 0.45], 'User41': [0.35, 0.35, 0.25, 0.15, 0.45, 0.65, 0.85, 0.85, 0.65, 0.45], 'User43': [0.15, 0.95, 0.25, 0.45, 0.15, 0.65, 0.85, 0.65, 0.45, 0.45], 'User46': [0.15, 0.95, 0.55, 0.75, 0.25, 0.55, 0.85, 0.35, 0.05, 0.55], 'User58': [0.45, 0.35, 0.65, 0.35, 0.15, 0.35, 0.75, 0.45, 0.55, 0.85]}, 'Group4': {'User1': [0.35, 0.75, 0.35, 0.15, 0.35, 0.45, 0.95, 0.85, 0.65, 0.15], 'User7': [0.45, 0.55, 0.15, 0.15, 0.25, 0.75, 0.65, 0.95, 0.35, 0.75], 'User11': [0.45, 0.45, 0.65, 0.65, 0.55, 0.25, 0.25, 0.05, 0.85, 0.85], 'User12': [0.65, 0.45, 0.45, 0.65, 0.75, 0.45, 0.35, 0.45, 0.35, 0.45], 'User15': [0.15, 0.95, 0.35, 0.35, 0.15, 0.55, 0.85, 0.55, 0.65, 0.45], 'User17': [0.45, 0.65, 0.15, 0.15, 0.25, 0.75, 0.75, 0.95, 0.45, 0.45], 'User30': [0.45, 0.65, 0.45, 0.65, 0.35, 0.45, 0.45, 0.35, 0.45, 0.65], 'User31': [0.15, 0.95, 0.25, 0.65, 0.25, 0.65, 0.75, 0.45, 0.35, 0.55], 'User33': [0.55, 0.25, 0.25, 0.45, 0.85, 0.75, 0.05, 0.55, 0.35, 0.95], 'User36': [0.35, 0.55, 0.25, 0.15, 0.25, 0.85, 0.75, 0.85, 0.35, 0.65], 'User37': [0.55, 0.25, 0.75, 0.75, 0.65, 0.45, 0.05, 0.15, 0.75, 0.65], 'User45': [0.55, 0.25, 0.35, 0.45, 0.75, 0.65, 0.05, 0.45, 0.25, 0.85], 'User49': [0.45, 0.55, 0.85, 0.85, 0.75, 0.45, 0.25, 0.05, 0.35, 0.45], 'User50': [0.45, 0.45, 0.15, 0.15, 0.35, 0.65, 0.65, 0.75, 0.25, 0.65], 'User51': [0.15, 0.95, 0.15, 0.75, 0.45, 0.75, 0.55, 0.25, 0.25, 0.75]}, 'Group5': {'User3': [0.65, 0.15, 0.45, 0.45, 0.85, 0.45, 0.05, 0.25, 0.75, 0.95], 'User4': [0.35, 0.55, 0.15, 0.15, 0.35, 0.75, 0.75, 0.95, 0.35, 0.65], 'User5': [0.35, 0.65, 0.45, 0.65, 0.25, 0.55, 0.75, 0.25, 0.35, 0.75], 'User6': [0.35, 0.85, 0.25, 0.35, 0.35, 0.45, 0.95, 0.75, 0.45, 0.25], 'User8': [0.25, 0.85, 0.65, 0.45, 0.15, 0.45, 0.75, 0.25, 0.85, 0.35], 'User9': [0.45, 0.65, 0.35, 0.25, 0.55, 0.55, 0.75, 0.95, 0.15, 0.25], 'User14': [0.45, 0.25, 0.25, 0.35, 0.65, 0.75, 0.35, 0.85, 0.15, 0.75], 'User19': [0.45, 0.65, 0.15, 0.15, 0.25, 0.85, 0.55, 0.85, 0.35, 0.75], 'User20': [0.15, 0.85, 0.35, 0.55, 0.15, 0.55, 0.55, 0.35, 0.85, 0.65], 'User23': [0.15, 0.75, 0.65, 0.45, 0.15, 0.45, 0.85, 0.25, 0.85, 0.45], 'User24': [0.45, 0.65, 0.15, 0.15, 0.25, 0.65, 0.65, 0.95, 0.35, 0.75], 'User25': [0.25, 0.65, 0.25, 0.15, 0.35, 0.75, 0.75, 0.75, 0.45, 0.65], 'User26': [0.35, 0.65, 0.05, 0.25, 0.35, 0.75, 0.55, 0.75, 0.35, 0.95], 'User27': [0.25, 0.85, 0.75, 0.75, 0.45, 0.35, 0.75, 0.15, 0.45, 0.25], 'User29': [0.75, 0.15, 0.35, 0.25, 0.65, 0.55, 0.65, 0.95, 0.45, 0.25], 'User32': [0.45, 0.55, 0.15, 0.15, 0.25, 0.65, 0.75, 0.95, 0.55, 0.45], 'User34': [0.55, 0.45, 0.05, 0.45, 0.15, 0.65, 0.35, 0.75, 0.45, 0.95], 'User35': [0.15, 0.85, 0.25, 0.45, 0.15, 0.35, 0.65, 0.35, 0.85, 0.75], 'User38': [0.35, 0.65, 0.15, 0.15, 0.35, 0.85, 0.75, 0.85, 0.35, 0.55], 'User39': [0.35, 0.75, 0.05, 0.25, 0.35, 0.85, 0.65, 0.85, 0.35, 0.55], 'User40': [0.25, 0.85, 0.15, 0.35, 0.25, 0.75, 0.75, 0.75, 0.35, 0.55], 'User44': [0.45, 0.45, 0.95, 0.75, 0.55, 0.25, 0.25, 0.05, 0.55, 0.75], 'User47': [0.15, 0.85, 0.55, 0.55, 0.15, 0.55, 0.85, 0.35, 0.65, 0.35], 'User53': [0.25, 0.95, 0.15, 0.25, 0.55, 0.75, 0.75, 0.65, 0.25, 0.45], 'User54': [0.65, 0.65, 0.25, 0.35, 0.35, 0.65, 0.85, 0.75, 0.25, 0.25], 'User55': [0.45, 0.55, 0.15, 0.15, 0.25, 0.75, 0.65, 0.95, 0.35, 0.75], 'User56': [0.55, 0.65, 0.15, 0.15, 0.45, 0.65, 0.75, 0.95, 0.35, 0.35], 'User57': [0.65, 0.15, 0.55, 0.45, 0.95, 0.35, 0.05, 0.25, 0.65, 0.85], 'User59': [0.35, 0.75, 0.65, 0.85, 0.35, 0.55, 0.45, 0.05, 0.25, 0.75]}}
ground_truth_group_ratings = {'Group1': [0.55, 0.3, 0.15, 0.35000000000000003, 0.75, 0.8, 0.3, 0.75, 0.3, 0.7], 'Group2': [0.44999999999999996, 0.575, 0.225, 0.325, 0.4749999999999999, 0.65, 0.525, 0.7250000000000001, 0.375, 0.625], 'Group3': [0.39285714285714285, 0.5642857142857144, 0.39285714285714285, 0.3642857142857143, 0.36428571428571427, 0.55, 0.7071428571428572, 0.6071428571428571, 0.4642857142857143, 0.5785714285714285], 'Group4': [0.41000000000000003, 0.5766666666666667, 0.37, 0.4633333333333333, 0.4633333333333333, 0.5900000000000001, 0.49, 0.51, 0.4433333333333333, 0.6166666666666668], 'Group5': [0.3879310344827586, 0.6293103448275861, 0.3258620689655173, 0.36724137931034484, 0.3741379310344827, 0.6017241379310346, 0.6258620689655172, 0.6120689655172412, 0.4603448275862068, 0.5844827586206897]}

start_time = time.time()
# Calculate Shapley values
shapley_values = calculate_shapley_values(user_ratings, ground_truth_group_ratings, group_names)
end_time = time.time()
 # Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Execution time: {elapsed_time} seconds")
# Display results
#for group_name, values in shapley_values.items():
    #print(f"Shapley values for {group_name}:")
    #for subset, value in values.items():
        #print(f"Subset {subset}: {value}")
    #print("\n")

# Select top 2 subsets for each group based on Shapley values
top_subsets = select_top_subsets(shapley_values)
# Select top 2 subsets for each group based on Shapley values
top_subsets = select_top_subsets(shapley_values)
# Determine the preferred items for each top subset
preferred_items = {}
for group_name, subsets in top_subsets.items():
    preferred_items[group_name] = {}
    for subset, _ in subsets:
        subset_ratings = {user: user_ratings[group_name][user] for user in subset}
        coalition_ratings = combine_user_ratings(subset_ratings)
        sorted_indices = np.argsort(coalition_ratings)[::-1]
        preferred_items[group_name][subset] = sorted_indices[:3] + 1  # Adding 1 to convert to 1-based indexing

# Display top 2 preferred items for each subset in top 2
for group_name, items in preferred_items.items():
    print(f"Top 2 Preferred items for {group_name}:")
    for subset, preferred_item_indices in items.items():
        print(f"Subset {subset}: Preferred items - {preferred_item_indices}")
    print("\n")


# Dictionary to store the top 2 items for each group
top_items_per_group = {}

# Iterate through each group
for group_name, subsets in preferred_items.items():
    # Counter to store the counts of items in subsets of the group
    counts = Counter()
    # Iterate through subsets of the group
    for subset, preferred_items in subsets.items():
        # Increment counts for each item in the preferred items
        counts.update(preferred_items)
    # Select the top 2 items with the highest counts
    top_items = [item for item, _ in counts.most_common(3)]
    # Store the top items for the group
    top_items_per_group[group_name] = top_items
    print("top_items_per_group",top_items_per_group)
# Display the top 2 items for each group
for group_name, top_items in top_items_per_group.items():
    print(f"Top 2 items for {group_name}: {top_items}")

def calculate_user_satisfaction(group_name, top_items_per_group, user_ratings, threshold):
    top_items_for_group = top_items_per_group[group_name]
    satisfaction_values = []
    total_users_in_group = len(user_ratings[group_name])

    for item_idx in top_items_for_group:
        #print("ratings[item_idx]", item_idx)
        print(user_ratings[group_name].items())
        satisfied_count = sum(
            #1 for user in user_ratings[group_name] if user_ratings[group_name][user][item_idx] >= threshold)

            1 for user, ratings in user_ratings[group_name].items() if  ratings[item_idx-1] >= threshold)

        #print("satisfied_count",satisfied_count)
        satisfaction_fraction = satisfied_count / total_users_in_group
       # print("satisfaction_fraction",satisfaction_fraction)
        satisfaction_values.append(satisfaction_fraction)
    print("satisfaction_values",satisfaction_values)
    return satisfaction_values
# Calculate and print user satisfaction for each group
threshold = 0.4  # Set your desired threshold value
for group_name in group_names:
    satisfaction_values = calculate_user_satisfaction(group_name, top_items_per_group, user_ratings, threshold)
    total_satisfaction = sum(satisfaction_values) / len(top_items_per_group[group_name])
    print(f"Total User Satisfaction for Group {group_name}: {total_satisfaction * 100:.2f}%")
# Function to calculate precision, recall, f1-score, and accuracy

# Calculate and print user satisfaction for each group
threshold = 0.4  # Set your desired threshold value
total_weighted_satisfaction = 0
total_weight = 0

for group_name in group_names:
    satisfaction_values = calculate_user_satisfaction(group_name, top_items_per_group, user_ratings, threshold)
    total_satisfaction = sum(satisfaction_values) / len(top_items_per_group[group_name])
    total_weighted_satisfaction += total_satisfaction * len(top_items_per_group[group_name])
    total_weight += len(top_items_per_group[group_name])
    print(f"Total User Satisfaction for Group {group_name}: {total_satisfaction * 100:.2f}%")

# Calculate overall user satisfaction
overall_satisfaction = total_weighted_satisfaction / total_weight
print(f"\nOverall User Satisfaction: {overall_satisfaction * 100:.2f}%")

def calculate_precision1(ground_truth_group_ratings, top_items_per_group, group_names, num_top_items=3, threshold=0.4):
    y_true = []
    y_pred = []

    for group_name in group_names:
        # Convert ground truth ratings to a NumPy array
        ground_truth_ratings = np.array(ground_truth_group_ratings[group_name])

        # Threshold ground truth ratings for the top items
        y_true.extend((ground_truth_ratings[:num_top_items] > threshold).astype(int))

        # Convert top_items_group to a NumPy array
        top_items_group = np.array(top_items_per_group[group_name])

        # Threshold predicted ratings for the top items
        y_pred.extend((top_items_group[:num_top_items] > threshold).astype(int))

    # Convert the lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #print("y_true shape:", y_true.shape)
    #print("y_pred shape:", y_pred.shape)

    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall,f1,accuracy


# Calculate precision, recall, f1-score, and accuracy
precision, recall, f1, accuracy = calculate_precision1(ground_truth_group_ratings, top_items_per_group, group_names)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)