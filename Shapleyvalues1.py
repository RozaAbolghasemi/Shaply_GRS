from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

# Function to select the top 2 subsets for each group based on Shapley values
def select_top_subsets(shapley_values, top_n=200):
    top_subsets = {}
    for group_name, values in shapley_values.items():

        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        top_subsets[group_name] = sorted_values[:top_n]
    return top_subsets

def calculate_shapley_values(user_ratings, ground_truth_group_ratings, group_names):
    shapley_values = {}

    for group_name in group_names:
        print(f"Group name: {group_name}")
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
    print("subset",subset)

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

# Input data
group_names = ['Group1', 'Group2', 'Group3','Group4','Group5','Group6','Group7','Group8','Group9','Group10','Group11','Group12']

user_ratings = {
    'Group1': {'User1': [.34, .33, .54, .36, .67, .77], 'User2': [.51, .45, .55, .42, .66, .42],
               'User3': [.47, .52, .49, .53, .47, .53], 'User4': [.71, .62, .57, .44, .35, .32],
               'User5': [.35, .74, .41, .49, .55, .46]},
'Group2': {'User6': [.23, .79, .44, .53, .65, .66], 'User7': [.20, .52, .31, .55, .76, .52],
           'User8': [.45, .48, .55, .64, .47, .41],
           'User9': [.35, .34, .46, .57, .61, .69], 'User10': [.32, .60, .39, .75, .51, .44]},
        'Group3': {'User11': [.38, .34, .47, .57, .63, .61], 'User12': [.48, .48, .52, .39, .67, .46],
                   'User13': [.43, .55, .46, .48, .52, .56], 'User14': [.68, .62, .53, .42, .38, .37],
                   'User15': [.37, .67, .43, .47, .79, .53]},
        'Group4': {'User16': [.27, .47, .43, .54, .64, .65], 'User17': [.28, .54, .37, .56, .89, .51],
                   'User18': [.41, .51, .51, .60, .52, .47], 'User19': [.41, .37, .47, .58, .60, .66],
                   'User20': [.29, .55, .43, .67, .59, .47]},
        'Group5': {
            'User3': [.47, .52, .49, .53, .47, .53],
            'User4': [.71, .62, .57, .44, .35, .32],
            'User8': [.45, .48, .55, .64, .47, .41],
            'User9': [.35, .34, .46, .57, .61, .69],
            'User13': [.43, .55, .46, .48, .52, .56],
            'User14': [.68, .62, .53, .42, .38, .37],
            'User15': [.37, .67, .43, .47, .79, .53],
            'User17': [.28, .54, .37, .56, .89, .51],
            'User18': [.41, .51, .51, .60, .52, .47],
            'User19': [.41, .37, .47, .58, .60, .66]},
        'Group6': {'User1': [.34, .33, .54, .36, .67, .77],
                   'User10': [.32, .60, .39, .75, .51, .44],
                   'User12': [.48, .48, .52, .39, .67, .46],
                   'User16': [.27, .47, .43, .54, .64, .65],
                   'User20': [.29, .55, .43, .67, .59, .47]},
        'Group7': {'User11': [.38, .34, .47, .57, .63, .61],
                   'User6': [.23, .79, .44, .53, .65, .66]},
        'Group8': {'User19': [0.5491680391401026, 0.4091159881431616, 0.5238989337199339, 0.4253645624334623,
                              0.5179955334454852, 0.5744569431178544],
                   'User9': [0.5073736059037911, 0.5173342165588989, 0.487800914983563, 0.4907568092728453,
                             0.4863949930461826, 0.5103394602347191],
                   'User24': [0.57856469275859, 0.44488525775372345, 0.5111817029841895, 0.49244928784992575,
                              0.4227241540950725, 0.5501949045584988],
                   'User29': [0.6825017643028186, 0.38451661384219077, 0.6134287290633799, 0.5019716119424882,
                              0.26961398254570845, 0.5479672983034142],
                   'User20': [0.6484139140799251, 0.5961266731894571, 0.5190812087069546, 0.49257057834261136,
                              0.3023249689883916, 0.44148265669266024],
                   'User30': [0.5101538358494029, 0.5925846090445834, 0.4002018683286202, 0.7034593272468473,
                              0.3348622369388216, 0.4587381225917246],
                   'User13': [0.7079540480563146, 0.2526116870044225, 0.6490527139328618, 0.4349512261565043,
                              0.43765895313320796, 0.5177713717166889],
                   'User3': [0.6298934473338592, 0.3964527065681313, 0.56961282847719, 0.509634786431142,
                             0.3975567875967332, 0.4968494435929444],
                   'User4': [0.51044533366122, 0.5348922970382303, 0.5030626587374444, 0.41024996484860593,
                             0.5127910884865944, 0.5285586572279048],
                   'User14': [0.5478790071412487, 0.47230169885603535, 0.5188067125670089, 0.5119460699837155,
                              0.465408332965161, 0.4836581784868305],
                   'User17': [0.32224325916571833, 0.5228935971318422, 0.429571267704719, 0.6286967523603214,
                              0.6229375899007911, 0.4736575337366078],
                   'User23': [0.543114939660581, 0.4477826768605702, 0.47134676630670275, 0.5137910157686174,
                              0.4802398289896075, 0.5437247724139213]},
        'Group9': {'User13': [0.7079540480563146, 0.2526116870044225, 0.6490527139328618, 0.4349512261565043,
                              0.43765895313320796, 0.5177713717166889],
                   'User17': [0.32224325916571833, 0.5228935971318422, 0.429571267704719, 0.6286967523603214,
                              0.6229375899007911, 0.4736575337366078],
                   'User14': [0.5478790071412487, 0.47230169885603535, 0.5188067125670089, 0.5119460699837155,
                              0.465408332965161, 0.4836581784868305],
                   'User28': [0.5394440187444208, 0.4705764940039812, 0.4829233799776344, 0.5189443360502198,
                              0.46408122962395787, 0.5240305415997857],
                   'User26': [0.4987837121546044, 0.46128688930234985, 0.5052310419789682, 0.481649125482996,
                              0.49789216846407963, 0.555157062617002],
                   'User18': [0.6030973254896549, 0.48887630443079827, 0.5198043060095414, 0.5389307664963858,
                              0.40588245053463584, 0.4434088470389838]},
    'Group10': {
        'User10': [0.4416748851621039, 0.4808340871184649, 0.4772634280547472, 0.5484929669053865,
                   0.5026392455059032, 0.5490953872533941],
        'User6': [0.5293631530302498, 0.47084038736821515, 0.4974465103332209, 0.45045009091004634,
                  0.46975708086787954, 0.5821427774903883],
        'User3': [0.6298934473338592, 0.3964527065681313, 0.56961282847719, 0.509634786431142, 0.3975567875967332,
                  0.4968494435929444],
        'User23': [0.543114939660581, 0.4477826768605702, 0.47134676630670275, 0.5137910157686174,
                   0.4802398289896075, 0.5437247724139213],
        'User1': [0.5364127124157784, 0.4872918672266379, 0.4976566906486129, 0.503292207210146, 0.4624961193816854,
                  0.5128504031171394]},
'Group11': {
            'User18': [0.6030973254896549, 0.48887630443079827, 0.5198043060095414, 0.5389307664963858,
                       0.40588245053463584, 0.4434088470389838],
            'User6': [0.5293631530302498, 0.47084038736821515, 0.4974465103332209, 0.45045009091004634,
                      0.46975708086787954, 0.5821427774903883],
            'User28': [0.5394440187444208, 0.4705764940039812, 0.4829233799776344, 0.5189443360502198,
                       0.46408122962395787, 0.5240305415997857],
            'User5': [0.620899541849531, 0.7242200374023963, 0.36034684000147227, 0.5150608555782032,
                      0.17104251251589664, 0.6084302126525005],
            'User10': [0.4416748851621039, 0.4808340871184649, 0.4772634280547472, 0.5484929669053865,
                       0.5026392455059032, 0.5490953872533941],
            'User1': [0.5364127124157784, 0.4872918672266379, 0.4976566906486129, 0.503292207210146, 0.4624961193816854,
                      0.5128504031171394],
            },
    'Group12': {
            'User25': [0.5272610227773545, 0.4753575060749822, 0.4897669815812587, 0.5081430583039808,
                       0.4616813440900173, 0.5377900871724065],
            'User29': [0.6825017643028186, 0.38451661384219077, 0.6134287290633799, 0.5019716119424882,
                       0.26961398254570845, 0.5479672983034142],
            'User2': [0.5575603170327784, 0.42263181822082396, 0.5090018457953226, 0.43905166409395485,
                      0.500141325504767, 0.571613029352353]},

}
ground_truth_group_ratings = {
    'Group1': [.48, .53, .51, .45, .54, .50],
'Group2': [.31, .55, .43, .61, .60, .55],
        'Group3': [.47, .53, .48, .47, .60, .51],
        'Group4': [.33, .49, .44, .59, .65, .55],
        'Group5': [0.45600000000000007, 0.522, 0.4840000000000001, 0.5289999999999999, 0.5599999999999999, 0.505],
        'Group6': [0.34, 0.48599999999999993, 0.462, 0.542, 0.616, 0.558],
        'Group7': [0.305, 0.5650000000000001, 0.45499999999999996, 0.55, 0.64, 0.635],
        'Group8': (0.5614754905877977, 0.4642915018326039, 0.5164205254593807, 0.5096534993864239, 0.4375423708443131,
                   0.5106166118894808),
        'Group9': (
        0.536566895125327, 0.4447577784549049, 0.5175649036951223, 0.5191863794216904, 0.48231012077030555,
        0.49961392253264975),
'Group10': (
        0.5360918275205144, 0.4566403450284039, 0.5026652447640948, 0.5051322134450676, 0.4625378124683618,
        0.5369325567735574),
'Group11': (
        0.54514860611529, 0.520439862925082, 0.472573525837538,0.512528537191731, 0.41264977307166, 0.536659694858699),
        'Group12': (
        0.5891077013709838, 0.42750197937933226, 0.5373991854799871, 0.48305544478014123, 0.4104788840468309,
        0.5524568049427246),
}

# Calculate Shapley values
shapley_values = calculate_shapley_values(user_ratings, ground_truth_group_ratings, group_names)

# Display results
for group_name, values in shapley_values.items():
    print(f"Shapley values for {group_name}:")
    for subset, value in values.items():
        print(f"Subset {subset}: {value}")
    print("\n")

# Select top 2 subsets for each group based on Shapley values
top_subsets = select_top_subsets(shapley_values)
# Select top 2 subsets for each group based on Shapley values
top_subsets = select_top_subsets(shapley_values)

# Determine the preferred item index for each top subset
preferred_items = {}
for group_name, subsets in top_subsets.items():
    preferred_items[group_name] = []
    for subset, _ in subsets:
        subset_ratings = {user: user_ratings[group_name][user] for user in subset}
        coalition_ratings = combine_user_ratings(subset_ratings)
        preferred_item_index = np.argmax(coalition_ratings)
        preferred_items[group_name].append(preferred_item_index + 1)  # Add 1 to convert to 1-based indexing

# Count the frequency of each preferred item index for each group
top_item_counts = {}
for group_name, items in preferred_items.items():
    top_item_counts[group_name] = Counter(items)

# Select the top 2 items for each group based on frequency
top_items = {}
for group_name, counts in top_item_counts.items():
    # Extract the top 2 items
    top_items[group_name] = [item for item, _ in counts.most_common(2)]

    # Display the top 2 items for each group
for group_name, items in top_items.items():
    print(f"Top 2 items for {group_name}: {items}")

def calculate_user_satisfaction(group_name, top_items, user_ratings, threshold):
    top_items_for_group = top_items[group_name]
    satisfaction_values = []
    total_users_in_group = len(user_ratings[group_name])

    unique_items = set(top_items_for_group)  # Get unique items chosen among top preferred items

    for item_idx in unique_items:
        print(item_idx)
        satisfied_count = sum(
            1 for user, ratings in user_ratings[group_name].items() if len(ratings) > item_idx and ratings[item_idx] >= threshold)

        satisfaction_fraction = satisfied_count / total_users_in_group

        satisfaction_values.append(satisfaction_fraction)

    return satisfaction_values

# Set the threshold value
threshold = 0.4

# Call the function for each group
for group_name in group_names:
    satisfaction_values = calculate_user_satisfaction(group_name, top_items, user_ratings, threshold)
    total_satisfaction = sum(satisfaction_values) / len(satisfaction_values)
    print(f"Total User Satisfaction for Group {group_name}: {total_satisfaction * 100:.2f}%")

def calculate_precision1(ground_truth_group_ratings, top_items, group_names, num_top_items=2, threshold=0.3):
    y_true = []
    y_pred = []

    for group_name in group_names:
        # Convert ground truth ratings to a NumPy array
        ground_truth_ratings = np.array(ground_truth_group_ratings[group_name])

        # Threshold ground truth ratings for the top items
        y_true.extend((ground_truth_ratings[:num_top_items] > threshold).astype(int))

        # Convert top_items_group to a NumPy array
        top_items_group = np.array(top_items[group_name])

        # Threshold predicted ratings for the top items
        y_pred.extend((top_items_group[:num_top_items] > threshold).astype(int))

    # Convert the lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)

    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall,f1,accuracy
precision,recall,f1,accuracy= calculate_precision1(ground_truth_group_ratings, top_items, group_names)
print("Precision:", precision)
print("recall:", recall)
print("f1:", f1)
print("accuracy:", accuracy)

