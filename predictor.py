import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# from pandas.plotting import scatter_matrix

def get_groups(my_list, variable, is_catagorical, num_cats):
    val_counts = my_list[variable].value_counts().index.tolist()
    val_counts.insert(0, is_catagorical)
    val_counts.insert(0, variable)

    if len(val_counts) <= num_cats + 2:
        return val_counts
    else:
        out_list = list()
        for i in range(0, num_cats + 2):
            out_list.append(val_counts[i])
        return out_list


def freq_loan_paid(my_list):
    test = my_list['loan_paid'].value_counts(normalize=True)
    return test[1]


def entropy(my_list):
    freq = freq_loan_paid(my_list)
    if freq <= 0:
        return 0
    elif freq >= 1:
        return 1
    return (math.log(freq, 2) * freq * -1) - (math.log(1 - freq, 2) * (1 - freq))


def gain(starting_list, list1, list2):
    if len(starting_list) <= 1 or len(list1) <= 1 or len(list2) <= 1:
        return 0
    size_list1 = len(list1) / len(starting_list)
    return (entropy(starting_list) - ((entropy(list1) * size_list1) + (entropy(list2) * (1 - size_list1))))


def split_numerical(variable, split_point, my_list):
    left = my_list[my_list[variable] <= split_point]
    right = my_list[my_list[variable] > split_point]
    return [left, right]


def split_catagorical(variable, split_point, my_list):
    left = my_list[my_list[variable] == split_point]
    right = my_list[my_list[variable] != split_point]
    return [left, right]


def get_best_split(my_list, data_points):
    best_gain = 0
    best_catagory = None
    best_point = None
    best_is_catagorical = None
    best_lrsplit = None
    best_is_catagorical = None
    for catagory in data_points:
        for i in range(2, len(catagory)):
            returned_lists = None
            if catagory[1]:
                returned_lists = split_catagorical(catagory[0], catagory[i], my_list)
            else:
                returned_lists = split_numerical(catagory[0], catagory[i], my_list)
            if len(returned_lists[0]) > 100 and len(returned_lists[1]) > 100 and len(my_list) > 100:
                found_gain = gain(my_list, returned_lists[0], returned_lists[1])
                if found_gain > best_gain:
                    best_gain = found_gain
                    best_catagory = catagory[0]
                    best_point = catagory[i]
                    best_lrsplit = returned_lists
                    best_is_catagorical = catagory[1]
    return {'gain': best_gain, 'catagory': best_catagory, 'point': best_point, 'lrsplit': best_lrsplit,
            'is_cata': best_is_catagorical}


def create_node(list_part):
    if freq_loan_paid(list_part) > 0.75:
        return 1
    return 0


def split(node, max_depth, min_size, curr_depth, data_points):
    left = node['lrsplit'][0]
    right = node['lrsplit'][1]
    del (node['lrsplit'])

    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = create_node(left + right)
        return
    if curr_depth >= max_depth:
        node['left'] = create_node(left)
        node['right'] = create_node(right)
        return

    if len(left) <= min_size:
        node['left'] = create_node(left)
    else:
        node['left'] = get_best_split(left, data_points)
        if node['left']['gain'] < 0.00001:
            node['left'] = create_node(left)
        else:
            split(node['left'], max_depth, min_size, curr_depth + 1, data_points)

    if len(right) <= min_size:
        node['right'] = create_node(right)
    else:
        node['right'] = get_best_split(right, data_points)
        if node['right']['gain'] < 0.00001:
            node['right'] = create_node(right)
        else:
            split(node['right'], max_depth, min_size, curr_depth + 1, data_points)


def build_tree(train_data, max_depth, min_size, data_points):
    root = get_best_split(train_data, data_points)
    split(root, max_depth, min_size, 1, data_points)
    return root


def print_tree(node, depth=0):
    if isinstance(node, dict):
        if node['is_cata']:
            print('%s[%s == %s]' % ((depth * ' ', (node['catagory']), node['point'])))
        else:
            print('%s[%s <= %.3f]' % ((depth * ' ', (node['catagory']), node['point'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


def make_prediction(node, data_to_predict):
    if node['is_cata']:
        if data_to_predict[node['catagory']] == node['point']:
            if isinstance(node['left'], dict):
                return make_prediction(node['left'], data_to_predict)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return make_prediction(node['right'], data_to_predict)
            else:
                return node['right']
    else:
        if data_to_predict[node['catagory']] <= node['point']:
            if isinstance(node['left'], dict):
                return make_prediction(node['left'], data_to_predict)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return make_prediction(node['right'], data_to_predict)
            else:
                return node['right']


# START OF MAIN


# DEFINE STUFF
max_depth = 100
min_size = 101
max_cats = 10

train_data = pd.read_csv("lending_train.csv", delimiter=",", header='infer', quotechar="\"", comment=None)
test_data = pd.read_csv("lending_topredict.csv", delimiter=",", header='infer', quotechar="\"", comment=None)

numerical_points = ["requested_amnt", "annual_income", "debt_to_income_ratio", "public_bankruptcies",
                    "delinquency_last_2yrs", "fico_score_range_low", "fico_score_range_high",
                    "fico_inquired_last_6mths", "months_since_last_delinq", "revolving_balance",
                    "total_revolving_limit", "any_tax_liens"]
# removed points due to lack of relation:
#
catagorical_points = ["loan_duration", "employment", "reason_for_loan", "employment_verified",
                      "state", "home_ownership_status", "type_of_application"]
# removed points due to lack of relation:
#
compiled_points = list()

for catagory in catagorical_points:
    compiled_points.append(get_groups(train_data, catagory, True, max_cats))

for catagory in numerical_points:
    compiled_points.append(get_groups(train_data, catagory, False, max_cats))

tree = build_tree(train_data, max_depth, min_size, compiled_points)
print_tree(tree)

num_right = 0
num_wrong = 0

d = {'ID': [], 'loan_paid': []}

print("beginning tests")
for i in range(0, len(test_data)):
    d['ID'].append(test_data.iloc[i]['ID'])
    d['loan_paid'].append(make_prediction(tree, test_data.iloc[i]))
print("done guessing :)")
df = pd.DataFrame(data=d)
df.to_csv("Loan_ToSubmit.csv", index=False)

