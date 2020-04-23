import os.path
from sklearn.model_selection import train_test_split

def train_test_split_dir(path, ratio = 0.7):
    sum_of_path = []
    for scene_list in os.listdir(path):  # for scene list e.g) alley_1, ambush_4
        grand_parent_dir = os.path.join(os.path.join(path, scene_list))
        for start_list in os.listdir(grand_parent_dir): 
            parent_dir = os.path.join(grand_parent_dir, start_list)
            for end_list in os.listdir(parent_dir):
                sum_of_path.append(os.path.join(parent_dir, end_list))
    sum_list = list(set(sum_of_path))
    training_dir, evaluation_dir = train_test_split(sum_list, train_size = ratio)
    return training_dir, evaluation_dir
