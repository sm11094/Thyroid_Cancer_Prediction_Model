from crud import insert 
import csv

def squared_euclidean_distance(p1, p2):
    """
    Calculates the squared Euclidean distance between two points
    Using squared distance during search avoids expensive square root calculations
    Author: Zaid
    """
    return sum((x - y) ** 2 for x, y in zip(p1, p2))


def get_max_distance(node):
    """
    Finds the worst (largest) distance by traveling down the right branch
    Author: Mohsin
    """

    if node is None:
        return float('inf')
        
    while node['right'] is not None:
        node = node['right']
        
    return node['point'][0] # Distance is stored as a 1D point


def remove_max(node):
    """
    Removes the rightmost node (worst distance) to keep size at k
    Author: Mohsin
    """

    if node is None:
        return None
        
    # If there is no right child, this node is the maximum. 
    if node['right'] is None:
        return node['left']
        
    node['right'] = remove_max(node['right'])
    return node


def flatten_neighbors(node, result_list):
    """
    In-order traversal to extract neighbors sorted from closest to furthest
    Author: Ahsan
    """

    if node is not None:
        flatten_neighbors(node['left'], result_list)
        
        # The neighbor dictionary is stored in the 'label'
        neighbor_data = node['label']
        neighbor_data['distance'] = neighbor_data['distance'] ** 0.5
        result_list.append(neighbor_data)
        
        flatten_neighbors(node['right'], result_list)


def search_knn(current_node, target_point, k, tracker):
    """
    Recursive helper function to traverse the main 9D KD-Tree.
    Updates the 1D Neighbor KD-Tree in place.
    Average time complexity: O(log n)
    Author: Ahsan
    """

    if current_node is None:
        return

    dist_sq = squared_euclidean_distance(current_node['point'], target_point)
    
    # Creating a new neighbor to add to the tracking tree after comparing its distance and other distances
    neighbor_data = {
        'point': current_node['point'],
        'label': current_node['label'],
        'distance': dist_sq
    }

    worst_dist = get_max_distance(tracker['root']) # get farthest node from target_point in our 1D Neighbor Tree

    if tracker['count'] < k: # If less than k neighbors in tracker, then add one
        tracker['root'] = insert(tracker['root'], [dist_sq], neighbor_data)
        tracker['count'] += 1
    elif dist_sq < worst_dist: # If tracker is full, but we found a closer neighbor, add it
        tracker['root'] = insert(tracker['root'], [dist_sq], neighbor_data)
        tracker['root'] = remove_max(tracker['root'])

    axis = current_node['axis'] # Getting axis of last splitting
    diff = target_point[axis] - current_node['point'][axis] # Distance btw current point and splitting axis (barrier between good side and bad side)

    if diff < 0:    
        good_side = current_node['left']
        bad_side = current_node['right']
    else:
        good_side = current_node['right']
        bad_side = current_node['left']

    search_knn(good_side, target_point, k, tracker) # search the good side
    current_worst = get_max_distance(tracker['root']) # Gets distance of current farthest neighbor

    # If either neighbors in tracker are less than k or if distance btw current point of splitting axis is smaller than or equal to worst neighbor distance, only then we go bad side, otherwise skip it.
    if tracker['count'] < k or (diff ** 2) <= current_worst:
        search_knn(bad_side, target_point, k, tracker)

def get_knn(node, target_point, k):
    """
    Main function to initialize the K-Nearest Neighbors search.
    Returns a sorted list of dictionaries.
    Author: Mohsin
    """
    tracker = {'root': None, 'count': 0} 
    
    search_knn(node, target_point, k, tracker)

    k_nearest = []
    flatten_neighbors(tracker['root'], k_nearest)
        
    return k_nearest
