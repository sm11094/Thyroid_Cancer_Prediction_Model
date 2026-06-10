import csv


def build_kdtree(data, depth=0):
    """
    Recursively builds a KD-Tree from a list of tuples
    The tuples are in the form (point, label)
    Returns the root node as a dictionary
    Average time complexity: O(n log n)
    Author: Zaid
    """

    if not data:    # Indicates leaf node has been reached, no more child nodes to add
        return None
        
    axis = depth % 9   # 9 is the number of dimensions, axis is the splitting axis.
    
    data.sort(key=lambda x: x[0][axis]) # Data is sorted by the current splitting axis, causes time complexity to O(nlogn)
    
    middle_index = len(data) // 2 # Using the middle index as the root node
    
    # Building the node
    node = {
        'point': data[middle_index][0], # features
        'label': data[middle_index][1], # label
        'axis': axis, # splitting axis used for this node
        'left': build_kdtree(data[:middle_index], depth + 1), # Recursive step to build other nodes to the left of this node
        'right': build_kdtree(data[middle_index + 1:], depth + 1) # Recursive step to build other nodes to the right of this node
    }

    return node


def search(node, target_point):
    """ 
    Takes a node (tree) as an input
    Searches for a specific point in the KD-Tree
    Returns the node if found, otherwise False
    Average time complexity: O(log n)
    Author: Mohsin
    """

    if node == None: # Target node doesnt doesnt exist in the tree
        return False 
        
    if node['point'] == target_point: # Target node found
        return node
        
    axis = node['axis'] # This is the splitting axis
    
    if target_point[axis] < node['point'][axis]:  # Traverse left or right based on the splitting axis
        return search(node['left'], target_point)
    
    elif target_point[axis] > node['point'][axis]:
        return search(node['right'], target_point) # Recursively search for target point
    
    else:    # target_point[axis] == node['point'][axis]
        left_result = search(node['left'], target_point) # Check the left branch first  
        if left_result:
            return left_result
            
        return search(node['right'], target_point) # If not in the left, it must be in the right



def insert(node, point, label, depth=0):
    """
    Takes a node (tree) as an input
    Inserts a new point and label into the KD-Tree
    Returns the updated node (tree)
    Average time complexity: O(log n)
    Author: Mohsin
    """

    k = len(point)
    
    if node == None: # end of the tree found (leaf node)
        return {
            'point': point,
            'label': label,
            'axis': depth % k,
            'left': None,
            'right': None
        }
        
    axis = node['axis']
    
    if point[axis] < node['point'][axis]:     # Route the point left or right
        node['left'] = insert(node['left'], point, label, depth + 1)

    else:
        node['right'] = insert(node['right'], point, label, depth + 1)
        
    return node



def find_min(node, target_axis):
    """
    inputs: 
        node: a dictionary representing the root node or the starting node to find minimum from.
        target_axis: the axis with respect to which, find the minimum, e.g x or y.
    Finds the node with the minimum point value along a specific axis
    Returns a dictionary representing the minimum node
    Average time Complexity: O(n^(1-1/k))
    Author: Ahsan
    """

    if node == None:
        return None  
          
    axis = node['axis']
    
    # If the axis along which we are dividing is target axis, then the minimum exist in towards left of current node
    if axis == target_axis: 
        if node['left'] == None:
            return node
        
        return find_min(node['left'], target_axis)
        
    # Otherwise the minimum can be to the left, right, or  current node can also be minimum
    left_minimum = find_min(node['left'], target_axis)
    right_minimum = find_min(node['right'], target_axis)
    
    min_node = node

    if left_minimum and left_minimum['point'][target_axis] < min_node['point'][target_axis]:
        min_node = left_minimum

    if right_minimum and right_minimum['point'][target_axis] < min_node['point'][target_axis]:
        min_node = right_minimum
    
    return min_node



def delete(node, target_point):
    """
    inputs: 
        node: a dictionary representing the KD-Tree or the starting node in the tree.
        target_point: the point of the node to delete.
    Deletes a target point from the KD-Tree 
    Rreturns a the updated KD-Tree after deleting the target_point
    Average time complexity: O(n^(1-1/k))
    Author: Ahsan
    """

    if node == None:
        return None
    
    axis = node['axis']
    
    if node['point'] == target_point:     # if the target_point is found, delete it

        if node['right'] != None:   # Finding minimum to the right of current node, to replace current node with it
            min_node = find_min(node['right'], axis)
            node['point'] = min_node['point']
            node['label'] = min_node['label']
            node['right'] = delete(node['right'], min_node['point'])

        elif node['left'] != None: # Swap all left child with right child so as to maintain tree properties, then find minimum
            min_node = find_min(node['left'], axis)
            node['point'] = min_node['point']
            node['label'] = min_node['label']
            node['right'] = delete(node['left'], min_node['point'])
            node['left'] = None
            
        else:
            return None # if it is a leaf node, we delete it
            
    # if current node is not the node to delete, keep going down
    elif target_point[axis] < node['point'][axis]: 
        node['left'] = delete(node['left'], target_point)
    else:
        node['right'] = delete(node['right'], target_point)
        
    return node


def initialize_tree(features_file, labels_file, id_file):
    '''
    Takes the feature file and label file as inputs
    Formats the input files to be usable
    Returns the initialized KD-Tree
    Average time complexity: O(log n)
    Author: Zaid
    '''

    print("Loading data...") # Added to account for wait time
    data = []
    id_database = {}
    
    with open(id_file, 'r') as ids, open(features_file, 'r') as features, open(labels_file, 'r') as labels:
        id_reader = csv.reader(ids)
        feature_reader = csv.reader(features)
        label_reader = csv.reader(labels)
        
        # Skip the headers in the feature and label files
        next(id_reader)
        next(feature_reader)
        next(label_reader)
        
        for id_row, feature_row, label_row in zip(id_reader, feature_reader, label_reader):
            patient_id = int(id_row[0])
            point = [float(x) for x in feature_row]
            label = int(label_row[3]) 
            data.append((point, label, id))
            id_database[patient_id] = point
            
    print(f"Data loaded: \nBuilding KD-Tree...") # Added to account for wait time

    kd_tree = build_kdtree(data)         # Initializing the tree
    print("KD-Tree built successfully!") # Added message to indicate tree is built because the tree itself is too large to be printed
    
    return kd_tree, id_database
