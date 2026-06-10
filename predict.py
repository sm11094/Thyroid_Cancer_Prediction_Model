from kNN import *
from crud import *
import csv

def generate_prediction(neighbors):
    """
    Takes the k nearest neighbors as input --> list of nodes containing point, label, distance
    Performs majority voting on the k nearest neighbors.
    Returns the predicted label and a confidence score (percentage).
    Time complexity: O(k) --> k is the number of nearest neighbors in kNN which is 5
    Author: Zaid
    """
    
    label_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0} # initialize labels and their counts
    
    for neighbor in neighbors:     
        label = neighbor['label']
        label_counts[label] += 1

    max_count = 0
    best_predicted_label = -1

    for predicted_label in label_counts:               # Voting classification loop
        if label_counts[predicted_label] > max_count:  # more counts == more votes
            max_count = label_counts[predicted_label]  
            best_predicted_label = predicted_label     # store most voted label
    
    return best_predicted_label # return prediction



def confidence_score(neighbors, predicted_label):
    """
    Takes the predicted label and neighbors as input
    Calculates a confidence score using distance weighting
    Closer neighbors have more influence on confidence score
    Returns confidence score
    Time complexity: O(k) --> k is the number of nearest neighbors in kNN which is 5
    Author: Zaid
    """

    weight_for_prediction = 0
    total_weight = 0
    
    epsilon = 0.000000001      # in case a neighbor has a distance of 0, preventing a division by zero error
    
    for neighbor in neighbors:
        distance = neighbor['distance']
        weight = 1 / (distance**2 + epsilon) # Calculate the weight, --> lower distance = greater weight
        
        total_weight += weight # Add to our total pool of weights
        
        if neighbor['label'] == predicted_label: # If this neighbor voted for our predicted label, add to the winner's pool
            weight_for_prediction += weight
            
    confidence_percentage = (weight_for_prediction / total_weight) * 100
    return confidence_percentage



def deduce_label(predicted_label):
    '''
    Takes the predicted label (0, 1, 2, 3, 4, 5) as input
    Decodes the predicted label into cancer risk level and tumor diagnosis.
    Prints the cancer risk level and tumor diagnosis
    Time complexity: O(1) 
    Author: Ahsan
    '''

    label_dict = {0: ['High', 'Benign'],    # dictionary to store labels
                1: ['High', 'Malignant'],
                2: ['Low', 'Benign'],
                3: ['Low', 'Malignant'], 
                4: ['Medium', 'Benign'], 
                5: ['Medium', 'Malignant']}

    cancer_risk = label_dict[predicted_label][0]
    tumor_diagnosis = label_dict[predicted_label][1]

    return cancer_risk, tumor_diagnosis



def insert_record(tree, point, label, new_id, features_file='features.csv', labels_file='labels.csv', ids_file='ids.csv'):
    """
    Appends the nsew patient data strictly to the bottom of the CSV files in O(1) time
    Insert the new patient into the KD-Tree
    Returns the updated KD-Tree
    Time complexity: O(logn)
    Author: Zaid
    """
    
    with open(ids_file, 'a', newline='') as ids:
        writer = csv.writer(ids)
        writer.writerow([new_id])
        
    with open(features_file, 'a', newline='') as features:
        writer = csv.writer(features)
        writer.writerow(point)
        
    with open(labels_file, 'a', newline='') as labels:
        writer = csv.writer(labels)
        writer.writerow(['', '', '', label])
        
    updated_tree = insert(tree, point, label) # insert point into the kdtree
    
    return updated_tree

