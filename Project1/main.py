import numpy as np
import project1_code as p1

#########################################################################
########################  PART 1: TOY EXAMPLE ###########################
#########################################################################

"""
TODO: 
    Implement the following functions in project1_code.py:
         1) perceptron 
         2) avg_perceptron 
         3) avg_passive_agressive 
"""

# Read data
toy_labels, toy_data = p1.read_toy_data('toy_data.tsv')

# Train classifiers
T = 5  # Choose values
l = 10
theta, theta_0 = p1.perceptron(toy_data, toy_labels, T)
p1.plot_2d_examples(toy_data, toy_labels, theta_0, theta, 'Perceptron')
theta, theta_0 = p1.avg_perceptron(toy_data, toy_labels, T)
p1.plot_2d_examples(toy_data, toy_labels, theta_0, theta, 'Averaged Perceptron')
theta, theta_0 = p1.avg_passive_aggressive(toy_data, toy_labels, T, l)
p1.plot_2d_examples(toy_data, toy_labels, theta_0, theta, 'Passive-Agressive')



##########################################################################
######################## PART 2 : ESSAY DATA #############################
#########################################################################


########## READ DATA ##########

#Training data
train_labels, train_text = p1.read_data('train.tsv')
dictionary = p1.extract_dictionary(train_text)
train_feature_matrix = p1.extract_feature_vectors(train_text, dictionary)

#Validation data
val_labels, val_text = p1.read_data('validation.tsv')
val_feature_matrix = p1.extract_feature_vectors(val_text, dictionary)

#Test data
test_labels, test_text = p1.read_data('test.tsv')
test_feature_matrix = p1.extract_feature_vectors(test_text, dictionary)


########## MODEL TRAINING ##########

"""
TODO:
 - Implement the following functions in the project1_code.py module:
      perceptron(features,labels,T)
      avg_perceptron(features,labels,T)
      avg_passive_aggressive(features,labels, T, l)
"""

T =  # maximum number of iterations through all data
l =  # lambda (used for passive-aggressive)

theta, theta_0 = p1.perceptron(train_feature_matrix, train_labels, T)
theta, theta_0 = p1.avg_perceptron(train_feature_matrix, train_labels, T)
theta, theta_0 = p1.avg_passive_aggressive(train_feature_matrix, train_labels, T, l)


########## VALIDATION ##########
"""
TODO:
1) For multiple values of T and lambda:
    - Predict labels for validation set (using function 'classify')
    - Calculate validation accuracy (using function 'score_accuracy')
2) Choose optimal learning method and parameters based on validation accuracy

"""



# ToDo: Choose optimal based on performance on validation set
T_optimal = 
lambda_optimal = 


########## TESTING ##########
optimal_theta, optimal_theta_0 = p1.avg_passive_aggressive(train_feature_matrix, train_labels, T_optimal, lambda_optimal)
predictions = p1.classify(test_feature_matrix, optimal_theta_0, optimal_theta)

print 'Performance on Test Set:'
test_score = p1.score_accuracy(predictions, test_labels)



"""
TODO:
 After you have found your best model/parameters/features, modify the 
 file best_learner.py accordingly. 
"""
