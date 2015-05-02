import numpy as np
import project1_code as p1

output_file = "challenge_predictions.txt"
name = "Hikari Senju"
pseudonym = "radicaledward"

train_labels, train_texts = p1.read_data('train.tsv')
dictionary = p1.extract_dictionary(train_texts)
train_feature_matrix = p1.extract_feature_vectors(train_texts, dictionary)

T = 5
l = 1

print "Calculating model..."
theta, theta_0 = p1.avg_passive_aggressive(train_feature_matrix, train_labels, T,l)

dummy_labels, test_texts = p1.read_data('submit.tsv')
test_feature_matrix = p1.extract_feature_vectors(test_texts, dictionary)

print "Making predictions..."
predictions = p1.classify(test_feature_matrix, theta_0, theta)

print "Writing labels to output file: "+str(output_file)
p1.write_submit_predictions(predictions,output_file,name,pseudonym)