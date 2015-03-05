from string import punctuation, digits
import numpy as np
import matplotlib.pyplot as plt
import re

text_length_quantiles = 0
word_length_quantiles = 0
stop_word_quantiles = 0
mispelled_words_quantiles = 0

def read_data(filepath):
    """
    Returns an array of labels and an array of the corresponding texts
    """
    f = open(filepath, 'r')
    all_labels = []
    all_texts = []
    for line in f:
        label, text = line.split('\t')
        all_labels.append(int(label))
        all_texts.append(text)
    return (all_labels, all_texts)

def read_toy_data(filepath):
    """
    Returns (labels, data) for toy data
    """
    f = open(filepath, 'r')
    toy_labels = []
    toy_data = []
    for line in f:
        label, x, y = line.split('\t')
        toy_labels.append(int(label))
        toy_data.append([float(x), float(y)])
    return (toy_labels, np.array(toy_data))

def extract_words(input_string):
    """
      Returns a list of lowercase words in a string.
      Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    
    return input_string.lower().split()

def extract_dictionary(texts):
    """
      Given an array of texts, returns a dictionary of unigrams and bigrams.
      Each line is passed into extract_words, and a list on unique
      unigrams and bigrams is maintained.
      
      In addition, it computes quartiles of text length (in # of words) and word
      length (in # of characters).
    """
    global text_length_quantiles
    global word_length_quantiles
    global stop_word_quantiles
    global mispelled_words_quantiles

    words = [line.strip() for line in open('words.txt')]
    stopwords = [line.strip() for line in open('Stopwords.txt')]

    unigrams = []
    bigrams  = []
    text_lengths = []
    word_lengths = []
    stop_words = []
    mispelled_words = []
    for text in texts:
        word_list = extract_words(text)
        text_lengths.append(len(word_list))
        stop_words_number = 0
        mispellings = 0
        for i, word in enumerate(word_list):
            word_lengths.append(len(word))

            if(word in stopwords):
                stop_words_number = stop_words_number + 1

            if(word not in words):
                mispellings = mispellings + 1

            if(word not in unigrams):
                unigrams.append(word)

            if(i > 0):
                bigram = previous_word + '_' + word
                if(bigram not in bigrams):
                    bigrams.append(bigram)

            previous_word = word
        stop_words.append(stop_words_number)
        mispelled_words.append(mispellings)
    dictionary = unigrams + bigrams
    text_length_quantiles = [np.percentile(text_lengths, k) for k in [25,50,75]]
    word_length_quantiles = [np.percentile(word_lengths, k) for k in [25,50,75]]
    stop_word_quantiles = [np.percentile(stop_words, k) for k in [25,50,75]]
    mispelled_words_quantiles = [np.percentile(mispelled_words, k) for k in [25,50,75]]
    return dictionary

def extract_feature_vectors(texts, dictionary):
    """
      Returns the feature representation of the data.
      The returned matrix is of shape (n, m), where n is the number of texts
      and m the total number of features (entries in the dictionary and any 
      additional feature).
    """
    SAT_words = [line.strip() for line in open('SAT_words.txt')]
    words = [line.strip() for line in open('words.txt')]
    stopwords = [line.strip() for line in open('Stopwords.txt')]

    num_of_new_features = 15 # Number of features other than bag of words
    num_texts = len(texts)
    feature_matrix = np.zeros([num_texts, len(dictionary) + num_of_new_features])
        
    for i, text in enumerate(texts):
        #### Unigrams and bigrams
        word_list = extract_words(text)
        num_words = len(word_list)
        first_word_quintile = 0
        second_word_quintile = 0
        third_word_quintile = 0
        fourth_word_quintile = 0
        stop_words = 0
        mispellings = 0
        for j,word in enumerate(word_list):
            if(word in dictionary):
                feature_matrix[i, dictionary.index(word)] = 1
            if(j > 0):
                bigram = previous_word + '_' + word
                if(bigram in dictionary):
                    feature_matrix[i, dictionary.index(bigram)] = 1

            #if (word in SAT_words):
                #feature_matrix[i, len(dictionary) + SAT_words.index(word)] = 1

            if (len(word) < word_length_quantiles[0]):
                first_word_quintile = first_word_quintile + 1

            if (len(word) < word_length_quantiles[1] and len(word) > word_length_quantiles[0]):
                second_word_quintile = second_word_quintile + 1
            
            if (len(word) < word_length_quantiles[2] and len(word) > word_length_quantiles[1]):
                third_word_quintile = third_word_quintile + 1

            if (len(word) > word_length_quantiles[2]):
                fourth_word_quintile = fourth_word_quintile + 1   

            if (word in stopwords):
                stop_words = stop_words + 1  

            if (word not in words):
                mispellings = mispellings + 1      

            previous_word = word
        
        #### Additional Features
        # Binary features for text length
        feature_matrix[i, len(dictionary) + 0] = (num_words < text_length_quantiles[0]) # Bottom 25% 
        feature_matrix[i, len(dictionary) + 1] = (num_words < text_length_quantiles[1]) # Bottom 50% 
        feature_matrix[i, len(dictionary) + 2] = (num_words < text_length_quantiles[2]) # Bottom 75%
        # lengths = [len(w) for w in word_list]
        # feature_matrix[i, len(dictionary) + 3] = sum(lengths)/float(len(lengths)) # Average word length
        # feature_matrix[i, len(dictionary) + 4] = len(set(word_list)) # Unique words

        # feature_matrix[i, len(dictionary) + len (SAT_words) + 5] = (first_word_quintile >= second_word_quintile and first_word_quintile >= third_word_quintile and first_word_quintile >= fourth_word_quintile)  
        # feature_matrix[i, len(dictionary) + len (SAT_words) + 6] = (second_word_quintile >= first_word_quintile and second_word_quintile >= third_word_quintile and second_word_quintile >= fourth_word_quintile) 
        # feature_matrix[i, len(dictionary) + len (SAT_words) + 7] = (third_word_quintile >= second_word_quintile and third_word_quintile >= first_word_quintile and third_word_quintile >= fourth_word_quintile) 
        # feature_matrix[i, len(dictionary) + len (SAT_words) + 8] = (fourth_word_quintile >= second_word_quintile and fourth_word_quintile >= third_word_quintile and fourth_word_quintile >= first_word_quintile) 

        # feature_matrix[i, len(dictionary) + 9] = (stop_words < stop_word_quantiles[0])
        # feature_matrix[i, len(dictionary) + 10] = (stop_words < stop_word_quantiles[1])
        # feature_matrix[i, len(dictionary) + 11] = (stop_words < stop_word_quantiles[2])

        feature_matrix[i, len(dictionary) + 12] = (mispellings < mispelled_words_quantiles[0])
        feature_matrix[i, len(dictionary) + 13] = (mispellings < mispelled_words_quantiles[1])
        feature_matrix[i, len(dictionary) + 14] = (mispellings < mispelled_words_quantiles[2])        

        """
        TODO: try more features
        Remember to change variable 'num_of_new_features'!
        """
    
    return feature_matrix

def sgn(i):
    if (i <= 0):
        return -1
    else:
        return 1

def eta(x,y,theta,l):
    return min(1.0/l,loss(x,y,theta)/np.dot(x,x))

def loss(x,y,theta):
    return max (0,1-np.dot(y*theta,x))

def avg(theta,tc):
    return np.mean(theta[0:tc], axis=0)

def sgnresult(theta,theta_0,x):
    trans = np.transpose(theta)
    dot = np.dot(trans,x)
    return sgn(dot+theta_0) 

def baseperceptron(feature_matrix, labels, T, eta,l):
    x = feature_matrix
    y = labels
    M,height = feature_matrix.shape
    theta = np.empty([T*M,height])
    theta_0 = np.empty([T*M,1])
    theta[0] = 0
    theta_0[0] = 0 
    tc = 0
    for h in range (T):
        for i in range (M):
            if (tc == T*M-1):
                break
            if (sgnresult(theta[tc],theta_0[tc],x[i])!= y[i] or l != 0):
                theta[tc+1] = theta[tc] + eta(x[i],y[i],theta[tc],l)*np.dot(y[i],x[i])
                theta_0[tc+1] = theta_0[tc] + eta(x[i],y[i],theta[tc],l)*y[i]
                tc = tc + 1
    return (theta,theta_0,tc)

def perceptron(feature_matrix, labels, T):
    theta, theta_0, tc = baseperceptron(feature_matrix, labels, T,lambda x,y,theta,l:1, 0)
    return (theta[tc],theta_0[tc])

def avg_perceptron(feature_matrix, labels, T):
    theta, theta_0, tc = baseperceptron(feature_matrix, labels, T,lambda x,y,theta,l:1, 0)
    return (avg(theta,tc),theta_0[tc])

def avg_passive_aggressive(feature_matrix, labels, T, l):
    theta, theta_0, tc = baseperceptron(feature_matrix, labels, T,eta, l)
    return (avg(theta,tc),theta_0[tc])

def classify(feature_matrix, theta_0, theta_vector):
    labels = []
    M,height = feature_matrix.shape
    for i in range (M):
        labels.append(sgnresult(theta_vector,theta_0,feature_matrix[i]))
    return labels

def score_accuracy(predictions, true_labels):
    """
    Inputs:
        - predictions: array of length (n,1) containing 1s and -1s
        - true_labels: array of length (n,1) containing 1s and -1s
    Output:
        - percentage of correctly predicted labels
    """
    correct = 0
    for i in xrange(0, len(true_labels)):
        if(predictions[i] == true_labels[i]):
            correct = correct + 1
    
    percentage_correct = 100.0 * correct / len(true_labels)
    print("Method gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(true_labels)) + ").")
    return percentage_correct

    

def write_submit_predictions(labels,outfile,name,pseudonym="Anonymous"):
    """
      Outputs your label predictions to a given file.
      Prints name on the first row.
      labels must be a list of length 500, consisting only of 1 and -1
    """
    
    if(len(labels) != 500):
        print("Error - output vector should have length 500.")
        print("Aborting write.")
        return
    
    with open(outfile,'w') as f:
        f.write("%s\n" % name)
        f.write("%s\n" % pseudonym)
        for value in labels:
            if((value != -1.0) and (value != 1.0)):
                print("Invalid value in input vector.")
                print("Aborting write.")
                return
            else:
                f.write("%i\n" % value)
    print('Completed writing predictions successfully.')



def plot_2d_examples(feature_matrix, labels, theta_0, theta, title):
    """
      Uses Matplotlib to plot a set of labeled instances, and
      a decision boundary line.
      Inputs: an (m, 2) feature_matrix (m data points each with
      2 features), a length-m label vector, and hyper-plane
      parameters theta_0 and length-2 vector theta.
    """
    
    cols = []
    xs = []
    ys = []
    
    for i in xrange(0, len(labels)):
        if(labels[i] == 1):
            cols.append('b')
        else:
            cols.append('r')
        xs.append(feature_matrix[i][0])
        ys.append(feature_matrix[i][1])
    
    plt.scatter(xs, ys, s=40, c=cols)
    [xmin, xmax, ymin, ymax] = plt.axis()
    
    linex = []
    liney = []
    for x in np.linspace(xmin, xmax):
        linex.append(x)
        if(theta[1] != 0.0):
            y = (-theta_0 - theta[0]*x) / (theta[1])
            liney.append(y)
        else:
            liney.append(0)
    plt.suptitle(title, fontsize=15)
    plt.plot(linex, liney, 'k-')
    plt.show()


def plot_scores(parameter,parameter_values,train_scores,validation_scores,title):
    """
      Uses Matplotlib to plot scores as a function of hyperparameters.
      Inputs:
           - parameter:  string, one of 'Lambda' or 'Iterations'
           - parameter_values: a list n of parameter values
           - train_scores: a list of n scores on training data
           - validations:  a list of n scores on validation data
           - title: String
    """
    
    plt.plot(parameter_values,train_scores,'-o')
    plt.plot(parameter_values,validation_scores,'-o')
    plt.legend(['Training Set','Validation Set'], loc='upper right')
    plt.title(title)
    plt.xlabel('Hyperparameter: ' + parameter)
    plt.ylabel('Accuracy (%)')
    plt.show()
