import csv
import sys
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import matplotlib.cm as cm
import random


# returns the feature set in a numpy ndarray
def loadCSV(filename):
    stuff = np.asarray(np.loadtxt(open(filename, 'rb'), delimiter=','))
    return stuff


# returns list of artist names
def getArtists(directory):
    return [name for name in os.listdir(directory)]


# loads all image files into memory
def loadImages():
    image_files = [f for f in listdir('../artworks_ordered_50') if f.endswith('.png')]
    images = []
    for f in image_files:
        images.append(mpimg.imread(os.path.join('../artworks_ordered_50', f)))
    return images

        
# convert color image to grayscale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


# creates a feature matrix using raw pixel values from all images, one image per row
def loadPixelFeatures():
    images = loadImages()
    X = []
    for img in images:
        img = rgb2gray(img)
        img = img.flatten()
        X.append(img)
    return np.array(X)


def ml_compute_eigenvectors_SVD(X,m):
    left, s, right = np.linalg.svd(np.matrix(X))    
    U = np.matrix.getA(right)    
    return (U[0:m])


#Colour function: helper function for plot_2D_clusters
def clr_function(labels):
    colors = []
    for i in range(len(labels)):
        if(labels[i] == 0):
            color = 'red'
        elif(labels[i] == 1):
             color = 'blue'
        elif(labels[i] == 2):
            color = 'green'
        elif(labels[i] == 3):
            color = 'yellow'
        elif(labels[i] == 4):
            color = 'orange'
        elif(labels[i] == 5):
            color = 'purple'
        elif(labels[i] == 6):
            color = 'greenyellow'
        elif(labels[i] == 7):
            color = 'brown'
        elif(labels[i] == 8):
            color = 'pink'
        elif(labels[i] == 9):
            color = 'silver'
        else:
            color = 'black'                
        colors.append(color)
    return colors


#Plot clusters of points in 2D
def plot_2D_clusters(X, clusterAssignments, cluster_centers):    
    
    points = X
    labels = clusterAssignments
    centers = cluster_centers
            
#    points = X.tolist()
#    labels = clusterAssignments.tolist()
#    centers = cluster_centers.tolist()
                                            
    N = len(points)
    K = len(centers)
    x_cors = []
    y_cors = []
    for i in range(N):
        x_cors.append( points[i][0] )
        y_cors.append( points[i][1] )
            
    plt.scatter(x_cors[0:N], y_cors[0:N], color = clr_function(labels[0:N]))                    
    plt.title('2D toy-data clustering visualization')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')    

    x_centers = [0]* K
    y_centers = [0]* K    
    for j in range(K):
        x_centers[j] = centers[j][0]
        y_centers[j] = centers[j][1]
        
    plt.scatter(x_centers, y_centers, color = 'black', marker = ',')
    plt.grid(True)
    plt.show()
    return


#Plot original and reconstructed points in 2D 
def plot_pca(X_original, X_recon):
    x_orig = []
    y_orig = []
    x_cors = []
    y_cors = []
    for i in range(len(X_original)):
        x_orig.append( X_original[i][0] )
        y_orig.append( X_original[i][1] )        
        x_cors.append( X_recon[i][0] )
        y_cors.append( X_recon[i][1] )                
    plt.title('2D toy-data PCA visualization')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')   
    plt.axis('equal')   #Suggestion: Try removing this command and see what happens!
                            
    plt.scatter(x_orig, y_orig, color = 'red' )    
    plt.scatter(x_cors, y_cors, color = 'green', marker = ',')        
    plt.grid(True)    
    plt.show()
    return            


# display paintings by artist, one artist per matplotlib figure
def plotArtworks():
    artists = getArtists('../selected_subset')
    figure_count = 0
    for artist in artists:
        artist_dir = os.path.join('../', 'selected_subset', artist)
        image_files = [f for f in listdir(artist_dir) if f.endswith('.png')]
        print image_files
        n_row = math.floor(math.sqrt(len(image_files)))
        n_col = math.ceil(len(image_files)/n_row)
        fig = plt.figure(figure_count)
        fig.canvas.set_window_title(artist)
        for i in range(len(image_files)):
            plt.subplot(n_row, n_col,i+1)
            plt.axis('off')
            plt.imshow(mpimg.imread(os.path.join(artist_dir, image_files[i])))
        figure_count += 1
    plt.show()


# creates a dictionary mapping cluster label to indices of X that belong to that cluster
def create_cluster_dict(cluster_labels):
    clusters = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in clusters.keys():
            clusters[cluster_labels[i]] = [i]
        else:
            clusters[cluster_labels[i]].append(i)
    return clusters


# plots clusters of images
def plotClusters(cluster_labels):
    ordered_artist_dir = os.path.join('../', 'artworks_ordered_50')
    image_files = [f for f in listdir(ordered_artist_dir) if f.endswith('.png')]
    clusters = create_cluster_dict(cluster_labels)
    figure_count = 0
    for key in clusters.keys():
        n_row = math.floor(math.sqrt(len(clusters[key])))
        n_col = math.ceil(len(clusters[key])/n_row)
        fig = plt.figure(figure_count)
        fig.canvas.set_window_title(str(key))
        for i in range(len(clusters[key])):
            plt.subplot(n_row, n_col, i+1)
            plt.axis('off')
            plt.imshow(mpimg.imread(os.path.join(ordered_artist_dir, image_files[clusters[key][i]])))
        figure_count += 1
    plt.show()


# displays images specified in labeled.csv after reconstruction (grayscale)
# Input:
    # matrix of pixel values, one image per row
# Output:
    # plot of the selected images in labeled.csv
def plotGallery(reconstruction):
    ordered_artist_dir = os.path.join('../', 'artworks_ordered_50')
    image_files = [f for f in listdir(ordered_artist_dir) if f.endswith('.png')]
    indices = loadCSV('labeled.csv')
    num_images = len(indices)
    n_row = math.floor(math.sqrt(num_images))
    n_col = math.ceil(num_images/n_row)
    for i in range(indices.shape[0]):
        plt.subplot(10, 5, i+1)
        plt.axis('off')
        img = np.reshape(reconstruction[int(indices[i])-1], (50,50))
        plt.imshow(img, cmap=cm.gray)
    plt.show()


# computes the majority vote for each cluster
# Input:
    # list of labels of each row in the feature matrix
    # fraction of data points that are labeled, defaults to 1 (all points have labels)
# Output:
    # a dictionary, with (key, value) = (cluster_label, majority)
def majorityVote(cluster_labels, labeled = 100):
    artist_labels = loadCSV('artist_labels_' + str(labeled) + '.csv')
    clusters = create_cluster_dict(cluster_labels)
    majorities = {} 
    for key in clusters.keys():
        votes = []
        for i in range(len(clusters[key])):
            label = artist_labels[clusters[key][i]]
            if label != -1:
                votes.append(label)
        if len(votes) == 0:
            votes.append(-1)
        votes = np.array(votes)
        majorities[key] = stats.mode(votes)[0][0]
    return majorities


# returns the total number of classification errors, comparing the majority vote label to true label
def computeClusterPurity(cluster_labels, majorities=None):
    if majorities == None:
        majorities = majorityVote(cluster_labels)
    artist_labels = loadCSV('artist_labels.csv')
    clusters = create_cluster_dict(cluster_labels)
    errors = 0 
    for key in clusters.keys():
        majority = majorities[key]
        for i in range(len(clusters[key])):
            if artist_labels[clusters[key][i]] != majority:
                errors += 1
    return 1-(float(errors)/float(len(cluster_labels)))


# computes the majority vote for each cluster
# Input:
    # list of labels of each row in the feature matrix
    # fraction of data you have labeled (valid inputs: 5, 15, 25, 50, 75, 100)
# Output:
    # classification accuracy
def classifyUnlabeledData(cluster_labels, labeled):
    majorities = majorityVote(cluster_labels, labeled)
    acc = computeClusterPurity(cluster_labels, majorities)
    return acc

# computes the maximum pairwise distance within a cluster
def intraclusterDist(cluster_values):
    max_dist = 0.0 
    for i in range(len(cluster_values)):
        for j in range(len(cluster_values)):
            dist = np.linalg.norm(cluster_values[i]-cluster_values[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


# helper function for Dunn Index
def index_to_values(indices, dataset):
    output = []
    for index in indices:
        output.append(dataset[index])
    return np.matrix(output)


# computes the Dunn Index, as specified in the project description
# Input:
    # cluster_centers - list of cluster centroids
    # cluster_labels - list of labels of each row in feature matrix
    # features - feature matrix 
# Output:
    # dunn index (float)
def computeDunnIndex(cluster_centers, cluster_labels, features):  
    clusters = create_cluster_dict(cluster_labels)
    index = float('inf')  
    max_intra_dist = 0.0
    # find maximum intracluster distance across all clusters
    for i in range(len(cluster_centers)):
        cluster_values = index_to_values(clusters[i], features)
        intracluster_d = float(intraclusterDist(cluster_values))
        if intracluster_d > max_intra_dist:
            max_intra_dist = intracluster_d

    # perform minimization of ratio
    for i in range(len(cluster_centers)):
        inner_min = float('inf')
        for j in range(len(cluster_centers)):
            if i != j:
                intercluster_d = float(np.linalg.norm(cluster_centers[i]-cluster_centers[j]))
                ratio = intercluster_d/max_intra_dist
                if ratio < inner_min:
                    inner_min = ratio
        if inner_min < index:
            index = inner_min
    return index

#helper function for init_medoids_plus
def weighted_choice(weights):
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i

#helper function for init_medoids_plus
def square_distance(point1, point2):
    value = 0.0    
    for i in range(0,len(point1)):
        value += (point1[i] - point2[i])**2    
    return value

#Function for generating initial centers uniformly at random (without replacement) from the data
def init_medoids(X, K):        
    points = X
    N = len(points)
    indices = range(0,N)
    centers = list([0]*K)
        
    for j in range(0,K):            
        temp = random.randrange(0,N-j)       
        centers[j] = indices[temp]
        del indices[temp]    

    medoids = []        
                        
    for j in range(0,K):
        medoids.append(points[centers[j]])
    
    return medoids


#Function for generating initial centers according to the KMeans++ initializer
def init_medoids_plus(X, K):        
    points = X
    N = len(points)
    indices = range(0,N)
    medoids = []
    weights = []
    
    #initialize first medoid
    temp = random.randrange(0,N)       
    medoids.append(list(points[indices[temp]]))
    del indices[temp]
    
    for i in range(len(indices)):
        weights.append(square_distance(medoids[0],points[indices[i]]))        
            
    
    for j in range(0,K):            
        if(j == 0):
            continue
               
        for i in range(len(indices)):
            c = medoids[j-1]
            if(square_distance(c, points[indices[i]]) < weights[i]):
                weights[i] = square_distance(c, points[indices[i]])
                                        
        temp = weighted_choice(weights)
        medoids.append(list(points[indices[temp]]))    
        del indices[temp]
        del weights[temp]
         
    #return np.array(medoids)
    return medoids
            

def ml_split(X):
    M,height = X.shape
    X_centered = np.empty([M,height])
    mean = np.empty([height,1])

    for i in range (height):
        mean[i] = np.average(X[:,i]);
    for j in range (height):
        for i in range (M):
            X_centered[i,j] = X[i,j] - mean[j]
    return X_centered,mean
    #pass
    # YOUR CODE HERE
    
    
def ml_compute_eigenvectors(X,m):
    C = np.dot(np.transpose(X),X)
    eigenValues,eigenVectors = np.linalg.eig(C)

    idx = eigenValues.argsort()[-m:][::-1]
    return np.transpose(eigenVectors[:,idx])
    #pass
    # YOUR CODE HERE


def ml_pca(X, U):
    return np.dot(X,np.transpose(U))
    #pass
    # YOUR CODE HERE


def ml_reconstruct(U, E, mean):
    product = np.dot(E, U)
    n, d = product.shape
    for i in range (n):
        product[i] = product[i] + np.concatenate(mean,axis=0)
    return product

# data = loadCSV('toy_pca_data.csv')
# X_centered,mean = ml_split(data)
# U = ml_compute_eigenvectors(X_centered,1)
# E = ml_pca(X_centered, U)
# X_recon = ml_reconstruct(U, E, mean)
# plot_pca(data, X_recon)

def ml_k_means(X, K, init):
    n, d = X.shape
    centroids = np.empty([K,d])
    centroids[:] = init
    clusterAssignments = np.empty([n,1])
    for i in range (50):
        for j in range (n):
            mindist = sys.maxint
            cluster = 0
            for k in range (K):
                dist = np.linalg.norm(X[j]-init[k])
                if (dist<mindist):
                    mindist = dist
                    cluster = k
            clusterAssignments[j] = cluster
        for k in range (K):
            summation = np.empty([n,d])
            for j in range (n):
                if (clusterAssignments[j] == k):
                    summation[j] = X[j]
            centroids[k] = summation.mean(0);
    return (centroids, clusterAssignments)

def ml_k_medoids(X, K, init):
    n, d = X.shape
    centroids = np.empty([K,d])
    centroids[:] = init
    clusterAssignments = np.empty([n,1])
    for i in range (50):
        for j in range (n):
            mindist = sys.maxint
            cluster = 0
            for k in range (K):
                dist = np.linalg.norm(X[j]-init[k])
                if (dist<mindist):
                    mindist = dist
                    cluster = k
            clusterAssignments[j] = cluster
        for k in range (K):
            mindist = sys.maxint
            for j in range (n):
                if (clusterAssignments[j] == k):
                    distsum = 0
                    for i in range (n):
                        if (clusterAssignments[i] == k):
                            distsum = distsum + np.linalg.norm(X[j]-X[i])
                    if (distsum<mindist):
                        centroids[k] = X[j]
                        mindist = distsum
    return (centroids, clusterAssignments)

X = loadCSV('toy_cluster_data.csv')
k = [2, 3, 4]
for i in range (4):
    cluster_centers, clusterAssignments = ml_k_medoids(X, k[i],X[0:k[i]:1])
    plot_2D_clusters(X, clusterAssignments, cluster_centers);
    