import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import cv2
# from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
# from scipy.spatial.distance import cdist


# def OptimumClusters(train_data_desc):
#         all_desc = []
#         for i in range(0, 60000):
#                 if(train_data_desc[i] is not None):
#                         for desc in train_data_desc[i]:
#                                 all_desc.append(desc)
#         all_desc = np.stack(all_desc)
#         wcss = []
#         for i in range(1, 201, 1): 
#                 kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 101)
#                 kmeans.fit(all_desc)
#                 wcss.append(kmeans.inertia_)
        
#         plt.plot(wcss)
#         plt.grid(True)
#         plt.show()


class KMeans:
        def __init__(self, K=128, max_iters=100):
                self.K = K
                self.max_iters = max_iters
                self.clusters = [[] for _ in range(self.K)]
                # the centers (mean feature vector) for each cluster
                self.centroids = []

        def fit(self, X):
                self.X = X
                self.n_samples, self.n_features = X.shape
                random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
                self.centroids = [self.X[idx] for idx in random_sample_idxs]
                # Optimize clusters
                print("_")
                for _ in range(self.max_iters):
                        # print(_)
                        # Assign samples to closest centroids (create clusters)
                        self.clusters = self._create_clusters(self.centroids)
                        # Calculate new centroids from the clusters
                        centroids_old = self.centroids
                        self.centroids = self._get_centroids(self.clusters)
                        
                        if self._is_converged(centroids_old, self.centroids):
                                break
                # Classify samples as the index of their clusters
                # return self._get_cluster_labels(self.clusters)
                return self.centroids
    
        def _get_cluster_labels(self, clusters):
                # each sample will get the label of the cluster it was assigned to
                labels = np.empty(self.n_samples)
                for cluster_idx, cluster in enumerate(clusters):
                        for sample_index in cluster:
                                labels[sample_index] = cluster_idx
                return labels

        def _create_clusters(self, centroids):
                # Assign the samples to the closest centroids to create clusters
                clusters = [[] for _ in range(self.K)]
                for idx, sample in enumerate(self.X):
                        centroid_idx = self._closest_centroid(sample, centroids)
                        clusters[centroid_idx].append(idx)
                return clusters

        def _closest_centroid(self, sample, centroids):
                # distance of the current sample to each centroid
                distances = np.linalg.norm(sample-centroids, axis=1)
                # distances = [euclidean_distance(sample, point) for point in centroids]
                closest_index = np.argmin(distances)
                return closest_index

        def _get_centroids(self, clusters):
                # assign mean value of clusters to centroids
                centroids = np.zeros((self.K, self.n_features))
                for cluster_idx, cluster in enumerate(clusters):
                        cluster_mean = np.mean(self.X[cluster], axis=0)
                        centroids[cluster_idx] = cluster_mean
                return centroids

        def _is_converged(self, centroids_old, centroids):
                # distances between each old and new centroids, fol all centroids
                # distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
                distances = np.linalg.norm(centroids_old-centroids)
                return np.sum(distances) == 0


def CreateVisualDictionary(train_data_desc):
        all_desc = []
        for i in range(0, 60000):
                if(train_data_desc[i] is not None):
                        for desc in train_data_desc[i]:
                                all_desc.append(desc)
        all_desc = np.stack(all_desc)

        # clusters
        k = 128
        kmeans = KMeans(k)
        codebook = kmeans.fit(all_desc)
        # kmeans = KMeans(n_clusters=k)
        # kmeans.fit(all_desc)
        # codebook = kmeans.cluster_centers_

        return codebook # visual dictionary matrix



def ComputeHistogram(feature_vector, visual_dictionary_matrix):
        visual_word = np.empty(feature_vector.shape[0])
        # for each image, map each descriptor to the nearest codebook entry
        if(feature_vector is not None):
                img_visual_words, distance = vq(feature_vector, visual_dictionary_matrix)
                visual_word = (img_visual_words)
        else:
                return
        # frequency vector
        img_frequency_vector = np.zeros(k)
        for word in visual_word:
                img_frequency_vector[word] += 1
        # histogram
        # plt.bar(list(range(10)), img_frequency_vector)
        # plt.show()
        return img_frequency_vector


def MatchHistogram(hist, train_hist):
        # cosine similarity
        cos_sim = np.dot(hist, train_hist.T)/(np.linalg.norm(hist)*np.linalg.norm(train_hist, axis=1))
        idx = np.argsort(-cos_sim)[:1][0]
        return labels[idx]


if __name__ == '__main__':
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0)
        # extractor = cv2.xfeatures2d.SIFT_create() # alternative

        train_data_key = {}
        train_data_desc = {}
        for i in range(0, train_labels.shape[0]):
                img = train_images[i]
                kp, des = orb.detectAndCompute(img, None)
                train_data_key[i] = (kp)
                train_data_desc[i] = (des)
        
        np.random.seed(101)     
        k = 128 # clusters
        codebook = CreateVisualDictionary(train_data_desc)
        # np.savetxt('codebook.txt', codebook)

        # directly load codebook to save computation time and comment above funcntion call
        # to call CreateVisualDictionary faster, comment above function
        # codebook = np.loadtxt('codebook.txt')

        all_desc_label = {}
        for i in range(0, 60000):
                if(train_data_desc[i] is not None):
                        all_desc_label[i] = (train_labels[i])

        global_desc = []
        labels = []

        for i, desc in train_data_desc.items():
                if(desc is not None):
                        global_desc.append(desc)
                        labels.append(all_desc_label[i])
        
        # Computing histogram for Train data
        frequency_vectors = []
        for img_desc in global_desc:
                img_frequency_vector = ComputeHistogram(img_desc, codebook)
                frequency_vectors.append(img_frequency_vector)
        frequency_vectors = np.stack(frequency_vectors)

        # normalizing the histogram
        df = np.sum(frequency_vectors > 0, axis=0)
        idf = np.log(60000/df)
        tfidf = frequency_vectors * idf

        
        gt_desc = []

        for i in range(0, test_labels.shape[0]):
                img = test_images[i]
                keypt, desc = orb.detectAndCompute(img, None)
                if desc is not None:
                        gt_desc.append(desc)
        
        # Computing histogram for Train data
        ft_vectors = []
        for img_desc in gt_desc:
                img_frequency_vector = ComputeHistogram(img_desc, codebook)
                ft_vectors.append(img_frequency_vector)

        ft_vectors = np.stack(ft_vectors)

        # normalizing the histogram
        dft = np.sum(ft_vectors > 0, axis=0)
        idft = np.log(10000/dft)
        tfidft = ft_vectors * idft

        
        test_desc = []
        label_test = []
        for i in range(0, test_labels.shape[0]):
                img = test_images[i]
                keypt, desc = orb.detectAndCompute(img, None)
                test_desc.append(desc)
                if desc is not None:
                        label_test.append(test_labels[i])

        # prediction

        label_predict = []
        class_predict = []
        print("Predictions for Test Images: ")
        for i in range(0, tfidft.shape[0]):
                label = MatchHistogram(tfidft[i], tfidf)
                print(f"{i} : {label} - {class_names[label]}")
                label_predict.append(label)
                class_predict.append(class_names[label])

        print(classification_report(label_test, label_predict))
        
        # met = (classification_report(label_test, label_predict, output_dict=True))
        # df_cp = pd.DataFrame(met).transpose()
        # df_cp.to_csv('classification_report.csv') 
