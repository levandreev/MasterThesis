import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
import itertools
import math

# Batches used for SVM
N = [10000,20000,30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

plot_f1 = []
mse_array = []
std_dev_array = []
std_error_array = []
plot_accuracy = []

def calculate_f1_array_per_dataset(slices):
#     classifier = GaussianNB()
    # classifier = SVC(kernel='rbf') # Gives warning
    classifier = LinearSVC(random_state=0, tol=1e-5, C=1)
    # classifier = KNeighborsClassifier(n_neighbors=3)
    corpus = []
    y = []
    plot_f1 = []
    with open('yelp_pol_nouns.csv', newline='') as csvfile:
        yelp = csv.reader(csvfile, delimiter=',')
        for slice in slices:
            for row in itertools.islice(yelp, slice):
                clean_row = row[1].strip().replace('"', '').replace(';', '')
                target_class = row[0]
                corpus.append(clean_row)
                y.append(int(target_class))
            tfidf_vectorizer = TfidfVectorizer(norm='l2', stop_words='english', ngram_range=(1, 1))
#             tfidf_vectorizer = TfidfVectorizer(norm='l2', ngram_range=(1, 1))
            matrix = tfidf_vectorizer.fit_transform(corpus)
            # sparse = tfidf_vectorizer.fit_transform(corpus).A
            # print(pd.DataFrame(matrix.todense(), columns=tfidf_vectorizer.get_feature_names()))
            print(slice)
            #Cross validation
            print("CROSS VALIDATION:")
            splits = 10
            kf_total = StratifiedKFold(n_splits=splits, shuffle=True)
            kf = KFold(n_splits=10)
            y1 = np.array(y)
            avg_f1 = 0
            avg_acc = 0
            avg_precision = 0
            avg_recall = 0
            avg_mse = 0
            avg_std_dev = 0
            avg_std_error = 0
            start = time.time()
            # for train_index, test_index in kf_total.split(sparse, y1):
            for train_index, test_index in kf_total.split(matrix, y1):
                # X_train, X_test = sparse[train_index], sparse[test_index]
                X_train, X_test = matrix[train_index], matrix[test_index]
                y_train, y_test = y1[train_index], y1[test_index]
#               use X_train and X_test .toarray() when using NB
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                f1 = f1_score(y_test, y_pred, average='macro')  # y_test = true y, y_pred = predicted y with the classifier
                accuracy = accuracy_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred);
                std_dev = np.std([y_test, y_pred])
                std_error = std_dev/math.sqrt(slice)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                print('F1 score: ', f1)
                print('Accuracy score: ', accuracy)
                print('Precision score: ', precision)
                print('Recall score: ', recall)
                print('Mean squared Error: ', mse)
                print('Standard Deviation: ', std_dev)
                print('Standard Error: ', std_error)
                print(classification_report(y_test, y_pred))
                avg_acc += accuracy
                avg_f1 += f1
                avg_precision += precision
                avg_recall += recall
                avg_mse += mse
                avg_std_dev += std_dev
                avg_std_error += std_error
            end = time.time()
            avg_recall = avg_recall / splits
            avg_precision = avg_precision / splits
            avg_f1 = avg_f1 / splits
            avg_acc = avg_acc / splits
            avg_mse = avg_mse / splits
            avg_std_dev = avg_std_dev / splits
            avg_std_error = avg_std_error / splits
            mse_array.append(avg_mse)
            std_dev_array.append(avg_std_dev)
            std_error_array.append(avg_std_error)
            print('----------------------')
            print('Average Accuracy:', avg_acc)
            print('Average Precision:', avg_precision)
            print('Average Recall:', avg_recall)
            print('Average F1', avg_f1)
            print('Average Mean Squared Error', avg_mse)
            print('Average Standard Deviation', avg_std_dev)
            print('Average Standard Error', avg_std_error)
            print('Runtime:', end - start)
            # return avg_f1
            plot_f1.append(avg_f1)
            plot_accuracy.append(avg_acc)
    return plot_f1


# for n in N:
#     plot_f1.append(calculate_avg_f1_per_dataset(n))

plot_f1 = calculate_f1_array_per_dataset(N)


# print('F1 values', plot_f1)
print('F1 values', plot_f1)
print('MSE values', mse_array)
print('Standatd Deviation values', std_dev_array)
print('Standatd Error values', std_error_array)
# plt.plot(plot_f1, 'ro')

d ={'F1': plot_f1, 'Standard Error': std_error_array, 'Accuracy': plot_accuracy}
df = pd.DataFrame(data=d)
df.to_csv('results_yelp_pol_nouns_svm1_stop_unigram.csv', index = False, header = True)

plt.errorbar(N, plot_f1, std_error_array, linestyle='None', marker='.')
plt.savefig('yelp_pol_nouns_svm1_stop_unigram_f1.png')
plt.clf()
plt.errorbar(N, plot_accuracy, std_error_array, linestyle='None', marker='.')
plt.savefig('yelp_pol_nouns_svm1_stop_unigram_acc.png')

