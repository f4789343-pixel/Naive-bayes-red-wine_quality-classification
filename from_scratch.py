import pandas as pd
import numpy as np

df = pd.read_csv('winequality-red.csv', sep=';')

print(df.columns)
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

X = df.drop(columns='quality')
y = df['quality']

print(X.shape)
print(y.shape)

np.random.seed(42)
indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

print(train_indices.shape)
print(test_indices.shape)

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print(type(x_train))
print(x_test.shape)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

def prior(y_train):
   classes, count = np.unique(y_train, return_counts=True)
   total = sum(count)
   priors = []
   for i in count:
      priors.append(i/total)
   return priors


def multi_features(x_train):
   all_values = []
   for feature_index in range(len(x_train[0])):
      values = []
      for i in range(len(x_train)):
         values.append(x_train[i][feature_index])
      all_values.append(values)
   return all_values
x = multi_features(x_train)


def mean(x_train):
   features = multi_features(x_train)
   means = []
   for feature in features:
      mean = sum(feature) / len(feature)
      means.append(mean)
   return means


def std(x_train):
   features = multi_features(x_train)
   means = mean(x_train)
   stds = []
   for feature_index in range(len(features)):
      d = 0
      for value in features[feature_index]:
         d += (value - means[feature_index])**2
      variance = d / len(features[feature_index])
      std = np.sqrt(variance)
      stds.append(std)
   return stds


def separate_classes(x_train, y_train):
   classes = np.unique(y_train)
   class_features = []
   for c in classes:
      features = []
      for i in range(len(y_train)):
         if y_train[i] == c:
            features.append(x_train[i])
      class_features.append(features)
   return classes, class_features


def class_mean(x_train, y_train):
   classes, class_features = separate_classes(x_train, y_train)
   classes_mean = []
   for features in class_features:
      class_means = mean(features)
      classes_mean.append(class_means)
   return classes, classes_mean

def class_std(x_train, y_train):
    classes, class_features = separate_classes(x_train, y_train)  
    classes_std = []
    for features in class_features:
       class_stds = std(features)
       classes_std.append(class_stds)
    return classes, classes_std
#print('stds:',class_std(x_train, y_train))
classes, class_features = separate_classes(x_train, y_train)

#print("classes:", classes)


def density(x_train, y_train, x_test):
   classes, classes_mean = class_mean(x_train, y_train)
   _, classes_std = class_std(x_train, y_train)
   all_densities = []
   for class_index in range(len(classes)):
      densities = []
      for feature_index in range(len(x_test)):
          mean = classes_mean[class_index][feature_index]
          std = classes_std[class_index][feature_index]
          deviation = (x_test[feature_index] - mean)**2 / (2*(std)**2)
          normalization = 1 / (std*np.sqrt(2*np.pi))
          e_exponent = np.exp(-deviation)
          density = normalization * e_exponent
          densities.append(density)
      all_densities.append(densities)
   return all_densities


def multiply_density(x_train, y_train, x_test):
    all_densities = density(x_train, y_train, x_test)
    likelihoods = []
    for class_densities in all_densities:
      likelihood = 1
      for d in class_densities:
         likelihood *= d
      likelihoods.append(likelihood)
    return likelihoods

def likelihoods(x_train, y_train, x_test):
    all_likelihoods = []

    for value in x_test:
      all_likelihoods.append(multiply_density(x_train, y_train, value))

    return all_likelihoods

def by_prior(x_train, y_train, x_test):
   priors = prior(y_train)
   likelihoods = multiply_density(x_train, y_train, x_test)
   scores = []
   for i in range(len(priors)):
      scores.append(priors[i]*likelihoods[i])
   return scores

def all_priors(x_train, y_train, x_test):
   all_scores = []
   for value in x_test:
      all_scores.append(by_prior(x_train, y_train, value))
   return all_scores

def prob(x_train,y_train,x_test):
   scores = all_priors(x_train, y_train, x_test)
   all_probabilities = []
   for sample_scores in scores:
      total = sum(sample_scores)
      probabilities = []
      for score in sample_scores:
          probabilities.append(score/total)
      all_probabilities.append(probabilities)
   return all_probabilities
print(prob(x_train,y_train,x_test))

def predict(x_train, y_train, x_test):

    probabilities = prob(x_train, y_train, x_test)
    classes, _ = class_mean(x_train, y_train)

    predictions = []

    for sample_probabilities in probabilities:
        max_index = np.argmax(sample_probabilities)
        predictions.append(classes[max_index])
    print(max_index)
    return predictions
predictions = predict(x_train, y_train, x_test)

#print(predictions)
#print(len(predictions))

def accuracy(y_test, predictions):

    correct = 0

    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            correct += 1

    return correct / len(y_test)
predictions = predict(x_train, y_train, x_test)

print("Accuracy:", accuracy(y_test, predictions))

def confusion_values(y_test, predictions, target_class):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for actual, predicted in zip(y_test,predictions):
      if actual == target_class and predicted == target_class:
          tp += 1
      elif actual != target_class and predicted != target_class:
         tn += 1
      elif actual != target_class and predicted == target_class:
         fp += 1
      else:
         fn += 1
    return tp,tn,fp,fn

classes = np.unique(y_test)

for target_class in classes:
   tp,tn,fp,fn = confusion_values(y_test,predictions, target_class)

   print(f"Class {target_class}: "f"TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}")

def precision_recall(y_test, predictions, target_class):
   tp,tn,fp,fn = confusion_values(y_test, predictions, target_class)
   if tp+fp == 0:
      precision = 0
   else:
      precision = tp / (tp + fp)

   if tp + fn == 0:
      recall = 0
   else:
      recall = tp / (tp + fn)
   return precision, recall

classes = np.unique(y_test)
for target_class in classes:
   precision,recall = precision_recall(y_test,predictions,target_class)

   print('class:', target_class)
   print('Precision:', precision)
   print('Recall:', recall)

def f1_score(y_test, predictions, target_class):

    precision, recall = precision_recall(y_test,predictions,target_class)
    if precision + recall == 0:
       return 0
    else:
       return 2 * (precision * recall) / (precision + recall)
       
f1_scores = []
for target_class in classes:
   f1 = f1_score(y_test,predictions,target_class)
   f1_scores.append(f"Class {target_class}: {f1}")
print('F1 scores:', f1_scores)

macro_sum = 0
for target_class in classes:
   f1 = f1_score(y_test,predictions,target_class)
   macro_sum += f1
macro_avg = macro_sum / len(classes)
print('average of f1:', macro_avg)

classes, counts = np.unique(y_test, return_counts=True)
total_count = sum(counts)
f1_weights = []
for target_class, count in zip(classes,counts):
   weight = count / total_count
   f1 = f1_score(y_test, predictions,target_class)
   f1_weighted = f1 * weight
   f1_weights.append(f1_weighted)
weighted_f1 = sum(f1_weights)
print("Weighted F1:", weighted_f1)

def confusion_matrix(y_test, predictions, classes):

    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for actual, predicted in zip(y_test, predictions):

        actual_index = np.where(classes == actual)[0][0]
        predicted_index = np.where(classes == predicted)[0][0]

        matrix[actual_index][predicted_index] += 1

    return matrix

classes = np.unique(y_test)

cm = confusion_matrix(y_test, predictions, classes)
print(cm)
