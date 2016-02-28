# Coverage Based Resampling implementation for Weka


###1. Algorithm Name
 Coverage-Based Resampling

###2. Reference
Ibarguren, I., Pérez, M., Muguerza, J., Gurrutxaga, I., Arbelaitz, O.
Coverage based resampling: building robust consolidated decision trees. Knowl.-Based Syst. (2015)

###3. Implemented By
Itay Hazan and Andrey Finkelstein
ISE Dept. Ben-Gurion University of the Negev, Israel

###4. Motivation for the algorithm
Class imbalance in datasets often appears in many real world problems and impose a great difficulty on the classification task for many learning algorithms (e.g. Decision Trees). Therefore a variety of techniques were suggest to handle this problem. These techniques can be separated into two main groups: oversampling and under sampling. 
In oversampling the idea is to create more instances of the minority class in order to balance the dataset – the problem is that the new instances are created from the existing data meaning that new information about the class will not be provided or it may not represent it well. In addition, techniques such as SMOTE result extra running time because of the instance creation and usually larger training time (larger training set). Under sampling on the other hand removes instances from the classes that are not the minority class in order to create a balanced training set. The process of under sampling is usually quick and the training time is short. However, removing instances from the training set will lead to information loss and sometimes poor performance. 
The Coverage-Based Resampling algorithm uses under sampling together with ensemble techniques in order to train classifiers on balanced training set with a minimal loss of information.  
 
###5. Short Description:
The Coverage-Based Resampling algorithm creates balanced datasets by resampling the training set. Each dataset is created from all the minority class instances and randomly selected instances from other classes (without replacement in the dataset). 
The number of datasets that are created depends on the imbalance rate in the original training set and on the number of instances from the majority class that we want to use (coverage rate). The algorithm creates the dataset independently and therefore the coverage is defined by the probability of an instance from the majority class to be selected at least to one dataset. After creating the datasets the algorithm uses a base classifier to train a model for each dataset. The classification is made by summing the probabilities outputted from the models for each class and selecting the class with the highest probability summation.

