# The Machine Learning Pipeline in Scikit-learn

<br>

## Table of Contents

- <a href="#cleaning-preprocessing" style="color: #d4d4d4;">Cleaning and Pre-Processing</a>
- <a href="#feature-selection-dimensionality-reduction" style="color: #d4d4d4;">Feature Selection and Dimensionality Reduction</a>
- <a href="#splitting" style="color: #d4d4d4;">Splitting</a>
- <a href="#classification-regression" style="color: #d4d4d4;">Classification and Regression Algorithms</a>
- <a href="#hyperparameter-tuning" style="color: #d4d4d4;">Hyperparameter Tuning</a>

<br>

### **<a id="cleaning-preprocessing"></a>Cleaning and Pre-Processing**

In this section, we'll discuss missing values, feature engineering, and feature scaling.

<br>

**Missing Values**

The fancy word for fixing missing values is "imputation."

Questions to Ask: _Are there missing values in my dataset? Why are they missing? Should I remove these values or replace them? How should I replace them?_

How to Fix:

- Delete the column(s) with missing values.
- Delete the row(s) with missing values.
- Impute the missing values.
  - Imputing means filling in the values using a strategy.
  - There are four main strategies to fill in the values: mean, median, mode, and constant.
  - Pros: Easy and fast. Works well with small datasets.
  - Cons: Doesn't factor correlations between features. Not accurate. Doesn't account for uncertainty.
- Interpolate the missing values.
  - Interpolation helps estimate unknown values by using two known values.
- Fill in the missing values using a model.
  - If the values are in one column, you can use a model to fill them in.
  - Consider using linear regression or k-nearest neighbors.
  - Pros: Can be much more accurate than normal imputation.
  - Cons: Computationally more expensive. The model that you use can be sensitive to outliers.

<br>

**Feature Engineering**

The fancy word for updating/preparing columns for our model(s) is "feature engineering."

Feature engineering helps: Polynomial features, categorical features, and numerical features.

- Polynomial Features
  - Used to improve the model by adding more/better complexity to numeric columns.
  - One reason to do this is if the model is underfitting.
- Categorical Features
  - Need to be transformed to a numerical representation for the model to understand.
  - You should determine if each categorical feature is nominal or ordinal.
    - Nominal example: A “color” variable with values “red“, “green“, and “blue“.
    - Ordinal example: A “place” variable with values “first“, “second“, and “third“.
  - The three common ways to convert ordinal/nominal values to numerical values: Ordinal encoding, one-hot encoding, and dummy variable encoding.
    - Ordinal Encoding: Each unique category value is assigned an integer value.
    - One-Hot Encoding: The initial variable is removed and one new binary variable is added for each unique integer value in the variable.
    - Dummy Variable Encoding: Similar to One-Hot Encoding but with less redundancy.
- Numerical Features
  - Can be decoded to categorical values if necessary.
  - The two common ways to convert numerical values to categorical values: discretization and binarization.
    - Discretization ("binning") divides a continuous feature into a pre-specified number of categories/ bins.
    - Binarization is the process of tresholding numerical features to get boolean values.

**Feature Scaling**

The fancy word for aligning/calibrating data for our model(s) is "feature scaling."

! **Split your data into a train and test set _before_ any feature scaling** !

There are two primary ways of feature scaling: Standardization and Normalization.

- Standardization
  - Is used to scale data so it has a mean of 0 and standard deviation of 1. It's typically more useful for classification problems.
  - There is one type of Standardization: StandardScaler.
    - Standard Scaler: Is the main scaler. It uses a strict definition to center the data.
  - Pros:
    - All features have the same mean and variance. This makes it easy to compare.
    - Less sensitive to extreme outliers than the MinMax Scaler (in Normalization).
    - It preserves the initial distribution.
  - Cons:
    - Works best when the initial distribution is a normal distribution.
    - No fixed range between features.
    - Keeps outliers.
- Normalization
  - Is used to process data into a fixed range, such as [0 to 1] or [-1 to 1]. It's typically more useful for regression problems.
  - There are three types of Normalization: MinMax Scaler, MaxAbs Scaler, and Robust Scaler. Each type has unique Pros, but their Cons are all the same.
    - MinMax Scaler: Scales the data so it has a fixed range of [0 to 1].
      - Pro: Helpful when data is only positive.
    - MaxAbs Scaler: Scales the data so it has a fixed range of [-1 to 1].
      - Pro: Helpful when data is positive and negative.
    - Robust Scaler: Sets the median to 0 and scales the data using the inter-quartile range.
      - Pro: Helpful when data has extreme outliers. Produces a better spread.
  - Cons:
    - The mean and variance vary between features.
    - It may change the shape of the initial distribution.
    - It is sensitive to extreme outliers (Except for Robust Scaler).

<br>
<br>

### **<a id="feature-selection-dimensionality-reduction"></a>Feature Selection and Dimensionality Reduction**

In this section, we'll discuss forms of feature selection and dimensionality reduction.

<br>

! **Split your data into a train and test set _before_ any dimensionality reduction and feature selection** !

<br>

**Feature Selection**

The fancy word for the columns in your data is "features."

Feature selection allows you to automatically select the features in your data that contribute most to the prediction variable.

Pros: Reduces overfitting, improves accuracy, and reduces training.

There are three primary ways to do feature selection: Univariate Selection, Recursive Feature Elimination (RFE), and Feature Importance.

- Univariate Selection
  - Uses statistical tests to select the features that have the strongest relationship with the output variable.
- Recursive Feature Elimination (RFE):
  - Uses a user-defined model's accuracy to identify which features (and combination of features) contribute most to predicting the target attribute.
- Feature Importance:
  - Decision Trees, Random Forest, and Extra Trees can be used to estimate feature importance too.

**Dimensionality Reduction**

While feature selection simply selects and excludes features without changing them, dimensionality reduction transforms features.

There are six primary ways to do dimensionality reduction: Principal Component Analysis (PCA), Singular Value Decomposition (SVD), Linear Discriminant Analysis (LDA), Isomap Embedding (Isomap), Locally Linear Embedding (LLE), and Modified Locally Linear Embedding (MLLE).

- Principal Component Analysis (PCA)
  - PCA summarizes the feature set by focusing on maximum variance in the data without relying on the output.
  - Best used on data with few zero values. Unsupervised.
  - Kernel PCA is a subset of PCA used for non-linear relationships.
- Singular Value Decomposition (SVD)
  - SVD is very similar to PCA except it's best used on data with many zero values.
- Linear Discriminant Analysis (LDA)
  - LDA is a multi-class technique that tries to find a decision boundary around each cluster of a class.
- Isomap Embedding (Isomap)
  - Isomap is best used on non-linear data.
  - Uses KNN to find the lower dimensional embedding of the initial dataset while maintaining distances between all its initial points.
- Locally Linear Embedding (LLE)
  - LLE is similar to Isomap in that it uses KNN and is best used on non-linear data.
  - LLE is usually more efficient than Isomap because it eliminates the need to estimate pairwise distances.
- Modified Locally Linear Embedding (MLLE)
  - This is an extension of LLE that creates multiple weight vectors for each neighborhood.

<br>
<br>

### **<a id="splitting"></a>Splitting**

Splitting the data allows us to use some of the data for training the model and the other part to test the model's performance/ accuracy.

There are four primary ways to split the data: Train-Test Split, Stratified Train-Test Split, Train-Test-Validate Split, and Kfolds Cross-Validator.

- Train-Test Split
  - Used for supervised classification and regression algorithms.
  - Used to cut the data into two subsets: The train data and the test data.
    - The train data is used to fit the machine learning model.
    - The test data is used to evaluate the fit of the machine learning model.
    - Typically, the common percentage split is 80% train, 20% test or 67% train, 33% test.
- Stratified Train-Test Split
  - Used for classification problems. When there's not a balanced number of examples for each class label.
  - Similar process to Train-Test Split.
- Train-Test-Validate Split
  - Used for supervised classification and regression algorithms.
  - Used to provide an unbiased evaluation of a model fitted on the training dataset while tuning model hyperparameters.
  - Taken out of the training set and used during training to validate the model's accuracy.
  - The testing set is fully disconnected until the model is finished training.
  - The validation set is used to validate it during training.
- Kfolds Cross-Validator
  - Used for supervised classification and regression algorithms.
  - Divides a limited dataset into non-overlapping folds to be tested.
  - k=10 was found to provide good trade-off between low computational cost and low bias in an estimate of model performance.

<br>
<br>

### **<a id="classification-regression"></a>Classification and Regression Algorithms**

- Classifcation predicts the class or category for a given observation. For example, an email of text can be classified as belonging to one of two classes: “spam“ and “not spam“.

- Regression is a continuous, real-value variable, such as an integer or floating point value. These are often quantities, such as amounts and sizes. For example, a house may be predicted to sell for a specific dollar value, perhaps in the range of $100,000 to $200,000.

**Classification Algorithms**

We'll discuss seven classifcation algorithms: K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Tree Classifiers, Random Forests, Naive Bayes, Logistic Regression, and eXtreme Gradient Boosting (XGBoost).

- [K-Nearest Neighbors (KNN)](https://miro.medium.com/max/1182/0*sYMSaIon56Qng2hF.png): Calculates the distance of a new data point to all other training data points. Then, it assigns the data point to the class to which the majority of the K data points belong.
  - Pros:
    - Easy to implement.
    - It's a lazy learning algorithm, meaning it requires no training prior to making real time predictions.
    - Since it requires no training before making predictions, new data can be added seamlessly.
    - Only two parameters required to implement KNN. K and the distance function.
  - Cons:
    - Doesn't work well with high dimensional data (many features, few observations).
    - Has high prediction cost for large datasets.
    - Doesn't work well with categorical features since it is difficult to find the distance between dimensions with categorical features.

- [Support Vector Machines (SVM)](https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm.png): Finds the most optimal decision boundary, which is the one that has the maximum margin from the nearest points of all the classes.
  - Pros:
    - Works well with a clear margin of separation.
    - Effective in high dimensional spaces.
    - Effective in cases where the number of dimensions is greater than the number of samples.
    - Uses a subset of training points in the decision function, so it's memory efficient.
  - Cons:
    - Doesn’t perform well on large datasets because the required training time is higher.
    - Doesn’t perform well when data has a lot of noise i.e. target classes are overlapping.

- [Decision Tree](https://miro.medium.com/max/781/1*fGX0_gacojVa6-njlCrWZw.png)
  - Pros:
    - Used for both classification and regression problems.
    - Output can be easily understood.
    - One of the quickest ways to identify relationships between variables.
    - Not largely influenced by outliers or missing values, and can handle both numerical and categorical variables.
  - Cons:
    - Overfitting.
    - Doesn’t perform well on continuous numerical variables.
    - A small change in the data tends to cause a big difference in the tree structure, which causes instability.
    - Relatively expensive as the amount of time taken and the complexity levels are greater.

- [Random Forest](https://i.stack.imgur.com/zGFsi.pn): Random Forest is based on ensemble learning.
  - Ensemble learning is a type of learning where you join different types of algorithms or same algorithm multiple times to form a more powerful prediction model.
  - Random Forest combines multiple decision trees.
  - Pros:
    - Overall bias is reduced.
    - Works well when you have both categorical and numerical features.
    - Works well when data has missing values or it has not been scaled well
  - Cons
    - Complex. Requires much more computational resources, owing to the large number of decision trees joined together.

- [Naive Bayes](https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.05-gaussian-NB.png):
Helps determine the probability of an event occurring based on prior knowledge of conditions related to the event.
  - Formula: P(H|E) = (P(E|H) * P(H)) / P(E)
    - P(H|E) is the probability of hypothesis H given the event E, a posterior probability.
    - P(E|H) is the probability of event E given that the hypothesis H is true.
    - P(H) is the probability of hypothesis H being true (regardless of any related event), or prior probability of H.
    - P(E) is the probability of the event occurring (regardless of the hypothesis).
  - A classical use case of Naive bayes is document classification. Determining whether a given document corresponds to certain categories.
  - Pros:
    - Used for both classification and regression problems.
    - It might outperform more complex models when the amount of data is limited.
    - Works well with numerical and categorical data.
  - Cons:
    - Does not work well if you're missing certain combination of values in your training data.
    - Does not work well with feature relationships.

- [Logistic Regression](https://www.tibco.com/sites/tibco/files/media_entity/2020-09/logistic-regression-diagram.svg): Predicts the probability of an occurrence of a binary event using a logit function.
  - Pros:
    - Easy to implement, interpret, and efficient to train.
    - Can interpret model coefficients as indicators of feature importance.
    - Makes no assumptions about class distributions.
  - Cons:
    - May lead to overfitting if number of observations is less than the number of features.
    - Assumes linearity between the dependent variable and the independent variables.
    - Requires average or no multicollinearity between independent variables.

- [XGBoost](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/xgboost/img-3.png): Is a powerful ensemble algorithm that fits boosted decision trees by minimizing the error gradient.
  - Used for classification and regression algorithms.
  - There are many types of Gradient Boosting. XGBoost is the most popular. There are also: Gradient Boosting with LightGBM and Gradient Boosting with CatBoost
  - In general, Gradient Boosting provides hyperparameters that should, and perhaps must, be tuned for a specific dataset. The primary hyperparameters are:
    - The number of trees or estimators in the model.
    - The learning rate of the model.
    - The row and column sampling rate for stochastic models.
    - The maximum tree depth.
    - The minimum tree weight.
    - The regularization terms alpha and lambda.
  - Pros:
    - Computationally efficient and often has better model performance than other algorithms.
    - Doesn't require as much feature engineering.
    - Less prone to overfitting.
  - Cons:
    - Overfitting is possible if parameters aren't tuned correctly.

**Regression Algorithms**
