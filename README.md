# The Machine Learning Pipeline in Scikit-learn

<br>

## Table of Contents

- <a href="#cleaning-preprocessing" style="color: #d4d4d4;">Cleaning and Pre-Processing</a>
- <a href="#feature-selection-dimensionality-reduction" style="color: #d4d4d4;">Feature Selection and Dimensionality Reduction</a>
- <a href="#splitting" style="color: #d4d4d4;">Splitting</a>
- <a href="#classification-regression" style="color: #d4d4d4;">Classification and Regression Algorithms</a>
- <a href="#compare-metrics-output" style="color: #d4d4d4;">Compare Metrics and Output</a>
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

- [Support Vector Machine Classification (SVM)](https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm.png): Finds the most optimal decision boundary, which is the one that has the maximum margin from the nearest points of all the classes.
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

We'll discuss seven regression algorithms: Linear Regression, Stochastic Gradient Descent (SGD), Ridge Regression, Elastic Net, Bayesian Ridge, Support Vector Machine (SVM), and eXtreme Gradient Boosting (XGBoost).

- [Linear Regression](https://miro.medium.com/max/1400/1*Cw5ZSYDkIFpmhBwr-hN84A.png): Gives us the most optimal value for the intercept by fitting multiple lines on the data points and returning the line that results in the least error.
  - Pros:
    - Simple to implement and easier to interpret the output coefficients.
    - Best to use if you know the relationship between the independent and dependent variable is linear.
    - Susceptible to over-fitting but can be avoided by using dimensionality reduction techniques.
  - Cons:
    - Outliers can have huge effects on the regression.
    - Assumes a linear relationship between dependent and independent variables and independence between attributes.

- [Stochastic Gradient Descent (SGD)](https://miro.medium.com/max/1005/1*_6TVU8yGpXNYDkkpOfnJ6Q.png): Is a simple yet efficient optimization algorithm used to find the values of parameters/coefficients of functions that minimize a cost function. It supports various loss functions and penalties to fit linear regression models.
  - Used for classification and regression algorithms.
  - There are many types of Gradient Descent. There are also: Batch Gradient Descent and Mini-Batch Gradient Descent
  - Pros:
    - One observation is processed by the network, so it's easier to fit into memory.
    - On a large dataset, it's likely to reach near the minimum faster than other SGDs.
    - Frequent updates create plenty of oscillations which can be helpful for getting out of local minimums.
  - Cons:
    - Can veer off in the wrong direction due to frequent updates.
    - Lose the benefits of vectorization since we process one observation per time.
    - Frequent updates are computationally expensive.
- [Ridge Regression](https://i.stack.imgur.com/s71QZ.png): Used to analyze multiple regression data that's multicollinear.
  - Multicollinearity occurs when there are high correlations between more than two predicted variables.
  - Multicollinearity can cause inaccurate results and p-values, making the model more redundant and reducing its efficiency and predictability.
  - Pros:
    - Protects model from overfitting.
    - Performs well when there's large, multivariate data with the number of predictors larger than the number of observations.
    - Very effective when it comes to improving the least-squares estimate where there's multicollinearity.
  - Cons:
    - Not capable of performing feature selection.
    - Shrinks coefficients towards zero.
    - Trades variance for bias.
- [Elastic Net](https://cdn.corporatefinanceinstitute.com/assets/elastic-net1-1024x642.png): An extension of linear regression that adds regularization penalties to the loss function during training.
  - Assumes a linear relationship between input variables and the target variable.
  - Elastic Net Regression = Lasso Regression + Ridge Regression.
  - Pros:
    - Better in handling collinearity and handling bias.
    - Performs well with complexity.
  - Cons:
    - Computational cost. Need to cross-validate the relative weight of L1 vs. L2 penalty.
    - Flexibility of the estimator. With greater flexibility comes increased probability of overfitting.
- [Bayesian Ridge](https://scikit-learn.org/stable/_images/sphx_glr_plot_bayesian_ridge_001.png): Applies Ridge regression and its coefficients under the Gaussian distribution.
  - Compared to the OLS (ordinary least squares), the coefficient weights are slightly shifted toward zeros.
  - Can be useful when there's insufficient data in the dataset or the data is poorly distributed.
  - The output and the model parameters are also assumed to come from a Gaussian distribution.
  - Pros:
    - Effective when the size of the dataset is small.
    - Particularly well-suited for real-time data based learning.
  - Cons:
    - Inference of the model can be time-consuming.
    - Is not worth doing if there's a large amount of data available in our dataset.
- [Support Vector Machine Regression (SVR)](https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm.png): Gives flexibility to define how much error is acceptable in our model and will find an appropriate line (or hyperplane in higher dimensions) to fit the data.
  - The objective of SVR is to minimize the coefficients (specifically l2-norm of the coefficient vector).
  - SVR pros and cons are similar to SVM.
  - Pros:
    - Works well with a clear margin of separation.
    - Effective in high dimensional spaces.
    - Effective in cases where the number of dimensions is greater than the number of samples.
    - Uses a subset of training points in the decision function, so it's memory efficient.
  - Cons:
    - Doesn’t perform well on large datasets because the required training time is higher.
    - Doesn’t perform well when data has a lot of noise i.e. target classes are overlapping.
- [XGBoost](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/xgboost/img-3.png): XGBoost is an efficient implementation of gradient boosting that can be used for regression predictive modeling.
  - Gradient boosting is a class of ensemble machine learning algorithms used for classification and regression problems.
  - Ensembles are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models.
  - Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm.
  - Use XGBoost for execution speed and model performance.
  - Dominates structured or tabular datasets.
  - XGBoost Regression pros and cons are similar to XGBoost Classification.
  - Pros:
    - Computationally efficient and often has better model performance than other algorithms.
    - Doesn't require as much feature engineering.
    - Less prone to overfitting.
  - Cons:
    - Overfitting is possible if parameters aren't tuned correctly.

<br>

### **<a id="compare-metrics-output"></a>Compare Metrics, Output, and Multicollinearity**

**Multicollinearity and Variance Inflation Factor**

- Multicollinearity is when one independent variable in a regression model is linearly correlated with another independent variable.
- This is bad because:
  - Fitted regression coefficients could change significantly if one of those variables changes.
  - It will be hard to detect statistical significance.
  - Makes prediction less accurate.
- To detect multicollinearity, use variance inflation factor (VIF).
  - VIF measures the ratio between one variance for a given regression coefficient to all other variances.
  - A VIF of 1 means the tested predictor is not correlated with other predictors.
  - A higher VIF means:
    - A predictor is more correlated with other predictors.
    - The standard error is inflated.
    - Larger the confidence interval.
    - Less likely that a coefficient will be statistically significant.

**Classification Metrics**

We'll discuss four [classification metrics](https://scikit-learn.org/stable/modules/classes.html#classification-metrics): Accuracy, confusion matrix (including Precision-recall and F1 Score), AUC-ROC, and log-loss.

- Accuracy: Measures how often the classifier predicts correctly.
  - Useful when the target class is well balanced but is not a good choice for the unbalanced classes.
  - Accuracy = number of correct predictions / total number of predictions
- Confusion Matrix: A table that's used to describe the performance of a classification model on the test data.
  - True Positives, False Positives (Type 1 Error), False Negatives (Type 2 Error), True Negatives.
- Precision: Explains how many of the correctly predicted cases actually were positive.
  - Useful when False Positives are higher concern than False Negatives.
  - Precision = number of true positives / number of predicted positives
- Recall: Explains how many of the actual positive cases we predicted correctly.
  - Useful when False Negatives are higher concern than False Positives.
  - Recall = number of true positives / total number of actual positives
- F1 Score: Gives a combined Precision and Recall score. Maximum when Precision is equal to Recall.
  - Useful when False Positives and False Negatives are equally bad, adding more data doesn't help the results, and/or true negatives are high.
- AUC-ROC
  - Receiver Operator Characteristic (ROC) is a probability curve that plots the True Positive Rate against the False Positive Rate at various threshold values to separate the signal from the noise.
  - The Area Under the Curve (AUC) is the measure of the classifier to distinguish classes.
    - When AUC = 1, the classifier perfectly distinguishs between all Positive and Negative classes.
    - When AUC = 0, the classifier predicts all Negatives as Positives and vice versa.
    - When AUC = 0.5, the classifier can't distinguish between the Positive and Negative classes.
- Log Loss

**Regression Metrics**

We'll discuss three [regression metrics](https://scikit-learn.org/stable/modules/classes.html#regression-metrics): Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

- MSE: Calculated as the mean or average of the squared differences between predicted and expected target values in a dataset.
  - Punishes models with larger errors.
  - A perfect mean squared error value is 0.0, which means that all predictions matched the expected values.
  - First, establish a baseline MSE for your dataset using a simple predictive model (such as predicting the mean target value from the training dataset). A model that achieves a better MSE than this is a better model.
- RMSE: An extension of MSE where the square root of the error is calculated too.
  - Consider using MSE to train a regression predictive model, and to use RMSE to evaluate and report its performance.
  - Similar to MSE, a perfect RMSE value is 0.0.
  - Similar to MSE, it's important to establish a RMSE baseline first.
- MAE: Similar to MSE and RMSE, except it does not give more or less weight to different types of errors. Instead, the scores increase linearly with increases in error.

<br>

### **<a id="hyperparameter-tuning"></a>Hyperparameter Tuning**

Every classification and regression model has many variables/ parameters in its function to improve our results. Hyperparameter tuning helps us automatically test these parameters to find the optimal values that maximize accuracy.

We'll discuss two frequently used hyperparameters: Grid Search and Random Search.

- Grid Search: Examines all combinations of the defined hyperparameters. In other words, we train a model on each possible combination of hyperparameters. The hyperparameters associated with the highest accuracy are chosen.
  - Grid search tends to be very slow since it examines all possible combinations of hyperparameters.

- Random Search: Set a lower and upper bound on the values (if continuous variable) or the possible values the hyperparameter can take on (if categorical variable) for each hyperparameter. A random search is then applied a total of N times, training a model on each set of hyperparameters.
  - Random search is much faster and typically obtains just as high accuracy as a grid search.
