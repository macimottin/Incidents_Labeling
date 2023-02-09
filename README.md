# Incidents_Labeling
This repo is about the development of a Machine Learning Model to Label IT Incidents automatically.

The provided code is a basic implementation of a machine learning model for text classification. The main goal is to classify incidents into different categories based on the text description provided. The code consists of the following steps:

**Load the data:** The first step is to load the data into the Python environment. In this case, the data is stored in a CSV file, which is loaded into a Pandas data frame using the pd.read_csv() function.

**Split the data into input and output variables:** Once the data is loaded, it is split into two parts: the input variable (X) and the output variable (y). In this case, the input variable is the description of the incident, and the output variable is the label (category) of the incident.

**Vectorize the input variable:** The input variable, which is in text format, needs to be converted into a numerical representation that the machine learning model can understand. This is done using a technique called TF-IDF (term frequency-inverse document frequency). The TfidfVectorizer function from the sklearn.feature_extraction.text module is used to create a TF-IDF matrix from the input variable.

**Split the data into training and test sets:** The next step is to split the data into training and test sets. This is done using the train_test_split function from the sklearn.model_selection module. The test set is used to evaluate the performance of the model.

**Train the model:** Once the data is split into training and test sets, a logistic regression model is trained using the fit method of the LogisticRegression class. The training set is used to fit the model to the data.

**Predict the labels:** Finally, the model is used to predict the labels for the test set using the predict method. The predicted labels are stored in a variable called y_pred.

**Evaluate the model:** To evaluate the performance of the model, metrics such as accuracy score and confusion matrix can be used. The accuracy_score and confusion_matrix functions from the sklearn.metrics module can be used for this purpose.
