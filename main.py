import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import column_or_1d

df = pd.read_csv("./pima-data.csv")

def plot_corr(df, size=10):
	"""
	Function plots a graphical correlation matrix for each pair of columns in the dataframe.

	Input:
			df: pandas DataFrame
			size: vertical and horizontal size of the plot

	Displays:
			matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
																							3 ------------------>  1
																							Expect a darkred line running from top left to bottom right
	"""

	corr = df.corr()  # data frame correlation function
	fig, ax = plt.subplots(figsize=(size, size))
	ax.matshow(corr)  # color code the rectangles by correlation value
	plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
	plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
	plt.show()


plot_corr(df)

print(df.shape)
del df['skin']
print(df.shape)

plot_corr(df)

print(df.head(5))

diabetes_map = {
	True: 1,
	False: 0
}

df['diabetes'] = df['diabetes'].map(diabetes_map)

print(df.head(5))

num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
num_all = num_true + num_false

print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, 1.0 * num_true / num_all * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, 1.0 * num_false / num_all * 100))


features_col_names = ["num_preg", "glucose_conc", "diastolic_bp", "thickness", "insulin", "bmi", "diab_pred", "age"]
predicted_class_names = ['diabetes']

X = df[features_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

print("{0:0.2f} in training set".format(len(X_train) * 1.0 / len(df.index)))
print("{0:0.2f} in test set".format(len(X_test) * 1.0 / len(df.index)))

num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
num_all = num_true + num_false

print("Original True: {0} ({1:0.2f}%)".format(num_true, 1.0 * num_true / num_all * 100))
print("Original False: {0} ({1:0.2f}%)".format(num_false, 1.0 * num_false / num_all * 100))
print()

num_train_true = len(y_train[y_train[:] == 1])
num_train_false = len(y_train[y_train[:] == 0])
num_train_all = num_train_true + num_train_false

print("Train True: {0} ({1:0.2f}%)".format(num_train_true, 1.0 * num_train_true / num_train_all * 100))
print("Train False: {0} ({1:0.2f}%)".format(num_train_false, 1.0 * num_train_false / num_train_all * 100))
print()

num_test_true = len(y_test[y_test[:] == 1])
num_test_false = len(y_test[y_test[:] == 0])
num_test_all = num_test_true + num_test_false

print("Test True: {0} ({1:0.2f}%)".format(num_test_true, 1.0 * num_test_true / num_test_all * 100))
print("Test False: {0} ({1:0.2f}%)".format(num_test_false, 1.0 * num_test_false / num_test_all * 100))
print()

print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("# rows missing diastolic_bp {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
print("# rows missing thickness {0}".format(len(df.loc[df['thickness'] == 0])))
print("# rows missing bmi {0}".format(len(df.loc[df['bmi'] == 0])))
print("# rows missing diab_pred {0}".format(len(df.loc[df['diab_pred'] == 0])))
print("# rows missing age {0}".format(len(df.loc[df['age'] == 0])))
print()


fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


nb_model = GaussianNB()
nb_model.fit(X_train, column_or_1d(y_train))
nm_predict_train = nb_model.predict(X_train)


accuracy = metrics.accuracy_score(y_train, nm_predict_train)
print("Accuracy {0:2.2f}%".format(100 * accuracy))

nm_predict_test = nb_model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, nm_predict_test)
print("Accuracy {0:2.2f}%".format(100 * accuracy))

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, nm_predict_test, labels=[1, 0]))
print()

print("Classification Report")
print(metrics.classification_report(y_test, nm_predict_test, labels=[1, 0]))
print()


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, column_or_1d(y_train))

nm_predict_train = rf_model.predict(X_train)
accuracy = metrics.accuracy_score(y_train, nm_predict_train)
print("Accuracy {0:2.2f}%".format(100 * accuracy))

nm_predict_test = rf_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, nm_predict_test)
print("Accuracy {0:2.2f}%".format(100 * accuracy))

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, nm_predict_test, labels=[1, 0]))
print()

print("Classification Report")
print(metrics.classification_report(y_test, nm_predict_test, labels=[1, 0]))
print()


lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, column_or_1d(y_train))

nm_predict_train = lr_model.predict(X_train)
accuracy = metrics.accuracy_score(y_train, nm_predict_train)
print("Accuracy {0:2.2f}%".format(100 * accuracy))

nm_predict_test = lr_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, nm_predict_test)
print("Accuracy {0:2.2f}%".format(100 * accuracy))

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, nm_predict_test, labels=[1, 0]))
print()

print("Classification Report")
print(metrics.classification_report(y_test, nm_predict_test, labels=[1, 0]))
print()

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []
C_val = C_start
best_recall_score = 0
while C_val < C_end:
	C_values.append(C_val)
	lr_model_loop = LogisticRegression(C=C_val, random_state=42, class_weight="balanced")
	lr_model_loop.fit(X_train, column_or_1d(y_train))
	lr_predict_loop_test = lr_model_loop.predict(X_test)
	recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
	recall_scores.append(recall_score)
	if recall_score > best_recall_score:
		best_recall_score = recall_score
		best_lr_predict_test = lr_predict_loop_test
	C_val += C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("First max value of {0:.3f} at C = {1:.3f}".format(best_recall_score, best_score_C_val))

plt.plot(C_values, recall_scores)
plt.xlabel("C Value")
plt.ylabel("Recall Score")
plt.show()


lr_cv_model = LogisticRegressionCV(random_state=42, n_jobs=-1, class_weight="balanced", refit=True, Cs=3, cv=10)
lr_cv_model.fit(X_train, column_or_1d(y_train))

nm_predict_train = lr_cv_model.predict(X_train)
accuracy = metrics.accuracy_score(y_train, nm_predict_train)
print("Accuracy {0:2.2f}%".format(100 * accuracy))

nm_predict_test = lr_cv_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, nm_predict_test)
print("Accuracy {0:2.2f}%".format(100 * accuracy))

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, nm_predict_test, labels=[1, 0]))
print()

print("Classification Report")
print(metrics.classification_report(y_test, nm_predict_test, labels=[1, 0]))
print()
