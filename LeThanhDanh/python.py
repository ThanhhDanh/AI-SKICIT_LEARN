from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import mglearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_digits



# Lê Thanh Danh_Số máy 32


def cau1():
    iris = load_iris()
    print(iris.data)


def cau2():
    print(iris.target)
    print(iris.target_names)


def cau3():
    sepal_length = iris.data[:, 0]
    sepal_width = iris.data[:, 1]

    plt.figure()

    plt.scatter(sepal_length, sepal_width, c=iris.target)

    plt.legend(iris.target_names)

    plt.show()


def cau4():
    pca = PCA(n_components=3)

    pca.fit(iris.data)

    pca_data = pca.transform(iris.data)

    print(pca.explained_variance_ratio_)


def cau5():
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=10, random_state=42)

    print("Training set size:", len(X_train))
    print("Test set size:", len(X_test))


def cau6():
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy:", knn.score(X_test, y_test))


def cau7():
    print(classification_report(y_test, y_pred))


def cau8():
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision boundaries for KNN classifier with K=5')

    plt.show()


def cau916():
    # Load the diabetes dataset
    diabetes = load_diabetes()

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=10, random_state=42)

    # Create a linear regression model
    lr = LinearRegression()

    # Fit the model to the training data
    lr.fit(X_train, y_train)

    # Predict the target values for the test set
    y_pred = lr.predict(X_test)

    # Print the mean squared error of the prediction
    print("Mean squared error:", lr.score(X_test, y_test))

    print("Coefficients:", lr.coef_)

    print("Predicted target values:", y_pred)
    print("Actual target values:", y_test)

    # Calculate the root mean squared error
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root mean squared error:", rmse)

    # Calculate the R-squared score
    r2 = r2_score(y_test, y_pred)
    print("R-squared score:", r2)

    # Select the age feature from the dataset
    X_age = diabetes.data[:, 2]

    # Create a linear regression model
    lr_age = LinearRegression()

    # Fit the model to the age feature
    lr_age.fit(X_age.reshape(-1, 1), diabetes.target)

    # Print the coefficient for the age feature
    print("Coefficient for age:", lr_age.coef_[0])

    # Create a list of feature names
    feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

    # Create a list to store the coefficients for each feature
    coefs = []

    # Loop over each feature and create a linear regression model
    for i in range(10):
        X_feature = diabetes.data[:, i]
        lr_feature = LinearRegression()
        lr_feature.fit(X_feature.reshape(-1, 1), diabetes.target)
        coefs.append(np.array(lr_feature.coef_)[0])

    # Plot the coefficients for each feature
    plt.bar(range(10), coefs)
    plt.xticks(range(10), feature_names)
    plt.xlabel("Physiological features")
    plt.ylabel("Coefficients")
    plt.title("Coefficients for each physiological feature")
    plt.show()


def cau1718():
    cancer = load_breast_cancer()
    print(cancer.keys())

    print("Shape of data:", cancer.data.shape)
    print("Number of benign tumors:", np.sum(cancer.target == 0))
    print("Number of malignant tumors:", np.sum(cancer.target == 1))


def cau19():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.3, random_state=42)

    neighbors = np.arange(1, 11)
    train_scores = []
    test_scores = []

    for k in neighbors:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

    plt.plot(neighbors, train_scores, label="Training score")
    plt.plot(neighbors, test_scores, label="Test score")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def cau20():
    X, y = mglearn.datasets.make_forge()

    lr = LogisticRegression()
    lsvc = LinearSVC()

    lr.fit(X, y)
    lsvc.fit(X, y)
    model = LogisticRegression(max_iter=1000)
    model = LogisticRegression(dual=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for model, ax in zip([lr, lsvc], axes):
        mglearn.plots.plot_2d_separator(model, X, fill=False, eps=0.5, ax=ax, alpha=.7)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{}".format(model.__class__.__name__))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend()

    plt.show()


def cau21():
    faces_dataset = fetch_olivetti_faces()
    print(faces_dataset.DESCR)


def cau2223(images, target, n_images=5):
    faces_dataset = fetch_olivetti_faces()
    fig, axes = plt.subplots(n_images, 1, figsize=(6, 2 * n_images))
    for i in range(n_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: {target[i]}')
        axes[i].axis('off')
    plt.show()

    cau2223(faces_dataset.images, faces_dataset.target)


def cau24():
    faces_dataset = fetch_olivetti_faces()
    svm = SVC(kernel='linear')
    svm.fit(faces_dataset.data, faces_dataset.target)
    predictions = svm.predict(faces_dataset.data)


def cau25():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def cau26(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    return np.mean(scores)


def cau2728(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f'Model score: {score:.2f}')
    print(f'Confusion matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification report:\n{classification_report(y_test, y_pred)}')

    svm = SVC(kernel='linear')
    cau2728(X_train, y_train, X_test, y_test, svm)

def glasses():
    glasses_ranges=[
        (10,19),(30,32),(37,38),(50,59),(63,64),(69,69),(120,121),(124,129),(130,139),(160,161)
        ,(164,169),(180,182),(185,185),(189,189),(190,192),(194,194),(196,199),(260,269),(270,279),(300,309),
        (330,339),(358,359),(360,369)
    ]
    return glasses_ranges

def cau293031(labels,data):
    # Assuming 1 represents faces with glasses and 0 represents faces without glasses
    new_target = [1 if label.startswith('glasses') else 0 for label in labels]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, new_target, test_size=0.2, random_state=42)

    # Define new SVC classifier with linear kernel
    svc_2 = SVC(kernel='linear')

    # Train the classifier with the new target vector
    svc_2.fit(X_train, y_train)

    # Perform cross-validation with 5 folds
    scores = cross_val_score(svc_2, X_train, y_train, cv=5)

    # Report mean accuracy
    mean_accuracy = scores.mean()
    print("Mean Accuracy with Cross-Validation:", mean_accuracy)

def cau3233(labels, data, svc_3):
    person_images = {}

    for i, label in enumerate(labels):
        person_id = label.split('_')[0]  # Extract the person ID from the label
        if person_id not in person_images:
            person_images[person_id] = []
        person_images[person_id].append(i)  # Store the index of the image for this person

    # Separate images of the same person, sometimes wearing glasses and sometimes not
    for person_id, indexes in person_images.items():
        glasses_indexes = [i for i in indexes if 'glasses' in labels[i]]
        noglasses_indexes = [i for i in indexes if 'noglasses' in labels[i]]

    # Assuming you define X_eval as a subset of images from indexes 30 to 39
    X_eval = data[30:40]

    # Get predictions on X_eval
    y_pred = svc_3.predict(X_eval)

    # Reshape the data from arrays to 64 x 64 matrices
    eval_faces = [np.reshape(a, (8, 8)) for a in X_eval]

    # Return the reshaped faces and predictions
    return eval_faces, y_pred

# Step 34: Load the 20 Newsgroups dataset
    news = fetch_20newsgroups(subset='all')

    # Step 35: Check the properties of the dataset
    print("Type of news.data:", type(news.data))
    print("Type of news.target:", type(news.target))
    print("Type of news.target_names:", type(news.target_names))
    print("news.target_names:", news.target_names)

    # Step 6: Preprocessing the data
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def vectorizer(X_train):
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the text data using the vectorizers
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    return X_train_counts, X_train_tfidf

def create_classifier(vectorizer):
    # Create a classifier by combining MultinomialNB with the specified vectorizer
    return MultinomialNB(), vectorizer

def k_fold_cross_validation(classifier, X, y, k=5):
    # Perform K-fold cross-validation for the given classifier
    scores = cross_val_score(classifier, X, y, cv=k)
    return scores.mean()

# Load sample data
digits = load_digits()
data = digits.data
labels = digits.target


X_train, _, y_train, _ = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train_counts, X_train_tfidf = vectorizer(X_train)
classifier_count = create_classifier(X_train_counts)
classifier_tfidf = create_classifier(X_train_tfidf)

score_count = k_fold_cross_validation(classifier_count, X_train_counts, y_train)
score_tfidf = k_fold_cross_validation(classifier_tfidf, X_train_tfidf, y_train)

# Compare performance
print("Mean Accuracy (CountVectorizer):", score_count)
print("Mean Accuracy (TfidfVectorizer):", score_tfidf)


