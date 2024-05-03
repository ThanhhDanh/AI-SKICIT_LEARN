from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import python

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score




if __name__ == '__main__':
    # Load sample data (you may replace this with your own data)
    digits = load_digits()
    data = digits.data
    # labels = digits.target
    labels = [str(label) for label in digits.target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train an SVC model
    svc_3 = SVC()
    svc_3.fit(X_train, y_train)

    X_train, _, y_train, _ = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train_counts, X_train_tfidf = python.vectorizer(X_train)
    classifier_count = python.create_classifier(X_train_counts)
    classifier_tfidf = python.create_classifier(X_train_tfidf)

    # python.cau1()
    # python.cau2()
    # python.cau3()
    # python.cau4()
    # python.cau5()
    # python.cau6()
    # python.cau7()
    # python.cau8()
    #python.cau916()
    #python.cau1718()
    #python.cau19()
    #python.cau20()
    #python.cau24()
    #python.cau2728()
    # python.cau293031(labels,data)
    # X_train, X_test, y_train, y_test = python.cau3233(labels,data,svc_3)
    # X_train_counts, X_train_tfidf = python.vectorizer(X_train)
    score_count = python.k_fold_cross_validation(classifier_count, X_train_counts, y_train)
    score_tfidf = python.k_fold_cross_validation(classifier_tfidf, X_train_tfidf, y_train)
