# knn_algorithm.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample Data - Replace with your actual dataset
def load_data():
    data = {
        'skills': [
            'tensorflow keras pytorch machine learning deep learning',
            'react django node js react js php',
            'android android development flutter kotlin xml',
            'ios ios development swift cocoa xcode',
            'ux adobe xd figma zeplin wireframes prototyping',
            'Problemsolving DSA SQL Basic Programming'
        ],
        'category': [
            'Data Science', 
            'Web Development', 
            'Android Development', 
            'IOS Development', 
            'UI-UX Development',
            'Junior Software Developer'
        ]
    }
    df = pd.DataFrame(data)
    return df

def train_knn_model():
    df = load_data()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['skills'])
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(f'KNN Accuracy: {accuracy_score(y_test, y_pred)}')

    model = {
        'vectorizer': vectorizer,
        'classifier': knn
    }

    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def predict_category(skills):
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    vectorizer = model['vectorizer']
    classifier = model['classifier']

    skills_vectorized = vectorizer.transform([skills])
    category = classifier.predict(skills_vectorized)[0]
    return category

def kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_

def logistic_regression(train_data, train_labels, test_data):
    lr = LogisticRegression()
    lr.fit(train_data, train_labels)
    return lr.predict(test_data)

def support_vector_machine(train_data, train_labels, test_data):
    svm = SVC()
    svm.fit(train_data, train_labels)
    return svm.predict(test_data)

def decision_tree(train_data, train_labels, test_data):
    dt = DecisionTreeClassifier()
    dt.fit(train_data, train_labels)
    return dt.predict(test_data)

def random_forest(train_data, train_labels, test_data):
    rf = RandomForestClassifier()
    rf.fit(train_data, train_labels)
    return rf.predict(test_data)

if __name__ == '__main__':
    train_knn_model()
