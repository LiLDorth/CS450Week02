import numpy
import scipy
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.preprocessing import StandardScaler

class Model:


    def trained(self):
        #print(*self.Train_Data, sep="\n")
        for item in self.Train_Data:
            print(item[0][0])

    def predict(self, data):
        prediction = []
        distances = []
        for item in data:
            distances.clear()
            for trainedItem in self.Train_Data:
                distances.append([(abs((item[0] - trainedItem[0][3])) + abs((item[1] - trainedItem[0][1])) + abs((item[2] - trainedItem[0][2])) + abs((item[3] - trainedItem[0][3]))), trainedItem[1]])
            distances.sort()
            targetNeighbors = []
            for closest in distances[:self.K]:
                targetNeighbors.append(closest[1])
            prediction.append(Counter(targetNeighbors).most_common()[0][0])
        return prediction


class HardcodedClassifier:
    def fit(X_Train, Y_Train, k):
        #print(X_Train, Y_Train)
        Model.Train_Data = list(zip(X_Train, Y_Train))
        Model.K = k
        return Model

def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    iris = datasets.load_iris()
    scaler = StandardScaler()
    iris.data = scaler.fit_transform(iris.data)
    #print(iris.data)
    data_train, data_test, target_train, target_test  = train_test_split(iris.data, iris.target, test_size=45)
    # Show the data (the attributes of each instance)
    #print(iris.data)
    # Show the target values (in numeric format) of each instance
    #print(iris.target)
    # Show the actual target names that correspond to each number
    #print(iris.target_names)

    classifier = HardcodedClassifier
    #classifier = GaussianNB()
    #classifier = KNeighborsClassifier(n_neighbors=3)


    model = classifier.fit(data_train, target_train, k = 3)
    #model.trained(model)

    #targets_predicted = model.predict(data_test)
    targets_predicted = model.predict(model, data_test)
    #print (targets_predicted)

    print(str(accuracy_score(target_test, targets_predicted)*100)+"%")

    #classes = datasets.load_iris().target_names
    #visualizer = ClassificationReport(classifier, classes=classes)
    #visualizer.fit(data_train, target_train)
    #visualizer.score(data_test, target_test)
    #visualizer.poof()

    #cm = ConfusionMatrix(classifier)
    #cm.fit(data_train, target_train)
    #cm.score(data_test, target_test)
    #cm.poof()
if __name__ == '__main__':
    main()

