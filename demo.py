from sklearn import tree, neighbors, svm, naive_bayes
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()

## CHALLENGE - create 3 more classifiers...
clf1 = neighbors.KNeighborsClassifier()
clf2 = svm.SVC()
clf3 = naive_bayes.GaussianNB()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

#CHALLENGE compare their results and print the best one!
y_pred1 = clf1.predict(X)
y_pred2 = clf2.predict(X)
y_pred3 = clf3.predict(X)

accuracy1 = {'name' : 'KNeighborsClassifier', 'accuracy' : accuracy_score(Y, y_pred1)}
accuracy2 = {'name' : 'SVC', 'accuracy' : accuracy_score(Y, y_pred2)}
accuracy3 = {'name' : 'GaussianNB', 'accuracy' : accuracy_score(Y, y_pred3)}

if (accuracy1['accuracy'] > accuracy2['accuracy']) and (accuracy1['accuracy'] > accuracy3['accuracy']):
   best = accuracy1
elif (accuracy2['accuracy'] > accuracy1['accuracy']) and (accuracy2['accuracy'] > accuracy3['accuracy']):
   best = accuracy2
else:
   best = accuracy3

print(best['name'], "is the best one with an accuracy of", best['accuracy'])
