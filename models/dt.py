# DT
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
def dt_model(feature_vector_x, feature_vector_y):
  tree_param = {'criterion':['gini','entropy'],
                'max_depth':range(1, 50, 5)}
  clf = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=10)
  clf.fit(feature_vector_x, feature_vector_y)
  print(clf.best_params_)
  print('train score:', metrics.accuracy_score(clf.predict(feature_vector_x), feature_vector_y))
  return dtclassifier
