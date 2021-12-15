from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def svm_model(feature_vector_x, feature_vector_y):
  params = {'C':[0.01, 0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
       'kernel':['rbf', 'poly', 'linear', 'sigmoid']}
  classifier_linear = GridSearchCV(SVC(decision_function_shape='ovr'), params, cv=10)
  classifier_linear.fit(feature_vector_x, feature_vector_y)
  print('train score:', metrics.accuracy_score(classifier_linear.predict(feature_vector_x), feature_vector_y))
  print(classifier_linear.best_params_)
  return classifier_linear

  