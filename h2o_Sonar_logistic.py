import h2o
h2o.init()

df = h2o.import_file("G:/Statistics (Python)/Cases/Sonar/Sonar.csv")
#df.summary()
df.col_names

y_col = 'Class'
x_cols = df.col_names[:60]
#x.remove(y)
#x.remove('ID')
print("Response = " + y_col)
print("Pridictors = " + str(x_cols))

df['Class'] = df['Class'].asfactor()
df['Class'].levels()

train,  test = df.split_frame(ratios=[.8])
print(df.shape)
print(train.shape)
#print(valid.shape)
print(test.shape)

from h2o.estimators.glm import H2OGeneralizedLinearEstimator
glm_logistic = H2OGeneralizedLinearEstimator(family = "binomial")
glm_logistic.train(x=x_cols, y=y_col, training_frame=train, 
                   validation_frame=test, model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.auc() )
print(glm_logistic.confusion_matrix() )
h2o.cluster().shutdown()
