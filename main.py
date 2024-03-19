import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

"""
    Uncomment the following lines to display all the columns and rows
"""
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class MyModel:
    def __init__(self, input_shape):
        super().__init__()
        # Create the model layers
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, input_shape=input_shape),
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-07,
                                             amsgrad=False)

        self.model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

        # Model checkpoint callback to save the best model based on lowest MAE
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                             monitor='mae',
                                                             verbose=1,
                                                             save_best_only=True,
                                                             mode='min')

    def train(self, X_train, y_train, epochs=1000, batch_size=32):

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 callbacks=[self.checkpoint])
        return history

    # Evaluate the model on the test data using average_precision_score
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        score = average_precision_score(y_test, y_pred)
        print("Average Precision Score: ", score)
        return score

    def predict(self, X_new):
        return self.model.predict(X_new)

    def load_best_model(self):
        # Loads the best model saved during training based on lowest validation MAE.
        self.model = tf.keras.models.load_model('best_model.h5')


# Read csv file to pandas dataframe
def Read_Data(file_path):
    # noinspection PyShadowingNames
    data = pd.read_csv(file_path, low_memory=False)
    # print(data.head())
    print(data.shape)
    return data


# Data Summary Function
def Data_Summary(data):
    print("Data Summary")
    print("=====================================")
    print("First 5 rows")
    print(data.describe())
    print("=====================================")
    print("Data types")
    print(data.info())
    print("=====================================")
    print("Data count")
    print(data.count())
    print("=====================================")
    print("Missing values")
    print(data.isnull().sum())
    print("=====================================")
    print("Data shape")
    print(data.shape)
    print("=====================================")
    print("Unique values in each column")
    print(data.nunique())
    print("=====================================")

    # describe the last column
    print("Last column description")
    print(data.iloc[:, -1].describe())
    print("---------------------------------------------------------------------------------")


def Feature_Selection(data):
    print("Processing Data")
    pattern = ["ID", "item", "cash_price", "make", "Nbr_of_prod_purchas", "Nb_of_items", "fraud_flag"]
    # Creating a new dataframe with the selected column names
    selected_df = pd.DataFrame()
    for pattern in pattern:
        for column in data.columns:
            if re.match(pattern, column):
                selected_df[column] = data[column]
    #print(selected_df.head())
    print(selected_df.shape)
    print("=====================================")
    return selected_df


# Changing the data type of the columns to category and then to numerical values for the model
# TODO: make all the NaN values -1 as this fucntion does that
def data_categories(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

    return data


def Data_Preprocessing(data):
    # Filling the missing values with the mode of the column with -1
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if re.match("cash_price", column):
                data[column] = data[column].fillna(0)
            else:
                data[column] = data[column].fillna(-1)

    # Scaling data only on cash_price column
    scaler = StandardScaler()

    for column in data.columns:
        if re.match("cash_price", column):
            data[column] = scaler.fit_transform(data[[column]])

    # Dropping the ID column
    data = data.drop(columns=["ID"])

    # Splitting the data into train and test
    X = data.drop(columns=["fraud_flag"])
    y = data["fraud_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Train data shape: ", X_train.shape)
    print("Test data shape: ", X_test.shape)
    print("Train label shape: ", y_train.shape)
    print("Test label shape: ", y_test.shape)

    print(X_train.head())
    print("=====================================")
    print(y_train.head())
    print("---------------------------------------------------------------------------------")

    return X_train, X_test, y_train, y_test


def Machine_Learning(_train, X_test, y_train, y_test):
    print("Machine Learning")
    print("=====================================")

    log_reg = lm.LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_score = log_reg.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: ", log_reg_score)
    print("Logistic Regression Classification Report: ")
    print(classification_report(y_test, log_reg.predict(X_test)))
    print(average_precision_score(y_test, log_reg.predict(X_test)))
    print("=====================================")

    # Decision Tree
    Decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=None, min_samples_leaf=5)
    Decision_tree.fit(X_train, y_train)
    Decision_tree_score = Decision_tree.score(X_test, y_test)
    print("Decision Tree Testing Accuracy: ", Decision_tree_score)
    print("Decision Tree Classification Report: ")
    print(classification_report(y_test, Decision_tree.predict(X_test)))
    print(average_precision_score(y_test, Decision_tree.predict(X_test)))
    print("=====================================")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print("KNN Testing Accuracy: ", knn_score)
    print("KNN Classification Report: ")
    print(classification_report(y_test, knn.predict(X_test)))
    print(average_precision_score(y_test, knn.predict(X_test)))
    print("=====================================")

    lin_reg = lm.LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)
    print("Linear Regression Testing Accuracy: ", lin_reg_score)
    print("=====================================")

    Sgd_reg = lm.SGDRegressor()
    Sgd_reg.fit(X_train, y_train)
    Sgd_reg_score = Sgd_reg.score(X_test, y_test)
    print("SGD Testing Accuracy: ", Sgd_reg_score)
    print("=====================================")

    Ridge_reg = lm.Ridge()
    Ridge_reg.fit(X_train, y_train)
    Ridge_reg_score = Ridge_reg.score(X_test, y_test)
    print("Ridge Testing Accuracy: ", Ridge_reg_score)
    print("=====================================")

    Lasso_reg = lm.Lasso()
    Lasso_reg.fit(X_train, y_train)
    Lasso_reg_score = Lasso_reg.score(X_test, y_test)
    print("Lasso Testing Accuracy: ", Lasso_reg_score)
    print("=====================================")

    Elastic_reg = lm.ElasticNet()
    Elastic_reg.fit(X_train, y_train)
    Elastic_reg_score = Elastic_reg.score(X_test, y_test)
    print("Elastic Testing Accuracy: ", Elastic_reg_score)
    print("=====================================")

    Huber_reg = lm.HuberRegressor()
    Huber_reg.fit(X_train, y_train)
    Huber_reg_score = Huber_reg.score(X_test, y_test)
    print("Huber Testing Accuracy: ", Huber_reg_score)
    print("=====================================")

    Ransac_reg = lm.RANSACRegressor()
    Ransac_reg.fit(X_train, y_train)
    Ransac_reg_score = Ransac_reg.score(X_test, y_test)
    print("Ransac Testing Accuracy: ", Ransac_reg_score)
    print("=====================================")

    # Theil regression is not working (Low memory)
    # Theil_reg = lm.TheilSenRegressor()
    # Theil_reg.fit(X_train, y_train)
    # Theil_reg_score = Theil_reg.score(X_test, y_test)
    # print("Theil Testing Accuracy: ", Theil_reg_score)

    models = [log_reg, Decision_tree, knn, lin_reg, Sgd_reg, Ridge_reg, Lasso_reg, Elastic_reg, Huber_reg, Ransac_reg]
    scores = [log_reg_score,
              Decision_tree_score,
              knn_score,
              lin_reg_score,
              Sgd_reg_score,
              Ridge_reg_score,
              Lasso_reg_score,
              Elastic_reg_score,
              Huber_reg_score,
              Ransac_reg_score]

    print("---------------------------------------------------------------------------------")

    return models, scores


if __name__ == '__main__':
    data = Read_Data("Data/train_dataset.csv")
    # Data_Summary(data)
    data = Feature_Selection(data)
    data = data_categories(data)
    X_train, X_test, y_train, y_test = Data_Preprocessing(data)

    # myModel = MyModel(input_shape=(X_train.shape[1],))
    # myModel.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    models, scores = Machine_Learning(X_train, X_test, y_train, y_test)
