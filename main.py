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
import tensorflow.keras.backend as K

# TODO: VERY IMPORTANT: THE OUTPUT SHOULD BE THE PROBABILITY OF THE INPUT BEING A FRAUDULENT TRANSACTION

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
            tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu)   # TODO: Might have to change to sigmoid later
        ])

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           optimizer=optimizer,
                           metrics=["mae"])

        # Model checkpoint callback to save the best model based on lowest MAE
        # self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
        #                                                      monitor='mae',
        #                                                      verbose=1,
        #                                                      save_best_only=True,
        #                                                      mode='min')

    def weighted_binary_crossentropy(y_true, y_pred):
        """
        Custom binary cross-entropy loss function with class weights.

        Args:
            y_true: True labels (one-hot encoded or integers).
            y_pred: Predicted probabilities.

        Returns:
            Weighted binary cross-entropy loss.
        """
        class_weights = {0: 0.1, 1: 0.9}
        # Clip predictions to avoid overflow
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Calculate binary cross-entropy loss
        bce = K.binary_crossentropy(y_true, y_pred)
        # Apply class weights based on true labels
        weighted_bce = K.mean(class_weights[K.cast(y_true[:, 0], dtype='int32')] * bce)
        return weighted_bce

    def train(self, X_train, y_train, epochs=1000, batch_size=32):

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
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
    print("Reading Data")
    # noinspection PyShadowingNames
    data = pd.read_csv(file_path, low_memory=False)
    # print(data.head())
    print("Data Shape: ", data.shape)
    print("---------------------------------------------------------------------------------")
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
    numerical_pattern = ["ID", "item", "cash_price", "make", "Nbr_of_prod_purchas", "Nb_of_items", "fraud_flag"]
    text_pattern = ["ID", "item", "make", "model", "goods_code","fraud_flag"]
    # Creating a new dataframe with the selected column names
    numerical_df = pd.DataFrame()
    text_df = pd.DataFrame()

    # Selecting the columns based on the pattern for numerical data
    for pattern in numerical_pattern:
        for column in data.columns:
            if re.match(pattern, column):
                numerical_df[column] = data[column]

    # Selecting the columns based on the pattern for text data
    for pattern in text_pattern:
        for column in data.columns:
            if re.match(pattern, column):
                text_df[column] = data[column]

    # print(numerical_df.head())
    # print("numerical df shape: ", numerical_df.shape)
    # print("=====================================")
    # print(text_df.head())
    # print("text df shape: ", text_df.shape)
    # print("=====================================")
    print("---------------------------------------------------------------------------------")
    return numerical_df, text_df


# Changing the data type of the columns to category and then to numerical values for the model
# TODO: makes all the NaN values -1 have to make sure it doesn't affect the model
# TODO: do one hot encoding with this [     data = pd.get_dummies(data, columns=[column])   ]
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

# TODO: use word2vec or something for the text data and one hot encode/code the categorical data
def Text_Data_Preprocessing(data):
    # Filling the missing values with the mode of the column with -1
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            data[column] = data[column].fillna(-1)

    # Scaling data only on cash_price column
    scaler = StandardScaler()

    # Categorizing goods_code column
    for column in data.columns:
        if re.match("goods_code", column):
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

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


def Machine_Learning(X_train, X_test, y_train, y_test):
    print("Machine Learning")
    print("=====================================")

    class_weights = {0: 0.1, 1: 0.9}

    log_reg = lm.LogisticRegression(class_weight=class_weights, random_state=42, max_iter=1000, n_jobs=-1, verbose=1)
    log_reg.fit(X_train, y_train)
    log_reg_score = log_reg.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: ", log_reg_score)
    print("Logistic Regression Classification Report: ")
    print(classification_report(y_test, log_reg.predict(X_test)))
    print(average_precision_score(y_test, log_reg.predict(X_test)))
    print("=====================================")

    # Decision Tree
    Decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=None, min_samples_leaf=5,
                                           class_weight=class_weights)
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

    models = [log_reg, Decision_tree, knn]
    scores = [log_reg_score,
              Decision_tree_score,
              knn_score]

    print("---------------------------------------------------------------------------------")

    return models, scores


if __name__ == '__main__':
    data = Read_Data("Data/train_dataset.csv")
    # Data_Summary(data)
    numerical_data, text_data = Feature_Selection(data)

    numerical_data = data_categories(numerical_data)

    X_train_text, X_test_text, y_train_text, y_test_text = Text_Data_Preprocessing(text_data)
    # X_train, X_test, y_train, y_test = Data_Preprocessing(data)

    # print(X_train.shape[1])
    # myModel = MyModel(input_shape=(X_train.shape[1],))
    # myModel.train(X_train, y_train, epochs=10, batch_size=32)
    # results = myModel.predict(X_test)
    # print(average_precision_score(y_test, results))
    # print(results)
#
    # models, scores = Machine_Learning(X_train, X_test, y_train, y_test)
