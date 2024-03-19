import pandas as pd
import re

"""
    Uncomment the following lines to display all the columns and rows
"""
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read csv file to pandas dataframe
def Read_Data(file_path):
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


def Process_Data(data):
    print("Processing Data")
    pattern = ["ID", "item", "cash_price", "make", "Nbr_of_prod_purchas, ", "Nb_of_items", "fraud_flag"]
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
#TODO: make all the NaN values -1 as this fucntion does that
def data_categories(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

    return data


if __name__ == '__main__':
    data = Read_Data("Data/train_dataset.csv")
    # Data_Summary(data)
    data = Process_Data(data)
    Data_Summary(data)
    data = data_categories(data)
    print(data.head())
    print(data.shape)
