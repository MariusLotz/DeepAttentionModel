# Import fix for Codespace:
import sys
sys.path.append("./")
from Helper_Functions import data_table_to_tensors
import pandas as pd
pd.options.mode.copy_on_write = True 


def FordA_dataset_trafo(pandas_dataset):
    """
    Modify the FordA dataset by converting -1 to 0 in the class column.

    Parameters:
    - pandas_dataset (pd.DataFrame): Pandas DataFrame representing the dataset.

    Returns:
    - tuple: A tuple containing NumPy arrays for the modified class column and time series columns.
    """
    # Separate the first column (class) and the rest of the columns (time series)
    df = pandas_dataset.copy()  # Make a copy to avoid modifying the original DataFrame
    class_column = df.iloc[:, 0]
    time_series_columns = df.iloc[:, 1:]
    
    # Convert -1 to 0 in the class column using .loc
    class_column.loc[class_column == -1] = 0
    
    return class_column.values, time_series_columns.values

def FordA_preprocessing(testset=False):
    if not testset:
        file_path = "Example_Problems/Raw_Data/FordA/FordA_TRAIN.tsv"

    else:
        file_path = "Example_Problems/Raw_Data/FordA/FordA_TEST.tsv"
    spec_type = "tsv"
    return data_table_to_tensors(file_path, spec_type, FordA_dataset_trafo)

if __name__ == "__main__":
    print(FordA_preprocessing())
