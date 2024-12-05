import os
import pandas as pd
def simplify_text(input_text):
    file_path = '../../CWID_NonNative_Regressor.joblib'

    if os.path.exists(file_path):
        # dev = pd.read_csv(file_path)

        return("File exists")
    else:
        return("File or directory does not exist.")
