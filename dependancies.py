
def standardizing():
    # Importing the required libraries
    import pandas as pd
    import numpy as np
    data = pd.read_csv("breast-cancer.csv")
    data.drop('id', axis=1, inplace= True)
    data.replace({"M":1, "B":0}, inplace=True)
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    # scaling data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    print("***000000000000000************************************************************")
    return scaler
    
def detect_and_change_separator(input_string):
    # Define a list of potential separators
    potential_separators = [',', '\t', ';']  # Add more if needed

    # Initialize variables to keep track of the best separator and its frequency
    best_separator = None
    max_separator_count = 0

    # Iterate through potential separators
    for separator in potential_separators:
        # Count occurrences of the separator in the input string
        separator_count = input_string.count(separator)
        
        # Update the best separator if the current one has a higher count
        if separator_count > max_separator_count:
            best_separator = separator
            max_separator_count = separator_count

    # If a separator is found, replace it with a comma
    if best_separator is not None:
        output_string = input_string.replace(best_separator, ',')
        return output_string
    else:
        # If no suitable separator is found, return the original string
        return input_string
