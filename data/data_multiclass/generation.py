import numpy as np
import pandas as pd

def generate_multiclass_data(n_samples):
    """
    Generate synthetic data with a multiclass sensitive attribute (Sensitive) with 4 classes (0, 1, 2, 3).

    Parameters:
    - n_samples (int): Number of samples to generate.

    Returns:
    - df (pd.DataFrame): DataFrame containing the generated data.
    """
    # Generate Age, Workclass, Education-Num, Marital Status, Occupation, Relationship, Race,
    # Capital Gain, Capital Loss, Hours per week, and Country
    Age = np.random.randint(18, 80, size=n_samples)
    Workclass = np.random.randint(0, 8, size=n_samples)
    Education_Num = np.random.uniform(0, 20, size=n_samples)
    Marital_Status = np.random.randint(0, 7, size=n_samples)
    Occupation = np.random.randint(0, 14, size=n_samples)
    Relationship = np.random.randint(0, 6, size=n_samples)
    Race = np.random.randint(0, 5, size=n_samples)
    Capital_Gain = np.random.uniform(0, 100000, size=n_samples)
    Capital_Loss = np.random.uniform(0, 5000, size=n_samples)
    Hours_per_week = np.random.uniform(0, 100, size=n_samples)
    Country = np.random.randint(0, 42, size=n_samples)

    # Generate Label
    Label = np.random.randint(0, 2, size=n_samples)

    # Generate Sensitive attribute with 4 classes (0, 1, 2, 3)
    Sensitive = np.random.randint(0, 4, size=n_samples)

    # Generate weight and neighbour (as lists)
    weight = [[i] for i in range(n_samples)]
    neighbour = [[i, i+1, i+2] for i in range(n_samples)]

    # Create DataFrame
    df = pd.DataFrame({
        'Age': Age,
        'Workclass': Workclass,
        'Education-Num': Education_Num,
        'Marital Status': Marital_Status,
        'Occupation': Occupation,
        'Relationship': Relationship,
        'Race': Race,
        'Capital Gain': Capital_Gain,
        'Capital Loss': Capital_Loss,
        'Hours per week': Hours_per_week,
        'Country': Country,
        'Label': Label,
        'Sensitive': Sensitive,
        'weight': weight,
        'neighbour': neighbour
    })

    return df

# Example usage:
n_samples = 32560  # Number of samples to generate
data_df = generate_multiclass_data(n_samples)

# Save to CSV
data_df.to_csv('multiclass_sensitive_data.csv', index=False)
