import time  # For measuring execution time
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced data visualization
from sklearn import datasets  # To load standard datasets
from sklearn.feature_selection import SelectKBest, f_classif  # For filter-based feature selection
from sklearn.feature_selection import RFE  # For wrapper-based feature selection
from sklearn.linear_model import LogisticRegression  # Machine learning model for RFE
from sklearn.preprocessing import StandardScaler  # For feature scaling

def load_dataset(choice):
    """
    Load the selected dataset based on user input.

    Parameters:
    - choice (str): User's dataset choice ('1', '2', or '3').

    Returns:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target vector.
    - data (Bunch): Complete dataset object with metadata.
    """
    if choice == '1':
        # Load the Iris dataset
        data = datasets.load_iris()
    elif choice == '2':
        # Load the Wine dataset
        data = datasets.load_wine()
    elif choice == '3':
        # Load the Breast Cancer dataset
        data = datasets.load_breast_cancer()
    else:
        # Raise an error for invalid choices
        raise ValueError("Invalid choice. Please select 1, 2, or 3.")
    
    # Convert the dataset to a pandas DataFrame for easier handling
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data

def filter_method(X, y, k=5):
    """
    Apply the Filter method using SelectKBest with ANOVA F-test.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target vector.
    - k (int): Number of top features to select.

    Returns:
    - scores (np.ndarray): ANOVA F-scores for each feature.
    - selected_features (pd.Index): Names of the selected features.
    - elapsed_time (float): Time taken to perform feature selection.
    """
    start_time = time.time()  # Start the timer

    # Initialize SelectKBest with ANOVA F-test as the scoring function
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit the selector to the data
    selector.fit(X, y)
    
    # Retrieve the scores for each feature
    scores = selector.scores_
    
    # Get the boolean mask of selected features and extract their names
    selected_features = X.columns[selector.get_support()]
    
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    return scores, selected_features, elapsed_time

def wrapper_method(X, y, k=5):
    """
    Apply the Wrapper method using Recursive Feature Elimination (RFE).

    Parameters:
    - X (pd.DataFrame): Scaled feature matrix.
    - y (pd.Series): Target vector.
    - k (int): Number of top features to select.

    Returns:
    - adjusted_scores (np.ndarray): Adjusted importance scores (higher means more important).
    - selected_features (pd.Index): Names of the selected features.
    - elapsed_time (float): Time taken to perform feature selection.
    """
    start_time = time.time()  # Start the timer

    # Initialize Logistic Regression model
    # 'liblinear' solver is suitable for smaller datasets
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    
    # Initialize RFE with the Logistic Regression model and desired number of features
    rfe = RFE(model, n_features_to_select=k)
    
    # Fit RFE to the data
    rfe.fit(X, y)
    
    # Original ranking: 1 (most important) to n_features (least important)
    original_ranking = rfe.ranking_
    
    # Adjust the ranking so that higher scores indicate more important features
    # Formula: adjusted_score = max_rank - original_rank + 1
    max_rank = np.max(original_ranking)  # Find the maximum rank value
    adjusted_scores = max_rank - original_ranking + 1  # Invert the rankings
    
    # Get the boolean mask of selected features and extract their names
    selected_features = X.columns[rfe.support_]
    
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    return adjusted_scores, selected_features, elapsed_time

def plot_filter_scores(feature_names, scores, dataset_name):
    """
    Plot the ANOVA F-scores for each feature obtained from the Filter method.

    Parameters:
    - feature_names (pd.Index): Names of the features.
    - scores (np.ndarray): ANOVA F-scores for each feature.
    - dataset_name (str): Name of the dataset (for plot title).
    """
    # Create a DataFrame for easy plotting
    df = pd.DataFrame({'Feature': feature_names, 'Score': scores})
    
    # Sort the DataFrame by scores in descending order
    df = df.sort_values(by='Score', ascending=False)
    
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot using seaborn
    sns.barplot(x='Score', y='Feature', data=df, palette='viridis')
    
    # Set plot title and labels
    plt.title(f'Filter Method: ANOVA F-scores for {dataset_name} Dataset')
    plt.xlabel('ANOVA F-score')
    plt.ylabel('Features')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.show()

def plot_wrapper_scores(feature_names, scores, dataset_name):
    """
    Plot the adjusted importance scores from the Wrapper method (RFE).

    Parameters:
    - feature_names (pd.Index): Names of the features.
    - scores (np.ndarray): Adjusted importance scores for each feature.
    - dataset_name (str): Name of the dataset (for plot title).
    """
    # Create a DataFrame for easy plotting
    df = pd.DataFrame({'Feature': feature_names, 'Adjusted Score': scores})
    
    # Sort the DataFrame by adjusted scores in descending order
    df = df.sort_values(by='Adjusted Score', ascending=False)
    
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot using seaborn
    sns.barplot(x='Adjusted Score', y='Feature', data=df, palette='magma')
    
    # Set plot title and labels
    plt.title(f'Wrapper Method: Feature Importance Scores for {dataset_name} Dataset')
    plt.xlabel('Importance Score (Higher = More Important)')
    plt.ylabel('Features')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.show()

def main():
    """
    Main function to execute the feature selection process.
    Handles user interaction, performs feature selection, and visualizes results.
    """
    # Display dataset options to the user
    print("Select a dataset for feature selection:")
    print("1. Iris")
    print("2. Wine")
    print("3. Breast Cancer")
    
    # Prompt the user to enter their choice
    choice = input("Enter the number corresponding to your choice (1/2/3): ").strip()
    
    try:
        # Attempt to load the selected dataset
        X, y, data = load_dataset(choice)
    except ValueError as e:
        # Handle invalid input gracefully
        print(e)
        return  # Exit the program
    
    # Extract the dataset name from the dataset description
    dataset_name = data.DESCR.split('\n')[0]
    
    # Display basic information about the selected dataset
    print(f"\nYou selected the {dataset_name} dataset.")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}\n")
    
    # **Preprocessing Step: Feature Scaling**
    # Feature scaling is important for methods like RFE to ensure all features contribute equally
    scaler = StandardScaler()  # Initialize the scaler
    
    # Fit the scaler to the data and transform it
    X_scaled = scaler.fit_transform(X)
    
    # Convert the scaled data back to a pandas DataFrame for consistency
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # **Feature Selection using Filter Method (SelectKBest)**
    print("Performing Feature Selection using Filter Method (SelectKBest)...")
    
    # Apply the filter method to obtain scores and selected features
    filter_scores, filter_selected, filter_time = filter_method(X, y, k=5)
    
    # Display the selected features from the filter method
    print(f"Selected Features (Filter): {list(filter_selected)}")
    
    # Display the time taken to perform the filter method
    print(f"Time taken for Filter Method: {filter_time:.4f} seconds\n")
    
    # **Feature Selection using Wrapper Method (RFE)**
    print("Performing Feature Selection using Wrapper Method (RFE)...")
    
    # Apply the wrapper method to obtain adjusted scores and selected features
    wrapper_scores, wrapper_selected, wrapper_time = wrapper_method(X_scaled, y, k=5)
    
    # Display the selected features from the wrapper method
    print(f"Selected Features (Wrapper): {list(wrapper_selected)}")
    
    # Display the time taken to perform the wrapper method
    print(f"Time taken for Wrapper Method: {wrapper_time:.4f} seconds\n")
    
    # **Visualization: Plotting the Results**
    
    # Plot ANOVA F-scores from the filter method
    plot_filter_scores(X.columns, filter_scores, dataset_name)
    
    # Plot adjusted importance scores from the wrapper method
    plot_wrapper_scores(X.columns, wrapper_scores, dataset_name)

# Ensure that the main function runs only when the script is executed directly
if __name__ == "__main__":
    main()
