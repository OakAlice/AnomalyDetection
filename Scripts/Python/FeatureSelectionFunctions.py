import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Optional, Union


def feature_selection(training_data: pd.DataFrame,
                     number_features: Optional[int] = None,
                     corr_threshold: float = 0.9,
                     forest: bool = True) -> List[str]:
    """
    Perform feature selection on training data for activity classification.
    
    Args:
        training_data: DataFrame containing features and activity labels
        number_features: Number of features to select
        corr_threshold: Threshold for removing highly correlated features
        forest: Whether to use Random Forest for feature selection
        
    Returns:
        List of selected feature names including 'Activity'
    """
    try:
        # Round number_features to integer
        if number_features is not None:
            number_features = round(number_features)
            
        # Clean and preprocess data
        clean_columns = clean_training_data(training_data, corr_threshold)
        training_data_clean = training_data[clean_columns].dropna()
        
        if forest:
            # Check for multiple classes
            if len(training_data_clean['Activity'].unique()) <= 1:
                print("Only one class detected; implementing PCA feature selection.")
                top_features = pca_feature_selection(
                    data=training_data_clean,
                    number_features=number_features,
                    model="OCC"
                )
            else:
                # Random Forest feature selection
                if len(clean_columns) > number_features:
                    print("Starting Random Forest feature selection.")
                    
                    # Sample 75% of data stratified by Activity
                    sampled_data = training_data_clean.groupby('Activity').sample(
                        frac=0.75,
                        random_state=42
                    )
                    
                    top_features = feature_selection_rf(
                        data=sampled_data,
                        n_trees=500,
                        number_features=number_features
                    )
                    
                    if top_features is None:
                        print("Random Forest feature selection failed; returning all features.")
                        top_features = clean_columns[:-1]  # Exclude Activity
                else:
                    top_features = clean_columns[:-1]  # Exclude Activity
        else:
            top_features = clean_columns[:-1]  # Exclude Activity
            
        # Add Activity back to feature list
        top_features = list(top_features) + ['Activity']
        
        return top_features
        
    except Exception as e:
        print(f"Error in feature_selection: {str(e)}")
        return None


def feature_selection_rf(data: pd.DataFrame,
                        n_trees: int,
                        number_features: int) -> List[str]:
    """
    Perform feature selection using Random Forest importance scores.
    
    Args:
        data: DataFrame containing features and Activity column
        n_trees: Number of trees in Random Forest
        number_features: Number of top features to select
        
    Returns:
        List of selected feature names
    """
    try:
        X = data.drop('Activity', axis=1)
        y = data['Activity']
        
        rf_model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42
        )
        rf_model.fit(X, y)
        
        # Get feature importance scores
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Select top features
        top_features = importance_df['Feature'].head(
            min(number_features, len(importance_df))
        ).tolist()
        
        return top_features
        
    except Exception as e:
        print(f"Error in Random Forest feature selection: {str(e)}")
        return None


def pca_feature_selection(data: pd.DataFrame,
                         number_features: int,
                         model: str,
                         variance_explained: float = 0.95) -> List[str]:
    """
    Perform feature selection using PCA.
    
    Args:
        data: DataFrame containing features
        number_features: Number of features to select
        model: Model type ('OCC', 'Binary', or 'Multi')
        variance_explained: Minimum cumulative variance to explain
        
    Returns:
        List of selected feature names
    """
    # Remove Activity column but keep for later
    numeric_data = data.drop('Activity', axis=1)
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Perform PCA
    pca = PCA()
    pca.fit(scaled_data)
    
    # Calculate variance explained
    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)
    
    # Get number of components needed
    n_components = np.argmax(cum_var >= variance_explained) + 1
    
    # Get feature loadings
    loadings = np.abs(pca.components_[:n_components].T)
    
    # Calculate feature importance
    if model == "OCC":
        # For OCC: Focus on magnitude of contribution
        feature_importance = loadings.sum(axis=1)
    else:
        # For Binary/Multi: Weight by explained variance
        feature_importance = (loadings * var_explained[:n_components]).sum(axis=1)
    
    # Get top features
    importance_df = pd.DataFrame({
        'Feature': numeric_data.columns,
        'Importance': feature_importance
    })
    top_features = importance_df.nlargest(number_features, 'Importance')['Feature'].tolist()
    
    return top_features


def clean_training_data(training_data: pd.DataFrame,
                       corr_threshold: float) -> List[str]:
    """
    Clean and preprocess training data.
    
    Args:
        training_data: Input DataFrame
        corr_threshold: Correlation threshold for feature removal
        
    Returns:
        List of column names to keep
    """
    # Get numeric columns
    numeric_columns = training_data.select_dtypes(include=[np.number]).columns
    numeric_data = training_data[numeric_columns].copy()
    
    # Remove columns with >50% NA
    valid_cols = numeric_data.columns[numeric_data.isna().mean() <= 0.5]
    numeric_data = numeric_data[valid_cols]
    
    # Remove zero-variance columns
    valid_cols = numeric_data.columns[numeric_data.std() > 0]
    numeric_data = numeric_data[valid_cols]
    
    # Remove highly correlated features if more than 1 feature remains
    if len(numeric_data.columns) > 1:
        corr_matrix = numeric_data.corr()
        
        # Replace NaN with 0 in correlation matrix
        corr_matrix = corr_matrix.fillna(0)
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > corr_threshold)]
        
        numeric_data = numeric_data.drop(columns=to_drop)
    
    if len(numeric_data.columns) == 0:
        raise ValueError("No valid features remaining after preprocessing.")
        
    clean_columns = list(numeric_data.columns) + ['Activity']
    
    return clean_columns
