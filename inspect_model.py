"""
Script to inspect the user's model and determine its properties.
"""
import joblib
import sys

try:
    # Load the model
    model = joblib.load('models/model.joblib')
    
    print("=" * 60)
    print("MODEL INSPECTION")
    print("=" * 60)
    print(f"\nModel Type: {type(model).__name__}")
    print(f"Model Class: {type(model)}")
    
    # Check if it has common attributes
    if hasattr(model, 'n_features_in_'):
        print(f"Number of features: {model.n_features_in_}")
    
    if hasattr(model, 'classes_'):
        print(f"Classes: {model.classes_}")
        print(f"Number of classes: {len(model.classes_)}")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature names: {model.feature_names_in_}")
    
    # Try to get model parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"\nModel Parameters:")
        for key, value in list(params.items())[:10]:  # Show first 10
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Model loaded successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
