import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Fixed sample data with equal lengths (130 entries each)
data = {
    'color': (
        ['red']*25 + ['yellow']*25 + ['orange']*20 + 
        ['green']*30 + ['purple']*20 + ['red']*10
    ),
    'weight': (
        list(np.random.randint(70, 220, 25)) +        # Apple
        list(np.random.randint(90, 160, 25)) +        # Banana
        list(np.random.randint(90, 210, 20)) +        # Orange
        list(np.random.randint(5, 15, 30)) +          # Grape
        list(np.random.randint(8, 35, 20)) +          # Strawberry
        list(np.random.randint(150, 300, 10))         # Pomegranate
     ) ,
    'texture': (
        ['smooth']*25 + ['smooth']*20 + ['bumpy']*5 +  # Apple(25) + Banana(25)
        ['rough']*20 +                                # Orange(20)
        ['smooth']*30 +                               # Grape(30)
        ['rough']*20 +                                # Strawberry(20)
        ['bumpy']*10                                  # Pomegranate(10)
    ),
    'fruit': (
        ['Apple']*25 + ['Banana']*25 + ['Orange']*20 + 
        ['Grape']*30 + ['Strawberry']*20 + ['Pomegranate']*10
    )
}

df = pd.DataFrame(data)

# Create categorical mappings
color_mapping = {'red': 0, 'yellow': 1, 'orange': 2, 'green': 3, 'purple': 4}
texture_mapping = {'smooth': 0, 'rough': 1, 'bumpy': 2}

# Convert categorical features
df['color'] = df['color'].map(color_mapping)
df['texture'] = df['texture'].map(texture_mapping)

X = df[['color', 'weight', 'texture']]
y = df['fruit']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Enhanced parameter grid for tuning
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_

# Evaluate model
y_pred = best_dt.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Best Parameters:", grid_search.best_params_)

# Display decision tree rules
tree_rules = export_text(best_dt, feature_names=['color', 'weight', 'texture'])
print("\nDecision Tree Rules:\n", tree_rules)

# Rule-based validation dictionary
fruit_rules = {
    'Apple': {'color': ['red', 'green'], 'weight': (70, 250), 'texture': ['smooth']},
    'Banana': {'color': ['yellow'], 'weight': (80, 160), 'texture': ['smooth', 'bumpy']},
    'Orange': {'color': ['orange'], 'weight': (90, 210), 'texture': ['rough']},
    'Grape': {'color': ['green', 'purple'], 'weight': (5, 15), 'texture': ['smooth']},
    'Strawberry': {'color': ['red'], 'weight': (8, 35), 'texture': ['rough']},
    'Pomegranate': {'color': ['red'], 'weight': (150, 300), 'texture': ['bumpy']}
}

def validate_with_rules(color, weight, texture, prediction):
    inverse_color = {v: k for k, v in color_mapping.items()}
    inverse_texture = {v: k for k, v in texture_mapping.items()}
    
    rules = fruit_rules.get(prediction, {})
    actual_color = inverse_color[color]
    actual_texture = inverse_texture[texture]
    
    validation = {
        'color_valid': actual_color in rules.get('color', []),
        'weight_valid': rules.get('weight', (0, 0))[0] <= weight <= rules.get('weight', (0, 0))[1],
        'texture_valid': actual_texture in rules.get('texture', [])
    }
    
    return validation

def predict_fruit():
    color_options = "/".join(color_mapping.keys())
    texture_options = "/".join(texture_mapping.keys())
    
    while True:
        color = input(f"\nEnter fruit color ({color_options}): ").lower()
        if color in color_mapping:
            break
        print(f"Invalid color! Please choose from {color_options}")
    
    while True:
        texture = input(f"Enter texture ({texture_options}): ").lower()
        if texture in texture_mapping:
            break
        print(f"Invalid texture! Please choose from {texture_options}")
    
    while True:
        try:
            weight = float(input("Enter weight in grams: "))
            if weight > 0:
                break
            print("Weight must be positive!")
        except ValueError:
            print("Please enter a valid number!")
    
    color_enc = color_mapping[color]
    texture_enc = texture_mapping[texture]
    
    features = pd.DataFrame([[color_enc, weight, texture_enc]],
                          columns=['color', 'weight', 'texture'])
    
    prediction = best_dt.predict(features)[0]
    proba = best_dt.predict_proba(features)[0]
    
    # Rule-based validation
    validation = validate_with_rules(color_enc, weight, texture_enc, prediction)
    
    print("\n" + "="*50)
    print("Prediction Process:")
    print(f"- Model Prediction: {prediction} (Confidence: {max(proba)*100:.1f}%)")
    print("- Rule-based Validation:")
    print(f"  Color: {'Valid' if validation['color_valid'] else 'Invalid'}")
    print(f"  Weight: {'Valid' if validation['weight_valid'] else 'Invalid'}")
    print(f"  Texture: {'Valid' if validation['texture_valid'] else 'Invalid'}")
    
    if all(validation.values()):
        print(f"\nFinal Decision: {prediction} (Validated by rules)")
    else:
        alternatives = []
        for fruit, rules in fruit_rules.items():
            if (color in rules['color'] and 
                rules['weight'][0] <= weight <= rules['weight'][1] and 
                texture in rules['texture']):
                alternatives.append(fruit)
        
        if alternatives:
            print(f"\nWarning: Model prediction doesn't match rules!")
            print(f"Possible alternatives: {', '.join(alternatives)}")
        else:
            print("\nUnknown fruit! Doesn't match any known patterns")

# Run the predictor
predict_fruit()