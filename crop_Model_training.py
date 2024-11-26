import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Sample data to simulate the given dataset
data = pd.read_csv('crop_data.csv')

# Load the dataset into a DataFrame
df = pd.DataFrame(data)

# Preprocess dataset
# Encoding the 'label' column to numeric values
df['label_encoded'] = df['label'].factorize()[0]

# Split features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Save the trained model to a pickle file
model_filename = 'decision_tree_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(dt_model, file)

# Save label encoding for reverse mapping
label_mapping = dict(enumerate(df['label'].factorize()[1]))
with open('label_mapping.pkl', 'wb') as file:
    pickle.dump(label_mapping, file)

print(f"Model saved to {model_filename}")
print("Label mapping saved to label_mapping.pkl")
