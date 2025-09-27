from sklearn.svm import SVC
from sklearn import datasets
import joblib
import os

#Load the Iris dataset
iris = datasets.load_iris()

#Create an SVM classifier
clf = SVC()

#Train the model using the iris dataset
model = clf.fit(iris.data, iris.target_names[iris.target])

# Save the trained model to the shared volume
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "iris_model.pkl")
joblib.dump(model, model_path)

print(f"Model training complete and saved to {model_path}")

