import matplotlib.pyplot as plt

# Define model names and their respective accuracies
model_names = ["DeepFace", "Custom Model"]
accuracies = [97.5, 72.8]  # Replace with the actual accuracy values of your models

# Create a bar chart to compare accuracies
plt.bar(model_names, accuracies, color=['blue', 'green'])

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

# Show the graph
plt.show()
