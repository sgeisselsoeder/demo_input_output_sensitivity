import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import copy


def occlusion_sensitivity(model, data: np.array, class_index: int, change_factor=0.0):
    # Checks the influence of all features on the output
    # Assumes examples of shape (1, number_of_features)
    # Reasonable values for change factor would be e.g. 0.0, 0.8, 1.25

    # Obtain output for actual input
    default_result = model.predict_proba(data)
    default_result_target = default_result[0][class_index]

    occlusion_results = np.zeros(data.shape)
    for feature in range(data.shape[1]):
        # Modify input data
        modified_data = copy.deepcopy(data)
        modified_data[0][feature] *= change_factor

        # Obtain output for modified input
        result = model.predict_proba(modified_data)
        class_result = result[0][class_index]

        # Store the difference
        difference = class_result - default_result_target
        occlusion_results[0][feature] = difference
    return occlusion_results


def get_features_and_labels():
    # Returns dummy features and labels
    number_examples = 1000
    number_classes = 10
    # Create random features and labels
    features = np.random.randint(0, 100, size=(number_examples, 30))
    labels = np.arange(0.0, number_classes, number_classes/number_examples)
    # Make one of the features highly correlated with the labels
    features[:, 28] = labels * 10.0 + 12.5
    return features, labels.astype(int)


# Setup data
X, y = get_features_and_labels()
# Train model
# model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=400)
model = DecisionTreeClassifier()
model.fit(X, y)

# Test multiple examples
for example_test in range(2):
    # Pick a random example
    random_example = int(np.random.random() * y.shape[0])
    print("Evaluating random example at index", random_example)
    sensitivity_example = np.reshape(X[random_example], (1, -1))
    true_class = int(y[random_example])

    # Test sensitivity of result against occlusion
    occlusion_result = occlusion_sensitivity(model=model, data=sensitivity_example, class_index=true_class)

    # plot result
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].set_aspect("auto")
    axs[0].imshow(occlusion_result)
    axs[1].plot(X[-1])
    plt.tight_layout()
    plt.show()
