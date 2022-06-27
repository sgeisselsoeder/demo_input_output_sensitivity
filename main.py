import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy


def occlusion_sensitivity(data: np.array, model, class_index: int):
    number_features = data.shape[0]

    data = np.reshape(data, (1, -1))
    default_result = model.predict_proba(data)
    default_result_target = default_result[0][class_index]

    modified_data = copy.deepcopy(data)
    occlusion_results = np.zeros(data.shape)

    for channel in range(number_features):
        modified_data = copy.deepcopy(data)
        modified_data[0][channel] = 0.0
        result = model.predict_proba(modified_data)
        class_result = result[0][class_index]
        difference = class_result - default_result_target
        occlusion_results[0][channel] = difference
    return occlusion_results


def get_features_and_labels():
    number_examples = 1000
    # 100 examples, 30 features 30, 1 label
    features = np.random.randint(0, 100, size=(number_examples, 30))
    number_classes = 10
    labels = np.arange(0.0, number_classes, number_classes/number_examples)
    features[:, 28] = labels * 10.0 + 12.5
    return features, np.reshape(labels, (-1, 1)).astype(int)


X, y = get_features_and_labels()
model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=400)
model.fit(X, y)
# result = model.predict(X)

# sensitivity_example = np.reshape(X[-1], (1, -1))
sensitivity_example = X[-1]
occlusion_result = occlusion_sensitivity(model=model, data=sensitivity_example, class_index=0)
# occlusion_result_image = np.reshape(occlusion_result, (1, occlusion_result.shape[1]))

fig, axs = plt.subplots(2, sharex=True)
axs[0].set_aspect("auto")
axs[0].imshow(occlusion_result)
axs[1].plot(sensitivity_example)

plt.tight_layout()
plt.show()
