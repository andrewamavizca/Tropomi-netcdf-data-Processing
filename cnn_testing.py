# %% [markdown]
# # CNN Automated Detection of Super Emitters Using Sattelite Data

# %% [markdown]
# ## pipeline: 1

# %% [markdown]
# ### setup

# %% [markdown]
# #### Imports

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import h5py
import cv2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import register_keras_serializable
import os
import warnings
import absl.logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Abseil logs
absl.logging.set_verbosity('error')

# Ignore all warnings
warnings.filterwarnings("ignore")

# Optionally suppress specific warnings, like the lambda serialization warning
warnings.filterwarnings("ignore", category=UserWarning, message="The object being serialized includes a `lambda`.")

# %% [markdown]
# #### custom binary functions

# %%

# @register_keras_serializable(name="weighted_binary_crossentropy")
# def weighted_binary_crossentropy(target, output, weights):
#     target = tf.convert_to_tensor(target)
#     output = tf.convert_to_tensor(output)
#     weights = tf.convert_to_tensor(weights, dtype=target.dtype)

#     epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
#     output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

#     # Compute cross entropy from probabilities.
#     bce = weights[1] * target * tf.math.log(output + epsilon_)
#     bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
#     return -bce

# %%
# @register_keras_serializable(name="WeightedBinaryCrossentropy")
# class WeightedBinaryCrossentropy:
#     def __init__(
#         self,
#         label_smoothing=0.0,
#         weights=[1.0, 1.0],
#         axis=-1,
#         name="weighted_binary_crossentropy",
#         fn=None,
#     ):
#         """Initializes `WeightedBinaryCrossentropy` instance."""
#         super().__init__()
#         self.weights = weights
#         self.label_smoothing = label_smoothing
#         self.name = name
#         self.fn = weighted_binary_crossentropy if fn is None else fn

#     def __call__(self, y_true, y_pred):
#         y_pred = tf.convert_to_tensor(y_pred)
#         y_true = tf.cast(y_true, y_pred.dtype)
#         self.label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

#         def _smooth_labels():
#             return y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

#         y_true = tf.__internal__.smart_cond.smart_cond(self.label_smoothing, _smooth_labels, lambda: y_true)

#         return tf.reduce_mean(self.fn(y_true, y_pred, self.weights), axis=-1)

#     def get_config(self):
#         config = {"name": self.name, "weights": self.weights, "fn": None}
#         return dict(list(config.items()))

#     @classmethod
#     def from_config(cls, config):
#         """Instantiates a `Loss` from its config (output of `get_config()`)."""
#         config["fn"] = weighted_binary_crossentropy if config.get("fn") is None else config["fn"]
#         return cls(**config)

# %%
@register_keras_serializable(name="WeightedBinaryCrossentropy")
def weighted_binary_crossentropy(y_true, y_pred, weight_0=1.0, weight_1=2.0):
    # Calculate the standard binary cross-entropy loss
    bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = bce(y_true, y_pred)

    # Apply weights: weight_1 for positive class (y_true=1) and weight_0 for negative class (y_true=0)
    weight_vector = y_true * weight_1 + (1.0 - y_true) * weight_0
    weighted_loss = loss * weight_vector

    # Return the mean loss
    return tf.reduce_mean(weighted_loss)

# %%
# Define a named loss function
@register_keras_serializable()
def custom_weighted_binary_crossentropy(y_true, y_pred):
    return weighted_binary_crossentropy(y_true, y_pred, weight_0=1.0, weight_1=2.0)


# %% [markdown]
# #### Data Preperation

# %%
data_type = [
            ('xch4_corrected', 'f4', ()),
            ('latitude_corners', 'f4', (4,)),
            ('longitude_corners', 'f4', (4,)),
            ('u10', 'f4', ()),
            ('v10', 'f4', ()),
            ('latitude_center', 'f4', ()),
            ('longitude_center', 'f4', ()),
            ('scanline', 'i4', ()),
            ('ground_pixel', 'i4', ()),
            ('time', 'i4', (7,)),
            ('solar_zenith_angle', 'f4', ()),
            ('viewing_zenith_angle', 'f4', ()),
            ('relative_azimuth_angle', 'f4', ()),
            ('altitude_levels', 'f4', (13,)),
            ('surface_altitude', 'f4', ()),
            ('surface_altitude_stdv', 'f4', ()),
            ('dp', 'f4', ()),
            ('surface_pressure', 'f4', ()),
            ('dry_air_subcolumns', 'f4', (12,)),
            ('fluorescence_apriori', 'f4', ()),
            ('cloud_fraction', 'f4', (4,)),
            ('weak_h2o_column', 'f4', ()),
            ('strong_h2o_column', 'f4', ()),
            ('weak_ch4_column', 'f4', ()),
            ('strong_ch4_column', 'f4', ()),
            ('cirrus_reflectance', 'f4', ()),
            ('stdv_h2o_ratio', 'f4', ()),
            ('stdv_ch4_ratio', 'f4', ()),
            ('xch4', 'f4', ()),
            ('xch4_precision', 'f4', ()),
            ('xch4_column_averaging_kernel', 'f4', (12,)),
            ('ch4_profile_apriori', 'f4', (12,)),
            ('xch4_apriori', 'f4', ()),
            ('fluorescence', 'f4', ()),
            ('co_column', 'f4', ()),
            ('co_column_precision', 'f4', ()),
            ('h2o_column', 'f4', ()),
            ('h2o_column_precision', 'f4', ()),
            ('spectral_shift', 'f4', (2,)),
            ('aerosol_size', 'f4', ()),
            ('aerosol_size_precision', 'f4', ()),
            ('aerosol_column', 'f4', ()),
            ('aerosol_column_precision', 'f4', ()),
            ('aerosol_altitude', 'f4', ()),
            ('aerosol_altitude_precision', 'f4', ()),
            ('aerosol_optical_thickness', 'f4', (2,)),
            ('surface_albedo', 'f4', (2,)),
            ('surface_albedo_precision', 'f4', (2,)),
            ('reflectance_max', 'f4', (2,)),
            ('convergence', 'i4', ()),
            ('iterations', 'i4', ()),
            ('chi_squared', 'f4', ()),
            ('chi_squared_band', 'f4', (2,)),
            ('number_of_spectral_points_in_retrieval', 'i4', (2,)),
            ('degrees_of_freedom', 'f4', ()),
            ('degrees_of_freedom_ch4', 'f4', ()),
            ('degrees_of_freedom_aerosol', 'f4', ()),
            ('signal_to_noise_ratio', 'f4', (2,)),
            ('rms', 'f4', ())
        ]

channel_map = {}
current_channel = 0

for name, field_type, *field_shape in data_type:
    # Ensure current_channel is an integer
    current_channel = int(current_channel)

    if not field_shape:  # Scalar field
        channel_map[name] = slice(current_channel, current_channel + 1)
        current_channel += 1
    else:  # Multi-dimensional field
        total_channels = int(np.prod(field_shape))
        channel_map[name] = slice(current_channel, current_channel + total_channels)
        current_channel += total_channels

# Add the normalized methane variable as the last channel
channel_map['normalized_ch4'] = slice(current_channel, current_channel + 1)


# %%
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)  # Properly calling the superclass initializer
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x_set))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.x_set) / self.batch_size))

    def __getitem__(self, index):
        if index == self.__len__() - 1:  # Last batch
            batch_indices = self.indices[index * self.batch_size:]  # Get all remaining indices
        else:
            batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = self.x_set[batch_indices]
        y_batch = self.y_set[batch_indices]
        return x_batch, y_batch

    def on_epoch_end(self):
        # Shuffle the indices after each epoch if shuffle is True
        if self.shuffle:
            np.random.shuffle(self.indices)


# %% [markdown]
# #### Data Load

# %%
x1_test = np.load("data/test/xtest.npy", allow_pickle=True)[:, :, :, channel_map['normalized_ch4']]
y_test = np.load("data/test/ytest.npy", allow_pickle=True)

x1_val = np.load("data/validation/xval.npy", allow_pickle=True)[:, :, :, channel_map['normalized_ch4']]
y_val = np.load("data/validation/yval.npy", allow_pickle=True)

x1_train = np.load("data/training/xtrain.npy", allow_pickle=True)[:, :, :, channel_map['normalized_ch4']]
y_train = np.load("data/training/ytrain.npy", allow_pickle=True)

# %% [markdown]
# #### Model Setup

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(shape=(32, 32, 1)),
    tf.keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.4),  # 40% dropout
    tf.keras.layers.Dense(70, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile your model with binary focal loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss=custom_weighted_binary_crossentropy,
#     metrics=['accuracy']
# )
# model.summary()




# %%
def get_class_weights(y):
    num_positives = np.sum(y == 1)
    num_negatives = np.sum(y == 0)
    class_weights = {0: 1.0, 1: 1.0*(num_negatives / num_positives) }
    print(class_weights)
    return class_weights

# Set class weights: weight for positive class is double the inverse ratio
class_weights = get_class_weights(y_train)

# %% [markdown]
# ### Model Training

# %%
from sklearn.metrics import cohen_kappa_score
import tempfile
import tensorflow as tf

batch_size = 128
train_generator = DataGenerator(x1_train, y_train, batch_size=batch_size, shuffle=True)
val_generator = DataGenerator(x1_val, y_val, batch_size=batch_size, shuffle=True)
test_generator = DataGenerator(x1_test, y_test, batch_size=batch_size, shuffle=False)

# Set up tf.callbacks: early stopping and model checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Reduce learning rate with a patience of 5 epochs
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=1e-12, verbose=0)

# Custom Learning Rate Reset at epoch 30
def reset_lr(epoch, lr):
    reset_epoch = 15  # The epoch at which to reset the learning rate
    if epoch == reset_epoch:
        new_lr = 1e-4  # Reset to a higher learning rate
        print(f"Resetting learning rate to {new_lr}")
        return new_lr
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(reset_lr)

# Combine the callbacks
callbacks = [early_stopping, model_checkpoint, reduce_lr]

# Assign the bias of the last layer (if needed)
#model.layers[-1].bias.assign([-0.99619189])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    class_weight=class_weights,
    epochs=100,  # Training for a maximum of 100 epochs
    callbacks=callbacks
)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Abseil logs
absl.logging.set_verbosity('error')

# Ignore all warnings
warnings.filterwarnings("ignore")

# Optionally suppress specific warnings, like the lambda serialization warning
warnings.filterwarnings("ignore", category=UserWarning, message="The object being serialized includes a `lambda`.")


# %% [markdown]
# #### plot graphs

# %%
test_loss, test_acc = model.evaluate(test_generator, verbose=0)

# Predict using the test generator
predictions = model.predict(test_generator)

# Convert probabilities to class labels (assuming binary classification with a threshold of 0.5)
predicted_labels = (predictions > 0.5).astype(int)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)

#Compute cohens kappa
kappa = cohen_kappa_score(y_test, predicted_labels)



# Extracting TP, TN, FP, FN
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

acc = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Save the model with kapp, accuracy, precision, recall, and F1 score om the filename
model.save(f"models/k_{kappa:.4f}_TA_{acc:.4f}_P_{precision:.4f}_R_{recall:.4f}_F_{f1:.4f}.keras")

print(f"k:{kappa} Test accuracy: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, F1 score: {f1:.4f}")

# Create subplots
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# Plot model accuracy
ax[0].plot(history.history['accuracy'], label='Train')
ax[0].plot(history.history['val_accuracy'], label='Validation')
ax[0].set_title('Model accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc='lower right')

# Plot model loss
ax[1].plot(history.history['loss'], label='Train')
ax[1].plot(history.history['val_loss'], label='Validation')
ax[1].set_title('Model loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend(loc='lower right')

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Plume', 'Plume'], 
            yticklabels=['No Plume', 'Plume'], ax=ax[2])
ax[2].set_xlabel('Predicted Label')
ax[2].set_ylabel('True Label')
ax[2].set_title(f'CNN Confusion Matrix\nKappa: {kappa:.4f}  Accuracy: {acc:.4f}\nPrecision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}', fontsize=10)

# Display the plots
plt.show()

# %% [markdown]
# ### Plots with CAM 

# %%
# Recreate your model using the Functional API
inputs = tf.keras.Input(shape=(32, 32, 1))

# Define the tf.keras.layers in your model
x = tf.keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
conv_output = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='last_conv')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv_output)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(70, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create the model
model = tf.keras.Model(inputs, outputs)




# %%

def get_heatmap(cam_model, input_image, scale_positive_factor=2.0):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = cam_model(input_image)
        predicted_label = predictions.numpy()[0, 0]  # Predicted label for the current image

        # Scale the loss for positive predictions
        loss = predictions[:, 0]
        if predicted_label > 0.5:  # Assuming 0.5 as the threshold for positive
            loss *= scale_positive_factor

    # Compute the gradients
    gradients = tape.gradient(loss, conv_outputs)

    # Compute the guided gradients
    pooled_grads = tf.reduce_mean(gradients, axis=(0, 1, 2))

    # Multiply each channel by the pooled gradient
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Compute the heatmap
    heatmap = np.mean(conv_outputs, axis=-1)

    # Apply ReLU to ensure only positive activations are considered
    heatmap = np.maximum(heatmap, 0)

    # Normalize the heatmap between 0 and 1 for visualization
    heatmap /= np.max(heatmap)

    # Upscale the heatmap to match the input image size using bilinear interpolation
    heatmap = cv2.resize(heatmap, (input_image.shape[2], input_image.shape[1]), interpolation=cv2.INTER_LINEAR)

    # Normalize the heatmap to ensure values are between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    return heatmap, predicted_label

# %%
cam_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer('last_conv').output, model.output]
)

# %%
# Define the range of images you want to visualize
start_index = 300
end_index = 500 # Adjust this if you want to limit the number of images

# Loop through the range and only process images with a true label of 1 (Plume)
for image_index in range(start_index, end_index):
    true_label = y_test[image_index]  # True label for the current image
    if true_label == 1:  # Only process images with a plume
        true_label_desc = "Plume"

        # Prepare the image to be passed to the model
        input_image = x1_test[image_index:image_index + 1]

        heatmap, predicted_label = get_heatmap(cam_model, input_image)

        # Display the original image and the heatmap
        plt.figure(figsize=(10, 5), dpi=50)

        # Display the input image
        plt.subplot(1, 2, 1)
        plt.imshow(input_image[0, :, :, 0], cmap='rainbow')
        plt.title(f'True Label: {true_label_desc}')

        # Display the Grad-CAM heatmap overlayed on the input image
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap='viridis', alpha=1)  # Overlay the heatmap on the image
        plt.title(f'True Label: {true_label_desc}')
        plt.show()


# %% [markdown]
# #### Model Recreation for CAM Visualization

# %%
# Compile the model with the named function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# %% [markdown]
# ## Loading In 2020 Data

# %% [markdown]
# ### Loading In Data

# %%
class HDF5ChannelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, hdf5_file, dataset_name, channel_slice, batch_size=32, shuffle=False, **kwargs):
        self.hdf5_file = hdf5_file
        self.dataset_name = dataset_name
        self.channel_slice = channel_slice
        self.batch_size = batch_size
        self.shuffle = shuffle

        with h5py.File(self.hdf5_file, 'r') as hf:
            self.num_samples = hf[self.dataset_name].shape[0]
            self.num_variables = hf[self.dataset_name].shape[1]  # Variables dimension

        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()
        super().__init__(**kwargs)

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        with h5py.File(self.hdf5_file, 'r') as hf:
            # Load the batch data in shape (batch_size, variables, 32, 32)
            batch_data = hf[self.dataset_name][batch_indices, :, :, :]

        # Reshape the data to (batch_size, 32, 32, variables)
        batch_data = np.transpose(batch_data, (0, 2, 3, 1))

        # Slice the specific channel
        batch_data = batch_data[:, :, :, self.channel_slice]

        return batch_data

    def on_epoch_end(self):
        # Shuffle indices after each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

# %% [markdown]
# ### Running Model on 2020 data

# %%
model = tf.keras.models.load_model('models/k_0.9726_TA_0.9892_P_0.9804_R_0.9796_F_0.9800.keras')

batch_size = 15

# Create the test data generator
test_generator2 = HDF5ChannelDataGenerator('2020/all_scenes_combined.h5', 'all_scenes', channel_map['normalized_ch4'], batch_size=batch_size, shuffle=False)

# Generate predictions using the model
predictions_2020 = model.predict(test_generator2)

# Optionally, print or analyze the predictions
print("Predictions:", predictions_2020)


# %% [markdown]
# ### Visualization of Results

# %%
import random

# Set the number of plumes you want to process
num_plumes_to_process = 50

#pd2020[pd2020 <= 0.5] = 0
# Get indices of all plumes
plume_indices = [i for i in range(len(predictions_2020)) if predictions_2020[i] > 0.5]

# Randomly sample indices from the plume_indices list
random_plume_indices = random.sample(plume_indices, num_plumes_to_process)

with h5py.File('2020/all_scenes_combined.h5', 'r') as hf:
    for i in random_plume_indices:
        # Extract the image based on the prediction index
        input_image = hf['all_scenes'][i:i+1, :, :, :]  # Extract the full image
        input_image = np.transpose(input_image, (0, 2, 3, 1))  # Reshape to (batch, 32, 32, variables)
        input_image = input_image[:, :, :, channel_map['normalized_ch4']]  # Slice to get the required channel
        
        heatmap, predicted_label = get_heatmap(cam_model, input_image)

        # Display the original image and the heatmap
        plt.figure(figsize=(10, 5), dpi=50)

        # Display the input image
        plt.subplot(1, 2, 1)
        plt.imshow(input_image[0, :, :, 0], cmap='rainbow')
        plt.title(f'Prediction: {predictions_2020[i]}')

        # Display the Grad-CAM heatmap overlayed on the input image
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap='viridis', alpha=1)  # Overlay the heatmap on the image
        plt.title(f"CAM visualization")
        plt.show()


# %% [markdown]
# #### masks

# %%
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# Function to compute plume shape and wind direction alignment
def compute_plume_shape_and_direction(plume_mask, wind_x, wind_y):
    pca = PCA(n_components=2)
    pca.fit(plume_mask)
    wind_x = np.nanmean(wind_x)
    wind_y = np.nanmean(wind_y)
    
    elongation_ratio = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]

    plume_direction = np.arctan2(pca.components_[0][1], pca.components_[0][0])
    wind_direction = np.arctan2(wind_y, wind_x)

    direction_difference = np.abs(wind_direction - plume_direction)
    return elongation_ratio, direction_difference

# %%
# Function to compute correlations
def compute_correlation(xch4_values, param_values, mask):
    return pearsonr(xch4_values[mask], param_values[mask])[0]


# %%
import numpy as np

def find_plume_source(mask, wind_u, wind_v):
    indices = np.where(mask == True)  # Get indices where mask is high-confidence
    min_dist = np.inf
    source_idx = None
    wind_u[np.isnan(wind_u)] = 0
    wind_v[np.isnan(wind_v)] = 0
    
    # Iterate over all high-confidence pixels
    for idx in zip(indices[0], indices[1]):
        current_pos = np.array(idx)
        wind_vector = -np.array([wind_u[idx], wind_v[idx]])  # Correctly reversed wind direction
        potential_source = current_pos + wind_vector
        
        # Calculate distance to the nearest edge in the direction of wind
        dist = np.linalg.norm(potential_source - current_pos)
        if dist < min_dist:
            min_dist = dist
            source_idx = idx
    
    return source_idx

def compute_correlation_within_mask(mask, param1, param2):
    # Flatten the arrays
    mask_flat = mask.flatten()
    param1_flat = param1.flatten()
    param2_flat = param2.flatten()

    param1_flat[np.isnan(param1_flat)] = 0
    param2_flat[np.isnan(param2_flat)] = 0

    # Ensure the mask has the same shape as the parameters
    if mask_flat.shape != param1_flat.shape or mask_flat.shape != param2_flat.shape:
        raise ValueError("Mask and parameters must have the same shape")

    # Apply the mask
    param1_masked = param1_flat[mask_flat > 0]
    param2_masked = param2_flat[mask_flat > 0]

    # Check if there are enough valid pixels
    if len(param1_masked) < 2 or len(param2_masked) < 2:
        # print(f"Not enough valid pixels: len(param1_masked)={len(param1_masked)}, len(param2_masked)={len(param2_masked)}")
        return np.nan  # or return 0.0 or some other value indicating no correlation
    
    # Compute and return the Pearson correlation
    return pearsonr(param1_masked, param2_masked)[0]

def get_correlations_for_mask(mask,xch4, swir_surface_albedo, aerosol_optical_thickness, chi_squared, surface_pressure):
    xch4_correlation = compute_correlation_within_mask(mask, xch4, xch4)
    swir_surface_albedo_correlation = compute_correlation_within_mask(mask, xch4, swir_surface_albedo)
    aerosol_optical_thickness_correlation = compute_correlation_within_mask(mask, xch4, aerosol_optical_thickness)
    chi_squared_correlation = compute_correlation_within_mask(mask, xch4, chi_squared)
    surface_pressure_correlation = compute_correlation_within_mask(mask, xch4, surface_pressure)

    correlations = {
        'xch4': xch4_correlation,
        'swir_surface_albedo': swir_surface_albedo_correlation,
        'aerosol_optical_thickness': aerosol_optical_thickness_correlation,
        'chi_squared': chi_squared_correlation,
        'surface_pressure': surface_pressure_correlation
    }
    return correlations


# %%
from collections import deque


def custom_dilation(starting_point, threshold, plume_mask, max_depth):
    image_shape = plume_mask.shape
    seed_mask = np.zeros(image_shape, dtype=bool)
    visited = np.zeros(image_shape, dtype=bool)  # To keep track of visited pixels
    seed_mask[starting_point] = True
    visited[starting_point] = True
    
    # Each element in the queue stores the pixel and the current depth
    queue = deque([(starting_point, 0)])  # Start with depth 0
    
    # Directions for 8-connectivity (N, S, E, W, NE, NW, SE, SW)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        current_pixel, current_depth = queue.popleft()
        
        # Stop processing if maximum depth is reached
        if current_depth >= max_depth:
            continue
        
        for d in directions:
            neighbor = (current_pixel[0] + d[0], current_pixel[1] + d[1])
            # Check boundaries
            if 0 <= neighbor[0] < image_shape[0] and 0 <= neighbor[1] < image_shape[1]:
                # Check if neighbor meets threshold and hasn't been visited
                if plume_mask[neighbor] >= threshold and not visited[neighbor]:
                    visited[neighbor] = True
                    seed_mask[neighbor] = True
                    # Add the neighbor with incremented depth
                    queue.append((neighbor, current_depth + 1))

    return seed_mask

# %%
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.stats import pearsonr
import numpy as np
import h5py
from scipy.ndimage import binary_erosion
from sklearn.decomposition import PCA
import random

# Set the number of plumes you want to process
num_plumes_to_process = 150

# Get indices of all plumes
plume_indices = [i for i in range(len(predictions_2020)) if predictions_2020[i] > 0.5]

# Randomly sample indices from the plume_indices list
random_plume_indices = random.sample(plume_indices, num_plumes_to_process)

with h5py.File('2020/all_scenes_combined.h5', 'r') as hf:
    for i in plume_indices:
        # Process each image in the batch
        input_image = hf['all_scenes'][i:i+1, :, :, :]
        input_image = np.transpose(input_image, (0, 2, 3, 1))

        cam, predicted_label = get_heatmap(cam_model, input_image[:, :, :, channel_map['normalized_ch4']])
        xch4 = input_image[0, :, :, channel_map['xch4_corrected']]
        xch4_non_corrected = input_image[0, :, :, channel_map['xch4']]
        norm_xch4 = input_image[0, :, :, channel_map['normalized_ch4']]
        u10 = input_image[0, :, :, channel_map['u10']][:, :, 0]
        v10 = input_image[0, :, :, channel_map['v10']][:, :, 0]
        swir_surface_albedo = input_image[0, :, :, channel_map['surface_albedo']][:, :, 1]
        aerosol_optical_thickness = input_image[0, :, :, channel_map['aerosol_optical_thickness']][:, :, 1]
        chi_squared = input_image[0, :, :, channel_map['chi_squared']][:, :, 0]
        surface_pressure = input_image[0, :, :, channel_map['surface_pressure']]


        # Calculate methane enhancement
        thresh = (np.nanmean(xch4) - np.nanstd(xch4))
        test = xch4.copy()

        rel = xch4.copy() - thresh
        #generating plume mask
        test[test < thresh] = 0
        test[test> thresh] = 1
        plume_mask = test[:,:,0].astype(np.uint8)*cam


        # Define starting point
        starting_point = np.unravel_index(np.argmax(plume_mask), plume_mask.shape)

        # Define a structural element that includes diagonal connectivity
        structure = generate_binary_structure(2, 2)  # 2D connectivity, 2 for including diagonals

        # High Threshold mask

        high_conf_threshold = np.mean(plume_mask) + np.std(plume_mask) * 1.8
        high_conf_mask_ = (plume_mask > high_conf_threshold).astype(np.uint8)


    
        # Low Threshold mask
        low_conf_threshold = np.mean(plume_mask) + np.std(plume_mask) * 0.8
        low_conf_mask_ = (plume_mask >= low_conf_threshold).astype(np.uint8)

        
        high_conf_mask = custom_dilation(starting_point, high_conf_threshold, high_conf_mask_, 5)
        low_conf_mask = custom_dilation(starting_point, low_conf_threshold, plume_mask, 10)
        low_conf_mask_2 = custom_dilation(starting_point, low_conf_threshold, plume_mask, 10)


        """
        If an enchancement in the xch4 field is caused by a surface(albedo) feature 
        or by (enchanced) scattering in the atomsphere, which is represented by the
        aerosol optical thickness, then we expect their a spatial patterns to be 
        similar. Therefore, we calculate the correlation between the xch4 field and
        the surface albedo(SWIR), aerosol optical thickness, chi squared
        (an indicator) for retrieval fit quality, and the surface pressure across the
        plume mask. We calculate these correlations for both the high and low confidence plume
        masks, 1- and 2-times dilated versions of the low confidence mask, and the entire image. 
        """

        high_conf_mask_correlations = get_correlations_for_mask(high_conf_mask, xch4, swir_surface_albedo, aerosol_optical_thickness, chi_squared, surface_pressure)

        low_conf_mask_correlations = get_correlations_for_mask(low_conf_mask, xch4, swir_surface_albedo, aerosol_optical_thickness, chi_squared, surface_pressure)

        low_conf_mask_2_correlations = get_correlations_for_mask(low_conf_mask_2, xch4, swir_surface_albedo, aerosol_optical_thickness, chi_squared, surface_pressure)

        all_mask_correlations = get_correlations_for_mask(np.ones_like(xch4), xch4, swir_surface_albedo, aerosol_optical_thickness, chi_squared, surface_pressure)

        """
        Another major indicator for artificats is a mismatch between the direction of the plume and
        the direction of the wind field. By applying a principal component analysis to the plume mask, we compute the two main axes of the pixels in the high confidence plume mask, after re-projecting the pixel centers to meter space and weighting them by there enchancement relative to the background. We use the ratio of the explained variance of the two axes as a measure of the elongation of the plume. We then compute the angle between the main axis of the plume and the wind direction(averaged across the plume mask) the smaller the difference, the more likely the plume is real. We also use the wind field to estimate the source of the plume by taking the most upwind pixel in the high confidence plume mask and moving it in the opposite direction of the wind field. The pixel with the shortest distance to the plume mask is considered the source of the plume.
        """

        # Compute plume shape and wind direction alignment
        elongation_ratio, direction_difference = compute_plume_shape_and_direction(high_conf_mask, u10, v10)

        # Find the plume source
        source_idx = find_plume_source(high_conf_mask, u10, v10)
        
        if source_idx is None:
            pass
        else:
            if high_conf_mask_correlations['swir_surface_albedo'] < 0.9 and high_conf_mask_correlations['aerosol_optical_thickness'] < 0.9 and high_conf_mask_correlations['chi_squared'] < 0.9 and high_conf_mask_correlations['surface_pressure'] < 0.9 and direction_difference < 1.7:

                print(f"elongation ratio: {elongation_ratio}\ndirection difference:{direction_difference}\n\nhigh_conf_corelations: {high_conf_mask_correlations}")#\n\nlow_conf_corelations: {low_conf_mask_correlations}\n\nlow_conf_2_corelations:  #{low_conf_mask_2_correlations}\n\nall_corelations: {all_mask_correlations}")

                subplots = 3
                fig, ax = plt.subplots(1, subplots, figsize=(5*subplots, 5),dpi = 80)
                im0 = ax[0].imshow(xch4, cmap='rainbow')
                ax[0].invert_yaxis()
                ax[0].set_xticks(np.arange(0, 32, 5))
                cbar0 = plt.colorbar(im0, ax=ax[0], location='right', shrink = 0.7, pad=0.06)  
                cbar0.ax.set_title('[ppb]', loc='center', fontsize=12)

                im1 = ax[1].imshow(cam, cmap='viridis')
                ax[1].invert_yaxis()
                ax[1].axis('off')  # Hides the axis
                cbar1 = plt.colorbar(im1, ax=ax[1], location='right',shrink = 0.7, pad=0.06)  #
                cbar1.set_ticks(np.arange(0,1,0.2))
        
                # xch4 relative to the background
                im2 = ax[2].imshow(rel, cmap='rainbow')
                
                ax[2].imshow(high_conf_mask, cmap='grey', alpha=0.6)
                ax[2].plot(source_idx[1], source_idx[0], 'kx', markersize=10)
                ax[2].invert_yaxis()
                cbar2 = plt.colorbar(im2, ax=ax[2], location='right', shrink = 0.7, pad=0.06)
                cbar2.ax.set_title('[ppb]', loc='center', fontsize=12)
                ax[2].set_xticks(np.arange(0, 32, 5))
                plt.show()

        if i == 15:
            break





            
  


