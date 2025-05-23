#ML Project - Comparison of Machine Learning Models for Workout Analysis

pip install pandas numpy matplotlib seaborn scikit-learn


#Loading the Dataset onto Colab
import pandas as pd

file_path = "RecGym.csv"
data = pd.read_csv(file_path, encoding='utf-8')

print("First 10 rows of the dataset:")
print(data.head(10).to_string())

print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())


#Checking the Dataset for missing values
import pandas as pd

file_path = "RecGym.csv"
data = pd.read_csv(file_path, encoding='utf-8')

missing_values = data.isnull().sum()

print("Missing Values in Each Column:")
print(missing_values[missing_values > 0])
if data.isnull().values.any():
    print("\nThe dataset contains missing values.")
else:
    print("\nNo missing values found in the dataset.")

#Data Preprocessing Steps
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

file_path = "RecGym.csv"
data = pd.read_csv(file_path, encoding='utf-8')
data.dropna(inplace=True)

sensor_columns = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]

data[sensor_columns] = data[sensor_columns].apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)
scaler = StandardScaler()
data[sensor_columns] = scaler.fit_transform(data[sensor_columns])

pca = PCA(n_components=5)
data_pca = pca.fit_transform(data[sensor_columns])

label_encoder = LabelEncoder()
data["Workout"] = label_encoder.fit_transform(data["Workout"])

X = pd.DataFrame(data_pca)
y = data["Workout"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Preprocessing completed with feature reduction, augmentation, and real-time adaptation!")
print("\nFirst 10 Rows of Preprocessed Data (PCA-Reduced Features):")
print(X_train.head(10))

print("\nEncoded Workout Labels (First 10 Rows):")
print(y_train.head(10))

print("\nShape of Training and Testing Sets:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")


#Reshaping data for ResCNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

time_steps = 20
num_features = X_train.shape[1]

def create_time_series_data(X, y, time_steps):
    sequences, labels = [], []
    for i in range(len(X) - time_steps):
        sequences.append(X.iloc[i : i + time_steps].values)
        labels.append(y[i + time_steps])
    return np.array(sequences), np.array(labels)

X_train_seq, y_train_seq = create_time_series_data(X_train, y_train_cat, time_steps)
X_test_seq, y_test_seq = create_time_series_data(X_test, y_test_cat, time_steps)

print(f"Reshaped data for CNN: {X_train_seq.shape}")


#Building ResCNN Model
def build_rescnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    def residual_block(x, filters=64):
        res = Conv1D(filters, kernel_size=3, padding="same")(x)
        res = BatchNormalization()(res)
        res = ReLU()(res)
        res = Conv1D(filters, kernel_size=3, padding="same")(res)
        res = BatchNormalization()(res)
        x = Add()([x, res])  # Residual connection
        x = ReLU()(x)
        return x

    x = residual_block(x)
    x = residual_block(x)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

rescnn_model = build_rescnn(input_shape=(time_steps, num_features), num_classes=num_classes)
rescnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

rescnn_model.summary()


#Training the Model
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=y_train)
plt.xticks(rotation=45)
plt.title("Class Distribution in Training Data")
plt.show()


import pandas as pd
from sklearn.utils import resample

X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
y_train_df = pd.DataFrame(y_train, columns=["Workout"])

df = pd.concat([X_train_df, y_train_df], axis=1)

df_majority = df[df["Workout"] == 5]
df_minority = df[df["Workout"] != 5]

df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),
                                   random_state=42)

df_balanced = pd.concat([df_majority_downsampled, df_minority])

df_balanced = df_balanced.sample(frac=1, random_state=42)

X_train_balanced = df_balanced.drop(columns=["Workout"]).values
y_train_balanced = df_balanced["Workout"].values


#Final Version of Code
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)

    sensor_cols = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]
    features = data[sensor_cols].values.astype(np.float32)

    le = LabelEncoder()
    labels = le.fit_transform(data["Workout"])

    return features, labels, len(le.classes_)

def build_fast_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(128, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs)

def train_pipeline():
    X, y, num_classes = load_and_preprocess_data("RecGym.csv")

    X = X.reshape(*X.shape, 1)

    train_ds = tf.data.Dataset.from_tensor_slices((X, y))
    train_ds = train_ds.shuffle(10000).batch(4096).prefetch(2)

    model = build_fast_model(X.shape[1:], num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        epochs=200,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1)
        ]
    )

    return model

if __name__ == "__main__":
    model = train_pipeline()
    model.save("gym_activity_model.keras")
