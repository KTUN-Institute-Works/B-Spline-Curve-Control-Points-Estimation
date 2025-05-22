import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from scipy.io import arff

class CPClassifier:
    def __init__(self):
        # Veri yükleme
        self.data, meta = arff.loadarff("dataset.arff")
        self.df = pd.DataFrame(self.data)
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.mask_train = None
        self.mask_test = None
        self.scaler_x = None
        self.scaler_y = None
        self.model = None
    def set_inout_data(self):

        # Girdi ve çıktı sütunlarını ayır
        input_cols = [col for col in self.df.columns if col.startswith("x") or col.startswith("y")]
        output_cols = [col for col in self.df.columns if col.startswith("dotx") or col.startswith("doty")]

        X = self.df[input_cols].astype(np.float32).values
        Y = self.df[output_cols].astype(np.float32).values

        # -1 olan kontrol noktalarını 0'a çek, mask olarak kaydet
        mask = (Y == -1).astype(np.float32)
        Y[Y == -1] = 0.0

        # Normalize X ve Y
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.X = self.scaler_x.fit_transform(X)
        self.Y = self.scaler_y.fit_transform(Y)

        # Veri bölme
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.mask_train, self.mask_test = train_test_split(X, Y, mask, test_size=0.2, random_state=42)

    def create_model(self):
        # Model oluştur
        self.model = keras.Sequential([
            layers.Input(shape=(self.X.shape[1],)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.Y.shape[1], activation='linear')  # Regressor
        ])

        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

        # Eğitim
        self.model.fit(self.X_train, self.Y_train, epochs=200, batch_size=16, validation_split=0.1)

        # ❗️Tahmin yap ve ters dönüşüm ile orijinal ölçeğe dön
        Y_pred = self.model.predict(self.X_test)

        # Ters dönüşüm (MinMaxScaler'dan çıkmak için)
        Y_pred_original = self.scaler_y.inverse_transform(Y_pred)
        Y_test_original = self.scaler_y.inverse_transform(self.Y_test)

        # Opsiyonel: Tahminleri sadece geçerli (mask=0) değerler üzerinde değerlendir
        valid_mask = (self.mask_test == 0)

        # Ortalama kare hata sadece geçerli noktalar için
        mse = np.mean((Y_pred_original[valid_mask == 1] - Y_test_original[valid_mask == 1]) ** 2)
        print(f"Masked MSE (sadece geçerli kontrol noktaları): {mse:.4f}")

    def predict(self, input_data:list)->list:

        input_data.append(0)
        input_data.append(0)
        print(input_data)
        input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)

        predicted_control_points = self.model.predict(input_array)  # şekli (1, 30)
        control_points = predicted_control_points[0].reshape(-1, 2)  # (15, 2)
        valid_points = control_points[~np.any(control_points == -1, axis=1)]

        return valid_points

    def masked_mse(self, y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)  # Sıfır olan yerler maskelenecek
        mse = tf.square(y_true - y_pred)
        masked_mse = tf.reduce_sum(mask * mse) / tf.reduce_sum(mask)
        return masked_mse

    # predict metodunda bu iş yeterli değil çünkü model -1 tahmin etmeyecek zaten
    # onun yerine ilk N kontrol noktası geçerli diyecek başka bir yöntem düşün

    # Örneğin, predict fonksiyonuna geçerli nokta sayısını da verirsen:
    def predict_(self, input_data: list, num_valid_points: int) -> list:
        input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)
        predicted_control_points = self.model.predict(input_array)
        control_points = predicted_control_points[0].reshape(-1, 2)
        return control_points[:num_valid_points]  # sadece geçerli noktalar
