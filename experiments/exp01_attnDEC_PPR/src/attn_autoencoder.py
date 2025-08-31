import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, MultiHeadAttention,
                                     Flatten, LayerNormalization, Reshape, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def build_attn_autoencoder(input_dim, latent_dim=8, heads=2, key_dim=8, dropout=0.15):
    inp = Input(shape=(input_dim,))
    seq = Reshape((input_dim, 1))(inp)                         # (B, L, 1)
    attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim)(seq, seq)
    attn = Dense(1)(attn)                                      # إسقاط للرؤوس إلى قناة واحدة
    attn = Flatten()(attn)                                     # (B, L)
    attn = LayerNormalization()(attn)
    x = Add()([inp, attn])                                     # Residual
    x = Dense(16, activation='relu')(x)
    x = Dropout(dropout)(x)
    bottleneck = Dense(latent_dim, activation='relu', name='latent')(x)
    x = Dense(16, activation='relu')(bottleneck)
    x = Dropout(dropout)(x)
    out = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inp, out)
    encoder = Model(inp, bottleneck)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def fit_autoencoder(autoencoder, X_train, X_val, epochs=8, batch_size=256, patience=2, verbose=2):
    cb = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    autoencoder.fit(
        X_train, X_train,
        epochs=epochs, batch_size=batch_size, shuffle=True,
        validation_data=(X_val, X_val),
        verbose=verbose, callbacks=[cb]
    )
