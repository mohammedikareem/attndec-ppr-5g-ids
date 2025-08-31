import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, 
                                     MultiHeadAttention, LayerNormalization,
                                     Reshape, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam

def create_attention_model(n_features):
    main_input = Input(shape=(n_features,))
    reshaped = Reshape((1, n_features))(main_input)
    attn = MultiHeadAttention(num_heads=2, key_dim=n_features)(reshaped, reshaped)
    attn = LayerNormalization()(attn + reshaped)
    pooled = GlobalAveragePooling1D()(attn)
    x = Dense(128, activation='relu')(pooled)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ])
    return model

def create_mlp_model(n_features):
    main_input = Input(shape=(n_features,))
    x = Dense(128, activation='relu')(main_input)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ])
    return model
