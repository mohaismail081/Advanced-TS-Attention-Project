import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_baseline_lstm(T_in, T_out, F, n_units):
    
    model = Sequential([
        LSTM(n_units, activation='tanh', input_shape=(T_in, F), return_sequences=True), 
        
        LSTM(n_units, activation='tanh', return_sequences=False), 
        
        Dense(T_out * 1) 
        
    ], name='Baseline_Stacked_LSTM_Model')
    
    return model