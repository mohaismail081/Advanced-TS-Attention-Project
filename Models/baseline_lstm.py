import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense # Note: 'Input' is removed here

def create_baseline_lstm(T_in, T_out, F, n_units):
    """
    Defines a simple Stacked LSTM model for baseline comparison.
    (Uses the input_shape parameter to avoid the NameError)
    """
    
    # We revert to passing input_shape to the first LSTM layer.
    # This avoids the NameError by not using the globally undefined 'Input' function.
    model = Sequential([
        # Layer 1: Pass input_shape directly (uses tanh and is numerically stable)
        LSTM(n_units, activation='tanh', input_shape=(T_in, F), return_sequences=True), 
        
        # Layer 2: (tanh activation)
        LSTM(n_units, activation='tanh', return_sequences=False), 
        
        # Output Layer
        Dense(T_out * 1) 
        
    ], name='Baseline_Stacked_LSTM_Model')
    
    return model