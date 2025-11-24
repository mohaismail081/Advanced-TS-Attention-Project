import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Attention, TimeDistributed

def create_attention_model(T_in, T_out, F, n_units):
    """
    Defines the Sequence-to-Sequence model with Attention, modified to output attention weights.
    """
    
    # --- 1. Define Encoder ---
    encoder_inputs = Input(shape=(T_in, F), name='encoder_input')
    
    # Use tanh activation for numerical stability
    encoder_outputs, state_h, state_c = LSTM(
        n_units, 
        activation='tanh', 
        return_sequences=True,
        return_state=True,
        name='encoder_lstm'
    )(encoder_inputs)
    encoder_states = [state_h, state_c] 

    # --- 2. Define Decoder ---
    decoder_inputs = Input(shape=(T_out, F), name='decoder_input') 
    
    decoder_lstm = LSTM(
        n_units, 
        activation='tanh', # Use tanh activation
        return_sequences=True, 
        return_state=True, 
        name='decoder_lstm'
    )
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs, 
        initial_state=encoder_states
    )

    # --- 3. Attention Mechanism (CRITICAL MODIFICATION HERE) ---
    # The Attention layer can now be configured to return the weights themselves.
    # The output of the Attention layer is now a list: [attention_context_vector, attention_weights]
    attention_output, attention_weights = Attention(
        name='attention_layer'
        # Setting use_scale=False for compatibility with older Keras versions
    )([decoder_outputs, encoder_outputs], return_attention_scores=True)
    
    # Concatenate the attention context vector with the decoder outputs
    # The context vector is the first item in the attention_output list
    concat = Concatenate(axis=-1, name='concat_attention_and_decoder')([decoder_outputs, attention_output])

    # --- 4. Output Layer ---
    decoder_dense = Dense(1) 
    # Rename for clarity when defining losses later
    output = TimeDistributed(decoder_dense, name='prediction_output')(concat) 

    # --- 5. Final Model (CRITICAL MODIFICATION HERE) ---
    # The model now outputs two items: the prediction and the attention weights
    model = Model(
        inputs=[encoder_inputs, decoder_inputs], 
        outputs=[output, attention_weights], 
        name='Seq2Seq_Attention_Visualization_Model'
    )
    
    return model