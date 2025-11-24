import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Attention, TimeDistributed

def create_attention_model(T_in, T_out, F, n_units):
   
    encoder_inputs = Input(shape=(T_in, F), name='encoder_input')
    
    encoder_outputs, state_h, state_c = LSTM(
        n_units, 
        activation='tanh', 
        return_sequences=True,
        return_state=True,
        name='encoder_lstm'
    )(encoder_inputs)
    encoder_states = [state_h, state_c] 

    decoder_inputs = Input(shape=(T_out, F), name='decoder_input') 
    
    decoder_lstm = LSTM(
        n_units, 
        activation='tanh',
        return_sequences=True, 
        return_state=True, 
        name='decoder_lstm'
    )
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs, 
        initial_state=encoder_states
    )

    attention_output, attention_weights = Attention(
        name='attention_layer'
    )([decoder_outputs, encoder_outputs], return_attention_scores=True)
    
    concat = Concatenate(axis=-1, name='concat_attention_and_decoder')([decoder_outputs, attention_output])

    decoder_dense = Dense(1) 
    output = TimeDistributed(decoder_dense, name='prediction_output')(concat) 

    model = Model(
        inputs=[encoder_inputs, decoder_inputs], 
        outputs=[output, attention_weights], 
        name='Seq2Seq_Attention_Visualization_Model'
    )
    
    return model