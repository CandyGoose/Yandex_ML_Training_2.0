import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    W_s = np.dot(W_mult, encoder_hidden_states)
    attention_scores = np.dot(decoder_hidden_state.T, W_s)
    weights_vector = softmax(attention_scores)
    attention_vector = np.dot(encoder_hidden_states, weights_vector.T)
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    n_features_int = v_add.shape[0] 

    z_enc = np.dot(W_add_enc, encoder_hidden_states)
    z_dec = np.dot(W_add_dec, decoder_hidden_state)
    z_dec_expanded = np.repeat(z_dec, encoder_hidden_states.shape[1], axis=1)
    z_combined = z_enc + z_dec_expanded
    z_tanh = np.tanh(z_combined)
    
    attention_scores = np.dot(v_add.T, z_tanh)
    
    weights_vector = softmax(attention_scores)
   
    attention_vector = np.dot(encoder_hidden_states, weights_vector.T)
    
    return attention_vector
