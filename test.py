import tensorflow as tf
from keras import backend as K
import gc

def cleanup_gpu_memory():
    """Cleanup GPU memory and release resources"""
    
    # Clear Keras/TF backend
    K.clear_session()
    tf.keras.backend.clear_session()
    
    # Reset all graph states
    tf.compat.v1.reset_default_graph()
    
    # Force garbage collection
    gc.collect()
    
    # Release CUDA memory
    if tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    
if __name__ == '__main__':
    cleanup_gpu_memory()