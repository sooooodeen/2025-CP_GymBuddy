import tensorflow as tf
import numpy as np
import os
import sys

# --- CONFIGURATION ---
# SCRIPT_DIR is the 'training' folder itself
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/Output Files
# Paths are relative to the script's location (the 'training' folder)
MODEL_FILENAME = os.path.join(SCRIPT_DIR, 'exercise_classifier_lstm.h5')
TFLITE_MODEL_FILENAME = os.path.join(SCRIPT_DIR, 'exercise_classifier_quant.tflite')
CALIBRATION_DATA_FILENAME = os.path.join(SCRIPT_DIR, 'calibration_features.npy') 

# Model Parameters (must match your LSTM training)
SEQUENCE_LENGTH = 90
FEATURE_SIZE = 8 # Number of angles extracted per frame

def representative_dataset_gen():
    """
    Generator function that loads and yields actual data samples from your training set.
    """
    print(f"Attempting to load calibration data from: {CALIBRATION_DATA_FILENAME}")
    
    try:
        # ⚠️ CRITICAL FIX: Use np.load() on a file-like object to bypass some file integrity checks 
        # that were causing issues, while still enforcing allow_pickle=True.
        # This is the most reliable way to load the data you just created.
        with open(CALIBRATION_DATA_FILENAME, 'rb') as f:
            calibration_data = np.load(f, allow_pickle=True).astype(np.float32)
        
        # Use a max of 100-200 samples for fast calibration
        calibration_data = calibration_data[:200]
        print(f"Loaded {len(calibration_data)} calibration sequences for quantization.")
        
    except FileNotFoundError:
        # ... (error handling remains the same)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR loading calibration data: {e}. Please ensure the file was created correctly.")
        # If the error still occurs, your only option is to delete the file and create it again 
        # using the provided creation script, as the file itself is malformed.
        sys.exit(1)

    for input_value in calibration_data:
        # TFLite interpreter expects a batch size of 1, so ensure the input is [1, 90, 8]
        if input_value.ndim == 2:
            input_value = np.expand_dims(input_value, axis=0)
        yield [input_value]

def convert_to_tflite():
    """Converts the Keras H5 model to a TFLite model with INT8 weights (Weight-Only Quantization)."""
    
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME)
        print("Keras model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR loading Keras model: {e}")
        return

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # --- ⚠️ FINAL, STABLE CONVERSION FOR LSTMs ⚠️ ---
    
    # 1. Enable default optimizations (performs weight-only quantization: INT8 weights, Float I/O)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 2. Crucial: Remove representative_dataset. This prevents the script from calling np.load()
    # and stopping on the File I/O error.
    # converter.representative_dataset = representative_dataset_gen # <- REMOVED!
    
    # 3. Use TFLITE_BUILTINS AND SELECT_TF_OPS (Required for LSTM Flex support)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # 4. Disable the experimental flag explicitly (Fixes the TensorListReserve error)
    converter._experimental_lower_tensor_list_ops = False
    
    # 5. I/O types remain Float32 for stability
    converter.inference_input_type = tf.float32 
    converter.inference_output_type = tf.float32 
    
    print("--- Starting FINAL TFLite Conversion (Weight-Only Quantization) ---")
    
    try:
        quantized_tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(TFLITE_MODEL_FILENAME, "wb") as f:
            f.write(quantized_tflite_model)
        
        print(f"✅ Conversion successful! Optimized TFLite model saved to {TFLITE_MODEL_FILENAME}")
        print("This model uses Weight-Only Quantization for maximum stability and speed.")
    
    except Exception as e:
        print(f"❌ TFLite Conversion failed: {e}")
    
if __name__ == '__main__':
    convert_to_tflite()