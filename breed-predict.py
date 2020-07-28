from IPython.display import display, Image
import os
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import base64
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)


def get_model():
    global model
    model_path = r".\Datos\ModeloCompleto.h5"
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
    return model

# Define the batch size, 32 is a good default
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch
    
    return data_batch

# Define image size
IMG_SIZE = 224

def process_image(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    # Read in image file
    #tf.io.decode_base64(image_path, name=None)
    image = tf.io.decode_base64(image_path)
    
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize the image to our desired size (224, 244)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    image = tf.expand_dims(image, axis=0)
    return image

# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

# Get custom image prediction labels
def show_results(custom_preds):
    #Get labels of the breeds
    labels_csv = pd.read_csv(r".\Datos\labels.csv")
    labels = labels_csv["breed"].to_numpy() # convert labels column to NumPy array
    
    # Find the unique label values
    unique_breeds = np.unique(labels)
    array = np.argsort(custom_preds)[0][-3:]
    
    breeds = unique_breeds[array]
    
    probs = []
    final = []
    for i in range(len(array)):
        probs.append("%.2f" % (100*custom_preds[0][array[i]]))
    
    for i in range(len(array)):
        final.append(str(probs[i])+"% :"+breeds[i])
       
    return final

#Get labels of the breeds
labels_csv = pd.read_csv(r".\Datos\labels.csv")
labels = labels_csv["breed"].to_numpy() # convert labels column to NumPy array



#print(" * Loading Keras model...")
get_model()


@app.route("/predecir", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    #decoded = base64.b64decode(encoded)
    #image = Image.open(io.BytesIO(decoded))
    
    processed_image = process_image(encoded)
    
    prediction = model.predict(processed_image)
    
    results = np.array(show_results(prediction)).tolist()

    response = {
        'prediction': {
            'tres':results[0],
            'dos':results[1],
            'uno':results[2],
        }
    }
    return jsonify(response)











