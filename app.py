import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from util import base64_to_pil
from pixellib.instance import custom_segmentation

# Declare a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/mask_rcnn_model_buttsqr.h5'
MODEL_PATH = 'models/mask_rcnn_model_tag.h5'

# Load your own trained model
segment_image = custom_segmentation()
segment_image.inferConfig(
    num_classes= 1,
    # class_names= ["BG", "butterfly", "squirrel"]
    class_names= ["BG", "tag"]
)
segment_image.load_model(MODEL_PATH)

print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to uploads
        img.save("uploads/image.jpg")

        # Make prediction
        print('Make prediction')
        segment_image.segmentImage(
            "uploads/image.jpg",
            show_bboxes=True,
            output_image_name="results/sample_out.jpg"
        )

        # Return saved status after prediction.
        return jsonify(result='Segmentation done! Check out into results/ folder')

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)
