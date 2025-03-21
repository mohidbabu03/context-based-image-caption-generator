from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import pickle
import uuid

app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Xception model and caption generator
base_model = Xception(weights='imagenet')
xception_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
caption_model = pickle.load(open('best_m.sav', 'rb'))  # Update path as needed
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Parameters
max_length = 35

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_context_based_caption(image_path, context_text, tokenizer, model, max_length):
    image = load_img(image_path, target_size=(299, 299))
    image_array = img_to_array(image)
    image = np.expand_dims(image_array, axis=0)
    image = preprocess_input(image)

    image_feature = xception_model.predict(image, verbose=0)
    in_text = f"startseq {context_text}" if context_text else "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += f" {word}"

    return in_text.replace('startseq', '').strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    context_text = request.form.get('context', '')

    filename = str(uuid.uuid4()) + os.path.splitext(image.filename)[-1]
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    caption = generate_context_based_caption(image_path, context_text, tokenizer, caption_model, max_length)
    return jsonify({
        "image_url": f"/uploads/{filename}",
        "caption": caption
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)
