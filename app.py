from flask import Flask, render_template, request
import numpy as np
import pickle
from dependancies import standardizing
from dependancies import detect_and_change_separator
from werkzeug.utils import secure_filename
import os
import tensorflow
scaler = standardizing()

app = Flask(__name__)

# Load the model
model = pickle.load(open('NN_saved_model.sav', 'rb'))

# # Load the CNN model
# cnnModel = pickle.load(open("CnnTrained_model2.sav", 'rb'))
# # with open('CnnTrained_model (1).sav', 'rb') as f:
# #     cnnModel = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', **locals())


@app.route('/predict' , methods=['POST', 'GET'])
def predict():
   if request.method == 'POST':
#   form_id = request.form.get('form_id')
#   if form_id == "form1":
    inputed_data = request.form['inputed_data']

    # change separator to comma
    inputed_data = detect_and_change_separator(inputed_data)

    # Split the string into a list of values
    values_list = inputed_data.split(",")

    # Convert the list of strings to a list of floats
    values_list = [float(value) for value in values_list]
    input_data = tuple(values_list)
    print(input_data)

    # change the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardizing the input data
    input_data_std = scaler.transform(input_data_reshaped)
    prediction = model.predict(input_data_std)
    prediction_label = [np.argmax(prediction)]

    if(prediction_label[0] == 0):
        result = 'The tumor is Benin'

    else:
        result = 'The tumor is Malignant'

   return render_template('index.html', **locals())

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
        
#         file = request.files['file']
        
#         # If the user does not select a file, the browser submits an empty file without a filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
        
#         # Check if the file is allowed
#         # You might want to add more sophisticated file type checking
#         allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
#         if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
#             flash('Invalid file type')
#             return redirect(request.url)

#         # Create the "uploads" folder if it doesn't exist
#         upload_folder = 'uploads'
#         os.makedirs(upload_folder, exist_ok=True)
#         # Save the file
#         file.save('uploads/' + secure_filename(file.filename))
#         # benin_image_path = "./uploads"
#         # data_dir = pathlib.Path(benin_image_path) / secure_filename(file.filename)
#         # img = tf.keras.utils.load_img(
#         #     data_dir,target_size =(img_height, img_width)
#         # )
#         # img_array = tf.keras.utils.img_to_array(img)
#         # img_array = tf.expand_dims(img_array, 0) # Create a batch

#         # predictions = cnnModel.predict(img_array)
#         # score = tf.nn.softmax(predictions[0])
#         # class_names = ['Benin', 'Maligant']

#         # print(
#         #     "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100*np.max(score))
#         # )

        
#         # Perform actions with the file here (e.g., image processing)

#     return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug = True)

