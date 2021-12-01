import tensorflow as tf
import flask
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from tensorflow.keras import models
import numpy as np
import imageio
import os
from keras.preprocessing import image

flask_app = flask.Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "ML React App", 
		  description = "Predict results using a trained model")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'file': fields.String(required = True, 
				  							   description="Image to predict", 
    					  				 	   help="Image cannot be blank")})

# classifier = joblib.load('classifier.joblib')
loaded_model = models.load_model('benign_malignant_tuning10.h5')
# graph = tf.get_default_graph()

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		#try: 				
		imagefile = request.files.get('file','')
		filename = werkzeug.utils.secure_filename(imagefile.filename)
		print("\nReceived image file name : " + imagefile.filename)
		imagefile.save(filename)
		img = image.load_img(filename, target_size=(100, 100))
		img_array = image.img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)
		# x = preprocess_input(x)

		print('predicting...')
# 		with graph.as_default():
		predictions = loaded_model.predict(img_array)
		print('done')
		print('predicted', predictions)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		class_names = ['benign', 'malignant']
		score = tf.nn.softmax(predictions[0])
		predicted_label = str("This image most likely belongs to {} with a {:.2f} percent confidence."
    		.format(class_names[np.argmax(score)], 100 * np.max(score)))
		print('Predicted:', )

		response = jsonify({
			"statusCode": 200,
			"status": "Prediction made",
			"result": predicted_label  # str(data)
			})
		response.headers.add('Access-Control-Allow-Origin', '*')
		return response
flask_app.run(host="127.0.0.1", port=os.environ.get('PORT', 5000), debug=True)