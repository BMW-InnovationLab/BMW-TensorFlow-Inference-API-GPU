import os
import jsonschema
import asyncio
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from object_detection.utils import label_map_util
from inference.base_inference_engine import AbstractInferenceEngine
from inference.exceptions import InvalidModelConfiguration, InvalidInputData, ApplicationError


class InferenceEngine(AbstractInferenceEngine):

	def __init__(self, model_path):
		self.label_path = ""
		self.NUM_CLASSES = None
		self.sess = None
		self.label_map = None
		self.categories = None
		self.category_index = None
		self.detection_graph = None
		self.image_tensor = None
		self.d_boxes = None
		self.d_scores = None
		self.d_classes = None
		self.num_d = None
		self.font = ImageFont.truetype("/main/fonts/DejaVuSans.ttf", 20)
		super().__init__(model_path)

	def load(self):
		with open(os.path.join(self.model_path, 'config.json')) as f:
			data = json.load(f)
		try:
			self.validate_json_configuration(data)
			self.set_model_configuration(data)
		except ApplicationError as e:
			raise e

		self.label_path = os.path.join(self.model_path, 'object-detection.pbtxt')
		self.label_map = label_map_util.load_labelmap(self.label_path)
		self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
																		 max_num_classes=self.NUM_CLASSES,
																		 use_display_name=True)
		for dict in self.categories:
			self.labels.append(dict['name'])

		self.category_index = label_map_util.create_category_index(self.categories)
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(os.path.join(self.model_path, 'frozen_inference_graph.pb'), 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
			self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
			self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
			self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
			self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
			self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
		self.sess = tf.Session(graph=self.detection_graph)
		img = Image.open("object_detection/image1.jpg")
		img_expanded = np.expand_dims(img, axis=0)
		self.sess.run(
			[self.d_boxes, self.d_scores, self.d_classes, self.num_d],
			feed_dict={self.image_tensor: img_expanded})

	async def infer(self, input_data, draw, predict_batch):
		await asyncio.sleep(0.00001)
		try:
			pillow_image = Image.open(input_data.file).convert('RGB')
			np_image = np.array(pillow_image)
		except Exception as e:
			raise InvalidInputData('corrupted image')
		try:
			with open(self.model_path + '/config.json') as f:
				data = json.load(f)
		except Exception as e:
			raise InvalidModelConfiguration('config.json not found or corrupted')
		json_confidence = data['confidence']
		json_predictions = data['predictions']
		with self.detection_graph.as_default():
			# Expand dimension since the model expects image to have shape [1, None, None, 3].
			img_expanded = np.expand_dims(np_image, axis=0)
			(boxes, scores, classes, num) = self.sess.run(
				[self.d_boxes, self.d_scores, self.d_classes, self.num_d],
				feed_dict={self.image_tensor: img_expanded})
		classes_names = ([self.category_index.get(i) for i in classes[0]])
		names_start = []
		for name in classes_names:
			if name is not None:
				names_start.append(name['name'])
		height, width, depth = np_image.shape
		names = []
		confidence = []
		ids = []
		bounding_boxes = []
		for i in range(json_predictions):
			if scores[0][i] * 100 >= json_confidence:
				ymin = int(round(boxes[0][i][0] * height)) if int(round(boxes[0][i][0] * height)) > 0 else 0
				xmin = int(round(boxes[0][i][1] * width)) if int(round(boxes[0][i][1] * height)) > 0 else 0
				ymax = int(round(boxes[0][i][2] * height)) if int(round(boxes[0][i][2] * height)) > 0 else 0
				xmax = int(round(boxes[0][i][3] * width)) if int(round(boxes[0][i][3] * height)) > 0 else 0
				tmp = dict([('left', xmin), ('top', ymin), ('right', xmax), ('bottom', ymax)])
				bounding_boxes.append(tmp)
				confidence.append(float(scores[0][i] * 100))
				ids.append(int(classes[0][i]))
				names.append(names_start[i])

		responses_list = zip(names, confidence, bounding_boxes, ids)

		output = []
		for response in responses_list:
			tmp = dict([('ObjectClassName', response[0]), ('confidence', response[1]), ('coordinates', response[2]),
						('ObjectClassId', response[3])])
			output.append(tmp)
		if predict_batch:
			response = dict([('bounding-boxes', output), ('ImageName', input_data.filename)])
		else:
			response = dict([('bounding-boxes', output)])

		if not draw:
			return response
		else:
			try:
				self.draw_image(pillow_image, response)
			except ApplicationError as e:
				raise e
			except Exception as e:
				raise e

	async def run_batch(self, input_data, draw, predict_batch):
		result_list = []
		for image in input_data:
			post_process = await self.infer(image, draw, predict_batch)
			if post_process is not None:
				result_list.append(post_process)
		return result_list

	def draw_image(self, image, response):
		"""
		Draws on image and saves it.
		:param image: image of type pillow image
		:param response: inference response
		:return:
		"""
		draw = ImageDraw.Draw(image)
		for bbox in response['bounding-boxes']:
			draw.rectangle([bbox['coordinates']['left'], bbox['coordinates']['top'], bbox['coordinates']['right'],
							bbox['coordinates']['bottom']], outline="red")
			left = bbox['coordinates']['left']
			top = bbox['coordinates']['top']
			conf = "{0:.2f}".format(bbox['confidence'])
			draw.text((int(left), int(top) - 20), str(conf) + "% " + str(bbox['ObjectClassName']), 'red', self.font)
		image.save('/main/result.jpg', 'PNG')

	def free(self):
		pass

	def validate_configuration(self):
		# check if weights file exists
		if not os.path.exists(os.path.join(self.model_path, 'frozen_inference_graph.pb')):
			raise InvalidModelConfiguration('frozen_inference_graph.pb not found')
		# check if labels file exists
		if not os.path.exists(os.path.join(self.model_path, 'object-detection.pbtxt')):
			raise InvalidModelConfiguration('object-detection.pbtxt not found')
		return True

	def set_model_configuration(self, data):
		self.configuration['framework'] = data['framework']
		self.configuration['type'] = data['type']
		self.configuration['network'] = data['network']
		self.NUM_CLASSES = data['number_of_classes']

	def validate_json_configuration(self, data):
		with open(os.path.join('inference', 'ConfigurationSchema.json')) as f:
			schema = json.load(f)
		try:
			jsonschema.validate(data, schema)
		except Exception as e:
			raise InvalidModelConfiguration(e)
