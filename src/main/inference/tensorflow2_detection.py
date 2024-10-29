import os
import jsonschema
import asyncio
import json
import numpy as np
from six import BytesIO
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from inference.base_inference_engine import AbstractInferenceEngine
from inference.exceptions import InvalidModelConfiguration, InvalidInputData, ApplicationError
from object_detection.utils import label_map_util



# noinspection PyMethodMayBeStatic
class InferenceEngine(AbstractInferenceEngine):

    def __init__(self, model_path):
        self.label_path = ""
        self.NUM_CLASSES = None
        self.label_map = None
        self.labels = None
        self.label_map_dict = None
        self.categories = None
        self.category_index = None
        self.detect_fn = None
        self.font = ImageFont.truetype("./fonts/DejaVuSans.ttf", 20)
        super().__init__(model_path)

    def _load_image_into_numpy_array(self, path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
          path: the file path to the image

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def _get_keypoint_tuples(self, eval_config):
        """Return a tuple list of keypoint edges from the eval config.

        Args:
          eval_config: an eval config containing the keypoint edges

        Returns:
          a list of edge tuples, each in the format (start, end)
        """
        tuple_list = []
        kp_list = eval_config.keypoint_edge
        for edge in kp_list:
            tuple_list.append((edge.start, edge.end))
        return tuple_list

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
                                                                         max_num_classes=label_map_util.get_max_label_map_index(
                                                                             self.label_map),
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.label_map_dict = label_map_util.get_label_map_dict(self.label_map, use_display_name=True)
        self.labels = [label for label in self.label_map_dict]
        # allow memory growth
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU')]
        self.detect_fn = tf.saved_model.load(self.model_path)



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

        input_tensor = tf.convert_to_tensor(np_image)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)

        height, width, depth = np_image.shape

        names = []
        confidence = []
        ids = []
        bounding_boxes = []
        names_start = []
        scores = detections["detection_scores"][0].numpy()
        boxes = detections["detection_boxes"][0].numpy()
        classes = (detections['detection_classes'][0].numpy()).astype(int)
        classes_names = ([self.category_index.get(i) for i in classes])
        for name in classes_names:
            if name is not None:
                names_start.append(name['name'])

        for i in range(json_predictions):
            if scores[i] * 100 >= json_confidence:
                ymin = int(round(boxes[i][0] * height)) if int(round(boxes[i][0] * height)) > 0 else 0
                xmin = int(round(boxes[i][1] * width)) if int(round(boxes[i][1] * height)) > 0 else 0
                ymax = int(round(boxes[i][2] * height)) if int(round(boxes[i][2] * height)) > 0 else 0
                xmax = int(round(boxes[i][3] * width)) if int(round(boxes[i][3] * height)) > 0 else 0
                tmp = dict([('left', xmin), ('top', ymin), ('right', xmax), ('bottom', ymax)])
                bounding_boxes.append(tmp)
                confidence.append(float(scores[i] * 100))
                ids.append(int(classes[i]))
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
        image.save('./result.jpg', 'PNG')

    def free(self):
        pass

    def validate_variables(self):
        valid: bool = False

        index_file: str = None
        meta_file: str = None

        for var in os.listdir(os.path.join(self.model_path, 'variables')):

            if var.startswith("variables") and var.endswith(".index"):
                index_file = var
            elif var.startswith("variables") and var.endswith(".data-00000-of-00001"):
                meta_file = var

        if meta_file is not None and index_file is not None:
            valid = True

        return valid

    def validate_configuration(self):
        # check if variables folder exist
        if not os.path.isdir(os.path.join(self.model_path, 'variables')):
            raise InvalidModelConfiguration('variables folder  not found')

        # check if variables are valid
        if not self.validate_variables():
            raise InvalidModelConfiguration('variables folder structure not valid')

        # check if weights file exists
        if not os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
            raise InvalidModelConfiguration('saved_model.pb not found')
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
