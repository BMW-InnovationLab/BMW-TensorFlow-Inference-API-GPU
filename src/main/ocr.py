import pytesseract
import unicodedata
import re
import numpy as np


# Define class variables

bounding_box_order = ["left", "top", "right", "bottom"]

# This method will take the model bounding box predictions and return the extracted text inside each box
def one_shot_ocr_service(image, output):
    # iterate over detections
    response = []
    detections = output['bounding-boxes']

    for i in range(0, len(detections)):

        # crop image for every detection:
        coordinates = (detections[i]["coordinates"])
        cropped = image.crop((float(coordinates["left"]), float(
            coordinates["top"]), float(coordinates["right"]), float(coordinates["bottom"])))

        # extract text with positive confidence from cropped image
        df = pytesseract.image_to_data(cropped, output_type='data.frame')
        valid_df = df[df["conf"] > 0]
        extracted_text = " ".join(valid_df["text"].values)

        # process text
        extracted_text = str(unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode()).strip().replace("\n", " ").replace(
            "...", ".").replace("..", ".").replace('”', ' ').replace('“', ' ').replace("'", ' ').replace('\"', '').replace("alt/1m", "").strip()
        extracted_text = re.sub(
            '[^A-Za-z0-9.!?,;%:=()\[\]$€&/\- ]+', '', extracted_text)
        extracted_text = " ".join(extracted_text.split())

        # wrap each prediction inside a dictionary
        if len(extracted_text) is not 0:
            prediction = dict()
            prediction["text"] = extracted_text
            bounding_box = [coordinates[el] for el in bounding_box_order]
            prediction["box"] = bounding_box
            prediction["score"] = valid_df["conf"].mean()/100.0

            response.append(prediction)

    return response

# This method will take an image and return the extracted text from the image
def ocr_service(image):

    # Get data including boxes, confidences, line and page numbers
    df = pytesseract.image_to_data(image, output_type='data.frame')
    valid_df = df[df["conf"] > 0]

    # process text
    extracted_text = " ".join(valid_df["text"].values)
    extracted_text = str(unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode()).strip().replace("\n", " ").replace(
        "...", ".").replace("..", ".").replace('”', ' ').replace('“', ' ').replace("'", ' ').replace('\"', '').replace("alt/1m", "").strip()
    extracted_text = re.sub(
        '[^A-Za-z0-9.!?,;%:=()\[\]$€&/\- ]+', '', extracted_text)
    extracted_text = " ".join(extracted_text.split())

    # calculate the bounding box data based on pytesseract results
    coordinates = {}
    index = valid_df.index.values
    coordinates["left"] = valid_df.loc[index[0], "left"]
    coordinates["top"] = valid_df.loc[index[0], "top"]
    coordinates["bottom"] = valid_df.loc[index[-1],
                                         "top"] + valid_df.loc[index[-1], "height"]
    coordinates["right"] = valid_df.loc[index[-1],
                                        "left"] + valid_df.loc[index[-1], "width"]
    bounding_box = [coordinates[el].item() for el in bounding_box_order]

    # wrap each prediction inside a dictionary
    response = {}
    response["text"] = extracted_text
    response["box"] = bounding_box
    response["score"] = valid_df["conf"].mean()/100.0

    return [response]
