import sys
from typing import List
from models import ApiResponse
from inference.errors import Error
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from deep_learning_service import DeepLearningService
from fastapi import FastAPI, Form, File, UploadFile, Header
from inference.exceptions import ModelNotFound, InvalidModelConfiguration, ApplicationError, ModelNotLoaded, \
    InferenceEngineNotFound, InvalidInputData

sys.path.append('./inference')

dl_service = DeepLearningService()
error_logging = Error()
app = FastAPI(version='1.0', title='BMW InnovationLab tensorflow gpu inference Automation',
              description="<b>API for performing tensorflow gpu inference</b></br></br>"
                          "<b>Contact the developers:</b></br>"
                          "<b>Antoine Charbel: <a href='mailto:antoine.charbel@inmind.ai'>antoine.charbel@inmind.ai</a></b></br>"
                          "<b>BMW Innovation Lab: <a href='mailto:innovation-lab@bmw.de'>innovation-lab@bmw.de</a></b>")


# app.mount("/public", StaticFiles(directory="/main/public"), name="public")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get('/load')
def load_custom():
    """
    Loads all the available models.
    :return: All the available models with their respective hashed values
    """
    try:
        return dl_service.load_all_models()
    except ApplicationError as e:
        return ApiResponse(success=False, error=e)
    except Exception:
        return ApiResponse(success=False, error='unexpected server error')


@app.post('/detect')
async def detect_custom(model: str = Form(...), image: UploadFile = File(...)):
    """
    Performs a prediction for a specified image using one of the available models.
    :param model: Model name or model hash
    :param image: Image file
    :return: Model's Bounding boxes
    """
    draw_boxes = False
    predict_batch = False
    try:
        output = await dl_service.run_model(model, image, draw_boxes, predict_batch)
        error_logging.info('request successful;' + str(output))
        return output
    except ApplicationError as e:
        error_logging.warning(model + ';' + str(e))
        return ApiResponse(success=False, error=e)
    except Exception as e:
        error_logging.error(model + ' ' + str(e))
        return ApiResponse(success=False, error='unexpected server error')


@app.post('/get_labels')
def get_labels_custom(model: str = Form(...)):
    """
    Lists the model's labels with their hashed values.
    :param model: Model name or model hash
    :return: A list of the model's labels with their hashed values
    """
    return dl_service.get_labels_custom(model)


@app.get('/models/{model_name}/load')
async def load(model_name: str, force: bool = False):
    """
    Loads a model specified as a query parameter.
    :param model_name: Model name
    :param force: Boolean for model force reload on each call
    :return: APIResponse
    """
    try:
        dl_service.load_model(model_name, force)
        return ApiResponse(success=True)
    except ApplicationError as e:
        return ApiResponse(success=False, error=e)


@app.get('/models')
async def list_models(user_agent: str = Header(None)):
    """
    Lists all available models.
    :param user_agent:
    :return: APIResponse
    """
    return ApiResponse(data={'models': dl_service.list_models()})


@app.post('/models/{model_name}/predict')
async def run_model(model_name: str, input_data: UploadFile = File(...)):
    """
    Performs a prediction by giving both model name and image file.
    :param model_name: Model name
    :param input_data: An image file
    :return: APIResponse containing the prediction's bounding boxes
    """
    try:
        output = await dl_service.run_model(model_name, input_data, draw=False, predict_batch=False)
        error_logging.info('request successful;' + str(output))
        return ApiResponse(data=output)
    except ApplicationError as e:
        error_logging.warning(model_name + ';' + str(e))
        return ApiResponse(success=False, error=e)
    except Exception as e:
        error_logging.error(model_name + ' ' + str(e))
        return ApiResponse(success=False, error='unexpected server error')


@app.post('/models/{model_name}/predict_batch', include_in_schema=False)
async def run_model_batch(model_name: str, input_data: List[UploadFile] = File(...)):
    """
    Performs a prediction by giving both model name and image file(s).
    :param model_name: Model name
    :param input_data: A batch of image files or a single image file
    :return: APIResponse containing prediction(s) bounding boxes
    """
    try:
        output = await dl_service.run_model(model_name, input_data, draw=False, predict_batch=True)
        error_logging.info('request successful;' + str(output))
        return ApiResponse(data=output)
    except ApplicationError as e:
        error_logging.warning(model_name + ';' + str(e))
        return ApiResponse(success=False, error=e)
    except Exception as e:
        print(e)
        error_logging.error(model_name + ' ' + str(e))
        return ApiResponse(success=False, error='unexpected server error')


@app.post('/models/{model_name}/predict_image')
async def predict_image(model_name: str, input_data: UploadFile = File(...)):
    """
    Draws bounding box(es) on image and returns it.
    :param model_name: Model name
    :param input_data: Image file
    :return: Image file
    """
    try:
        output = await dl_service.run_model(model_name, input_data, draw=True, predict_batch=False)
        error_logging.info('request successful;' + str(output))
        return FileResponse("/main/result.jpg", media_type="image/jpg")
    except ApplicationError as e:
        error_logging.warning(model_name + ';' + str(e))
        return ApiResponse(success=False, error=e)
    except Exception as e:
        error_logging.error(model_name + ' ' + str(e))
        return ApiResponse(success=False, error='unexpected server error')


@app.get('/models/{model_name}/labels')
async def list_model_labels(model_name: str):
    """
    Lists all the model's labels.
    :param model_name: Model name
    :return: List of model's labels
    """
    labels = dl_service.get_labels(model_name)
    return ApiResponse(data=labels)


@app.get('/models/{model_name}/config')
async def list_model_config(model_name: str):
    """
    Lists all the model's configuration.
    :param model_name: Model name
    :return: List of model's configuration
    """
    config = dl_service.get_config(model_name)
    return ApiResponse(data=config)


@app.get("/type")
async def get_inference_type():
    return ApiResponse(data="object_detection")
