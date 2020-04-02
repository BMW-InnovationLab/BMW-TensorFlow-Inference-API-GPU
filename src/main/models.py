class ApiResponse:

    def __init__(self, success=True, data=None, error=None):
        """
        Defines the response shape
        :param success: A boolean that returns if the request has succeeded or not
        :param data: The model's response
        :param error: The error in case an exception was raised
        """
        self.data = data
        self.error = error.get_message() if error is not None else ''
        self.success = success
