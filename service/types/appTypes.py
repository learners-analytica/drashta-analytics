from pydantic import BaseModel


class MLModel(BaseModel):
    model_name : str
    model_algorithim : str
    model_accuracy : float