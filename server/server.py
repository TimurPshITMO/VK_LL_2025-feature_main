from fastapi import FastAPI, HTTPException, Request, Response
import pandas as pd
from server.dto import Request, Response
import logging
from contextlib import asynccontextmanager
import joblib
from features_extractor import FeatureExtractor




# Настройка логирования
logging.basicConfig(
    filename="server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the server...")
    try:
        app.state.model = joblib.load("resources/model.pkl")
        logger.info("Model loaded")
    except Exception as e:
        logger.error(f"Failed to load the model: {e}", exc_info=True)
        app.state.model = None
    try:
        app.state.feature_extractor = FeatureExtractor(history_path="resources/history.tsv", users_path="resources/users.tsv")
        logger.info("Feature extractor loaded")
    except Exception as e:
        logger.error(f"Failed to load the feature extractor: {e}", exc_info=True)
        app.state.feature_extractor = None
    yield
    logger.info("Shutting down the server...")

app = FastAPI(lifespan=lifespan)



@app.post("/predict", response_model=Response)
async def predict(data: Request):
    model = app.state.model
    if model is None:
        logger.error("Model is not loaded")
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    try:
        logger.info(f"Received data: {data}")

        data_df = pd.DataFrame([data.model_dump()])

        features = app.state.feature_extractor.get_all_features(data_df)
        prediction = model.predict(features)
        logger.info(f"Prediction: {prediction}")

        response = Response(
            at_least_one=prediction[0][0],
            at_least_two=prediction[0][1],
            at_least_three=prediction[0][2]
        )
        logger.info(f"Sending response: {response}")

        return response
    
    except Exception as e:
        logger.error(f"Internal server error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
