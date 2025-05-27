from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import torch
from vllm import LLM, SamplingParams
import sys
import os
from PIL import Image
import base64
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add FoodLMM to path
FOODLMM_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(FOODLMM_ROOT, "model")
sys.path.append(FOODLMM_ROOT)

try:
    from model.utils.process_image import process_single_image
except ImportError as e:
    logger.error(f"Failed to import FoodLMM utilities: {e}")
    sys.exit(1)

app = FastAPI(title="FoodLMM Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = os.path.join(MODEL_ROOT, "checkpoints/foodlmm-7b")

class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: str = "Please identify the food items in this image and provide their nutritional information."

class FoodLMMServer:
    def __init__(self):
        try:
            logger.info(f"Initializing LLM with model path: {MODEL_PATH}")
            self.llm = LLM(
                model=MODEL_PATH,
                tensor_parallel_size=1,
                trust_remote_code=True,
                dtype="float16",
            )
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=512,
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        target_size = (224, 224)
        if image.size != target_size:
            image = image.resize(target_size)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    async def process_request(self, image: Image.Image, prompt: str) -> Dict:
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            logger.info("Image preprocessed successfully")
            
            # Process image using FoodLMM utilities
            image_features = process_single_image(processed_image)
            logger.info("Image features extracted successfully")
            
            # Generate response using vllm
            outputs = self.llm.generate(
                prompt,
                self.sampling_params,
                additional_inputs={"image_features": image_features}
            )
            
            response = outputs[0].outputs[0].text
            logger.info("Generated response from LLM")
            
            # Parse response
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            return {
                "status": "success",
                "foods": lines,
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Initialize server
try:
    server = FoodLMMServer()
except Exception as e:
    logger.error(f"Failed to initialize server: {e}")
    sys.exit(1)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FoodLMM"}

@app.post("/recognize")
async def recognize_food(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        logger.info("Received and decoded image successfully")
        
        # Process request
        result = await server.process_request(image, request.prompt)
        
        if result["status"] == "error":
            logger.error(f"Error processing request: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        logger.error(f"Error in recognize_food endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FoodLMM server...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 