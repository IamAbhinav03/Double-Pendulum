import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from seed_generator import SeedGenerator

# Create the FastAPI application
app = FastAPI()

# Middleware to allow the frontend to talk to the backend from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"]  # Allow all headers
)

# Define a model for the data we expect from the user (bit_length)
class SeedRequest(BaseModel):
    bit_length: int  # This is the number of random bits the user wants

# Define the API endpoint to generate a random seed
@app.post("/generate-seed")
def generate_seed(request: SeedRequest):
    bit_length: int = request.bit_length  # Get the bit length from the request

    # Check if the bit length is valid (must be greater than 0)
    if bit_length <= 0:
        raise HTTPException(status_code=400, detail="bit_length must be greater than 0")

    # Create a seed generator with the requested bit length
    generator = SeedGenerator(bit_length)

    # Start the video capture and frame processing
    generator.start_processing('udp://127.0.0.1:5000/?overrun_nonfatal=1')

    try:
        # Try to generate the seed
        seed: str = generator.request_seed()
        print(f"Generated Seed: {seed}")

    except Exception as e:
        # If something goes wrong, return a 500 error
        raise HTTPException(status_code=500, detail=f"Error generating seed: {str(e)}")

    finally:
        # Stop the video capture and processing
        generator.stop_processing()
        print("Generator successfully stopped")

    # Return the generated seed to the user
    return {"seed": seed}

