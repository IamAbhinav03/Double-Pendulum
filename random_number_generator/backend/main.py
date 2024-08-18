import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from seed_generator import SeedGenerator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class SeedRequest(BaseModel):
    bit_length: int

@app.post("/generate-seed")
async def generate_seed(request: SeedRequest):
    bit_length = request.bit_length

    if bit_length <= 0:
        raise HTTPException(status_code=400, detail="bit_length must be greater than 0")
    
    generator = SeedGenerator(bit_length)
    generator.start_processing('sample.mp4')

    try:
        seed = generator.request_seed()
        print(f"Generated Seed: {seed}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating seed: {str(e)}")
        
    finally:
        generator.stop_processing()
        print("Generator succesfully stopped")

    return {"seed": seed}





# def main():
#     bit_length = 256  # For example, 256 bits
#     generator = SeedGenerator(bit_length)
    
#     # Replace '0' with your actual video source
#     generator.start_processing('sample.mp4')  
    
#     print("Requesting seed")
#     try:
#         seed = generator.request_seed()
#         print(f"Generated Seed: {seed}")

#     except Exception as e:
#         print(e)
#         print("Exiting...")
        
#     finally:
#         generator.stop_processing()
#         print("Generator succesfully stopped")


# if __name__ == "__main__":
#     main()