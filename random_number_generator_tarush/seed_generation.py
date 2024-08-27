import hashlib
import logging
from config import PENDULUM_POINTS_BUFFER

def float_to_fixed_point(value, precision=1e6):
    return int(value * precision)

def interleave_bits(coords):
    interleaved = 0
    max_bits = max(len(bin(float_to_fixed_point(x)).lstrip('0b')) for coord in coords for x, y in [coord])
    
    for coord in coords:
        x, y = coord
        x_bits = float_to_fixed_point(x)
        y_bits = float_to_fixed_point(y)
        
        for i in range(max_bits):
            interleaved <<= 2
            interleaved |= ((x_bits >> i) & 1) << 1 | ((y_bits >> i) & 1)
    
    return interleaved

def hash_interleaved_bits(interleaved):
    interleaved_bytes = interleaved.to_bytes((interleaved.bit_length() + 7) // 8, 'big')
    return hashlib.sha256(interleaved_bytes).hexdigest()

def generate_seed(coords):
    try:
        if len(coords) < 5:
            raise ValueError("At least 5 unique coordinates are required to generate a seed")
        
        # Use only the first 5 unique coordinates
        coords = coords[:5]
        
        interleaved = interleave_bits(coords)
        seed_hash = hash_interleaved_bits(interleaved)
        return int(seed_hash, 16)
    except ValueError as e:
        logging.error(f"ValueError in generate_seed: {str(e)}")
        return None