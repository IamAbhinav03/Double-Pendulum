import random
from sympy import isprime, nextprime, mod_inverse
from config import RSA_KEY_SIZE, RSA_PUBLIC_EXPONENT, RANDOM_NUMBER_MODULUS
import logging

logger = logging.getLogger(__name__)

def generate_prime(bits=RSA_KEY_SIZE // 2):
    """Generate a prime number of specified bit length."""
    while True:
        prime_candidate = random.getrandbits(bits)
        prime_candidate |= (1 << (bits - 1)) | 1  # Ensure bit length and odd
        if isprime(prime_candidate):
            return prime_candidate

def generate_rsa_key_pair():
    # Step 1: Find primes p and q
    p = generate_prime()
    q = generate_prime()
    
    # Step 2: Calculate N
    N = p * q

    # Ensure e is coprime to (p-1)(q-1)
    phi = (p - 1) * (q - 1)
    e = RSA_PUBLIC_EXPONENT
    while True:
        try:
            mod_inverse(e, phi)
            break
        except ValueError:
            e = nextprime(e)
    
    return p, q, N, e

class RSARabinGenerator:
    def __init__(self, seed):
        self.p, self.q, self.N, self.e = generate_rsa_key_pair()
        self.x_i = seed % self.N
        logger.debug(f"RSA parameters: p={self.p}, q={self.q}, N={self.N}, e={self.e}")
        logger.debug(f"Initial seed: {seed}, Initial x: {self.x_i}")

    def __next__(self):
        self.x_i = pow(self.x_i, self.e, self.N)
        result = self.x_i % RANDOM_NUMBER_MODULUS
        logger.debug(f"Input: {self.x_i}, Output: {result}")
        return result

    def __iter__(self):
        return self

# This function will be called from main.py
def get_rsa_rabin_generator(seed):
    return RSARabinGenerator(seed)