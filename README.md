# signature_verification
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def load_signature_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image

def verify_signature(original_signature, test_signature, threshold=0.75):
    # Resize the images to the same dimensions for comparison
    original_signature = cv2.resize(original_signature, (200, 100))
    test_signature = cv2.resize(test_signature, (200, 100))
    
    # Calculate the structural similarity index
    similarity_index = ssim(original_signature, test_signature)
    
    if similarity_index > threshold:
        return True, similarity_index
    else:
        return False, similarity_index

if __name__ == "__main__":
    # Load the original signature image
    original_signature = load_signature_image("original_signature.png")
    
    # Load the test signature image (to be verified)
    test_signature = load_signature_image("test_signature.png")
    
    # Verify the signature
    result, similarity_index = verify_signature(original_signature, test_signature)
    
    if result:
        print("Signature verified! Similarity index:", similarity_index)
    else:
        print("Signature not verified. Similarity index:", similarity_index)
