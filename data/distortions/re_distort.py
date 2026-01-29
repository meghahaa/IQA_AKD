import cv2
import numpy as np
from PIL import Image

class ReDistortionPipeline:
    """
    Implementation of the Image Re-Distortion Strategy from the AKD-IQA paper.
    Uses a fixed sequence of maximum-level compound distortions to create a 
    stable low-quality reference.
    """
    def __init__(self, seed=42, verbose=False):
        self.verbose = verbose
        np.random.seed(seed)

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or np.array): input distorted image x
        Returns:
            np.array: re-distorted image xd
        """
        # Convert PIL to numpy for OpenCV processing
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        xd = img.copy()

        # The paper specifies a compound strategy: xd = D0(D1(...(DM(x))))
        # We apply ALL major distortions at their "Maximum Level" sequentially
        # to ensure the 'indistinguishable' low-quality requirement.
        
        if self.verbose: print("[AKD] Generating stable re-distorted reference...")

        # 1. Spatial Degradation (Resizing/Pixelation) - Destroys structural edges
        xd = self.apply_max_pixelate(xd, block_size=16)
        
        # 2. Additive Noise (Gaussian & Impulse) - Destroys local texture
        xd = self.apply_max_gaussian_noise(xd, std=50)
        xd = self.apply_max_impulse_noise(xd, prob=0.05)
        
        # 3. Blur (Gaussian & Motion) - Destroys high-frequency details
        xd = self.apply_max_gaussian_blur(xd, sigma=7.0)
        xd = self.apply_max_motion_blur(xd, kernel_size=21)
        
        # 4. Compression Artifacts (JPEG) - Final quantization stage
        xd = self.apply_max_jpeg_compression(xd, quality=5)

        return xd

    # ============================================================
    # Maximum Intensity Distortion Definitions (The "w" parameters)
    # ============================================================

    def apply_max_gaussian_blur(self, img, sigma=7.0):
        # Formula 8 in paper: uses 2D normal distribution
        ksize = int(6 * sigma + 1) | 1 # Ensure odd kernel
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def apply_max_motion_blur(self, img, kernel_size=21):
        kernel = np.zeros((kernel_size, kernel_size))
        # Strong horizontal motion blur
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(img, -1, kernel)

    def apply_max_jpeg_compression(self, img, quality=5):
        # Extreme quantization artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(encimg, 1)

    def apply_max_gaussian_noise(self, img, std=50):
        # High-intensity sensor noise simulation
        noise = np.random.normal(0, std, img.shape)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_max_impulse_noise(self, img, prob=0.05):
        # Salt and pepper noise at a high density
        noisy = img.copy()
        rnd = np.random.rand(*img.shape[:2])
        noisy[rnd < (prob/2)] = 0    # Salt
        noisy[rnd > 1 - (prob/2)] = 255 # Pepper
        return noisy

    def apply_max_pixelate(self, img, block_size=16):
        # Severe loss of resolution
        h, w = img.shape[:2]
        temp = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


# # Example usage:
# if __name__ == "__main__":
#     # Load an example image
#     input_image = Image.open("data/images/I01_01_02.png")

#     # Create the re-distortion pipeline
#     re_distorter = ReDistortionPipeline(verbose=True)

#     # Apply re-distortion
#     re_distorted_image = re_distorter(input_image)

#     # Save or display the result
#     cv2.imwrite("data/images/re_I01_01_02.png", re_distorted_image)