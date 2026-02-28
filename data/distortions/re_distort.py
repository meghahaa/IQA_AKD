import cv2
import numpy as np
from PIL import Image

class ReDistortionPipeline:
    """
    Image Re-Distortion Strategy from AKD-IQA paper.

    Applies a fixed compound distortion sequence at maximum severity to
    produce a stable low-quality reference image. The resulting re-distorted
    image is guaranteed to be of sufficiently low quality that differences
    between any two re-distorted images are indistinguishable by the human
    visual system — ensuring a stable contrast reference for the student model.

    All cv2 operations are performed in BGR. Input PIL images are converted
    to BGR on entry and back to PIL RGB on exit.
    """

    def __init__(self, seed=42, verbose=False):
        self.verbose = verbose
        np.random.seed(seed)

    def __call__(self, img):
        """
        Args:
            img: PIL.Image (RGB) or np.ndarray (BGR)

        Returns:
            PIL.Image (RGB): re-distorted image
        """
        # Normalize input to BGR numpy array
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        xd = img.copy()

        if self.verbose:
            print("[ReDistort] Applying compound distortion pipeline...")

        # Step 1 — Spatial degradation (pixelation)
        xd = self.apply_max_pixelate(xd, block_size=16)

        # Step 2 — Blur (Gaussian then motion)
        xd = self.apply_max_gaussian_blur(xd, sigma=7.0)
        xd = self.apply_max_motion_blur(xd, kernel_size=21)

        # Step 3 — Noise (Gaussian then impulse)
        xd = self.apply_max_gaussian_noise(xd, std=50)
        xd = self.apply_max_impulse_noise(xd, prob=0.05)

        # Step 4 — JPEG compression
        xd = self.apply_max_jpeg_compression(xd, quality=5)

        # Return as PIL RGB for compatibility with _sample_patches
        return Image.fromarray(cv2.cvtColor(xd, cv2.COLOR_BGR2RGB))

    # =========================================================
    # Distortion Implementations
    # =========================================================

    def apply_max_pixelate(self, img, block_size=16):
        """Severe resolution loss via downscale + upscale."""
        h, w = img.shape[:2]
        small = cv2.resize(
            img,
            (max(1, w // block_size), max(1, h // block_size)),
            interpolation=cv2.INTER_LINEAR
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def apply_max_gaussian_blur(self, img, sigma=7.0):
        """Gaussian blur per paper Eq. 8: G(x,y) = 1/(2πσ²) exp(-(x²+y²)/2σ²)"""
        ksize = int(6 * sigma + 1) | 1  # nearest odd number
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def apply_max_motion_blur(self, img, kernel_size=21):
        """Horizontal motion blur kernel."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(img, -1, kernel)

    def apply_max_gaussian_noise(self, img, std=50):
        """High-intensity additive Gaussian noise."""
        noise = np.random.normal(0, std, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def apply_max_impulse_noise(self, img, prob=0.05):
        """Salt-and-pepper impulse noise."""
        noisy = img.copy()
        rnd   = np.random.rand(*img.shape[:2])
        noisy[rnd < prob / 2]       = 0    # pepper
        noisy[rnd > 1 - prob / 2]   = 255  # salt
        return noisy

    def apply_max_jpeg_compression(self, img, quality=5):
        """Extreme JPEG quantization artifacts. img must be BGR."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(encimg, 1)