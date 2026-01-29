import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from data.distortions.re_distort import ReDistortionPipeline


class IQADataset(Dataset):
    """
    Unified Dataset Loader for:
    - Teacher training (FR-IQA)
    - Student training (NR-IQA with re-distortion)
    - Testing
    """

    def __init__(
        self,
        csv_file,
        root_dir,
        mode="student",
        patch_size=224,
        num_patches=10,
        training=True,
        seed=42,
        verbose=False,
    ):
        """
        Args:
            csv_file (str): path to metadata CSV
            root_dir (str): dataset root directory
            mode (str): ['teacher', 'student', 'test']
            patch_size (int): size of cropped patches
            num_patches (int): number of patches per image
            training (bool): training or evaluation
            seed (int): random seed
            verbose (bool): print debug info
        """
        assert mode in ["teacher", "student", "test"]

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.training = training
        self.verbose = verbose

        random.seed(seed)

        # Redistortion pipeline (only used for student mode)
        self.redistorter = ReDistortionPipeline(
            seed=seed,
            verbose=verbose
        )

        # Standard image transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        if self.verbose:
            print(f"[IQADataset] Loaded {len(self.data)} samples")
            print(f"[IQADataset] Mode: {self.mode}")

    def __len__(self):
        return len(self.data)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def _sample_patches(self, img):
        """
        Randomly sample patches from an image
        """
        w, h = img.size
        patches = []

        for _ in range(self.num_patches):
            if self.training:
                left = random.randint(0, w - self.patch_size)
                top = random.randint(0, h - self.patch_size)
            else:
                left = (w - self.patch_size) // 2
                top = (h - self.patch_size) // 2

            patch = img.crop(
                (left, top, left + self.patch_size, top + self.patch_size)
            )
            patches.append(self.transform(patch))

        return torch.stack(patches)  # [N, 3, H, W]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        mos = torch.tensor(row["mos"], dtype=torch.float32)

        dist_path = os.path.join(self.root_dir, row["dist_path"])
        dist_img = self._load_image(dist_path)

        dist_patches = self._sample_patches(dist_img)

        if self.mode == "test":
            return {
                "dist": dist_patches,
                "mos": mos,
            }

        if self.mode == "teacher":
            ref_path = os.path.join(self.root_dir, row["ref_path"])
            ref_img = self._load_image(ref_path)
            ref_patches = self._sample_patches(ref_img)

            return {
                "ref": ref_patches,
                "dist": dist_patches,
                "mos": mos,
            }

        if self.mode == "student":
            # Generate re-distorted image
            redist_img = self.redistorter(dist_img)
            redist_patches = self._sample_patches(redist_img)

            return {
                "dist": dist_patches,
                "redist": redist_patches,
                "mos": mos,
            }
