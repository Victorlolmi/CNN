# coffee_data_loader.py
import os
import random
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

class CoffeeDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, augmentations: dict = None):
        """
        Inicializa o Dataset lendo todos os nomes dos arquivos.
        Mantemos TODAS as imagens, incluindo os backgrounds puros para evitar Falsos Positivos.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations or {}
        
        # Lê arquivos de imagem (suporta dinamicamente .png e .tif)
        self.filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.tif', '.tiff'))])
        
    def __len__(self):
        """Retorna o número total de amostras no dataset."""
        return len(self.filenames)
        
    def __getitem__(self, idx):
        """
        Carrega a imagem e a máscara, aplica escala dinâmica de profundidade de cor,
        resolve máscaras faltantes (Background) e normaliza para o padrão ImageNet.
        """
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # ─────────────────────────────────────────────────────────────────
        # 1. LEITURA SEGURA (Preserva 8-bits ou 16-bits)
        # ─────────────────────────────────────────────────────────────────
        img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
        # ─────────────────────────────────────────────────────────────────
        # 2. ESCALA DINÂMICA MÁXIMA ABSOLUTA (0.0 a 1.0)
        # ─────────────────────────────────────────────────────────────────
        # Garante que o cálculo não esmague nem extrapole os valores reais
        if img_cv.dtype == np.uint16:
            img_array = img_cv.astype(np.float32) / 65535.0
        else:
            img_array = img_cv.astype(np.float32) / 255.0
            
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # ─────────────────────────────────────────────────────────────────
        # 3. LÓGICA DE BACKGROUND INTELIGENTE
        # ─────────────────────────────────────────────────────────────────
        if os.path.exists(mask_path):
            mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_tensor = torch.from_numpy((mask_cv > 0).astype(np.float32)).unsqueeze(0)
        else:
            _, h, w = img_tensor.shape
            mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)
        
        # ─────────────────────────────────────────────────────────────────
        # 4. DATA AUGMENTATION (Espacial)
        # ─────────────────────────────────────────────────────────────────
        if self.augmentations:
            if "hflip" in self.augmentations and random.random() < self.augmentations["hflip"]:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            if "vflip" in self.augmentations and random.random() < self.augmentations["vflip"]:
                img_tensor = TF.vflip(img_tensor)
                mask_tensor = TF.vflip(mask_tensor)

            angle = random.choice(self.augmentations.get("rotate", [0]))
            scale_factor = random.uniform(*self.augmentations["scale"]) if "scale" in self.augmentations else 1.0

            if angle != 0 or scale_factor != 1.0:
                img_tensor = TF.affine(
                    img_tensor, angle=angle, translate=[0, 0], scale=scale_factor, 
                    shear=0.0, interpolation=InterpolationMode.BILINEAR
                )
                mask_tensor = TF.affine(
                    mask_tensor, angle=angle, translate=[0, 0], scale=scale_factor, 
                    shear=0.0, interpolation=InterpolationMode.NEAREST
                )

        # ─────────────────────────────────────────────────────────────────
        # 5. NORMALIZAÇÃO ESTATÍSTICA (ImageNet)
        # ─────────────────────────────────────────────────────────────────
        # Alinha as cores da imagem com a distribuição esperada pela ResNet101
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        img_tensor = TF.normalize(img_tensor, mean=mean, std=std)

        return img_tensor, mask_tensor, img_name