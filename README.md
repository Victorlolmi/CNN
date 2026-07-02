# CNN

Repositório criado para a disciplina de CNN (Redes Neurais Convolucionais), reunindo o pipeline de preparação de dados e o treinamento de dois modelos de segmentação: U-Net e YOLO.

## Estrutura do repositório

```
CNN/
├── data_preparation/   # Scripts/notebooks de preparação e pré-processamento do dataset
├── train-unet/         # Treinamento do modelo U-Net
├── train_yolo/         # Treinamento do modelo YOLO
└── .gitattributes
```
## Sobre o projeto

Este repositório contém o fluxo de trabalho utilizado na disciplina de CNN, cobrindo desde a preparação dos dados de entrada até o treinamento e comparação de duas arquiteturas de segmentação (U-Net e YOLO).

- **`data_preparation/`** — notebooks/scripts responsáveis por organizar, limpar e formatar os dados (imagens e máscaras/labels) antes do treinamento.
- **`train-unet/`** — notebooks/scripts de treinamento do modelo U-Net.
- **`train_yolo/`** — notebooks/scripts de treinamento do modelo YOLO (segmentação).

OBS:. Os dados de treinamento foram retirados por não serem de autoria própria
## Requisitos

- Python 3.x
- Jupyter Notebook
- PyTorch e/ou Ultralytics YOLO (conforme o modelo)
- Bibliotecas de manipulação de imagem (ex: `opencv-python`, `Pillow`, `numpy`)

## Licença

Não especificada.
