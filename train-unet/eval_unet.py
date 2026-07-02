# eval_unet.py

import os
import cv2
import json
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm


from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


# ── IMPORTAÇÃO DO MONAI ── #
from monai.metrics import MeanIoU, ConfusionMatrixMetric

# ── IMPORTAÇÃO DA NOSSA CLASSE DE DADOS ── #
from coffee_data_loader import CoffeeDataset

## =============================================================================
# FUNÇÃO PARA RECONSTRUIR O MODELO AUTOMATICAMENTE
# =============================================================================
def load_trained_model(run_dir, device, cmd_attention=None):
    if not os.path.exists(run_dir) and os.path.exists(os.path.join("runs", run_dir)):
        run_dir = os.path.join("runs", run_dir)

    info_path = os.path.join(run_dir, "dataset_info.json")
    weights_path = os.path.join(run_dir, "best_unet_model.pth")
    
    if not os.path.exists(info_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Arquivos da run não encontrados em {run_dir}")
        
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    print(f"🔄 Reconstruindo modelo: {info['MODEL_TYPE'].upper()} com {info['ENCODER'].upper()}...")
    
    # 1. Pega do terminal se existir, senão pega do JSON, senão assume None do Python
    attention_type = cmd_attention if cmd_attention is not None else info.get("ATTENTION", None)

    # 2. SE FOR STRING, limpa espaços e trata variações de "none" ou "None"
    if isinstance(attention_type, str):
        attention_type = attention_type.strip()
        if attention_type.lower() in ["none", ""]:
            attention_type = None

    print(f"🧠 Mecanismo de atenção configurado: {str(attention_type).upper()}")

    arch = smp.UnetPlusPlus if info['MODEL_TYPE'] == "unet++" else smp.Unet
    
    model = arch(
        encoder_name=info['ENCODER'], 
        encoder_weights=None, 
        in_channels=info['IN_CHANNELS'], 
        classes=1, 
        activation=None,
        decoder_attention_type=attention_type  # 👈 Agora passa None (objeto) com segurança
    )

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model

# =============================================================================
# SCRIPT PRINCIPAL DE INFERÊNCIA E AVALIAÇÃO
# =============================================================================
def evaluate_and_infer(run_dir, batch_size=4, attention=None): # 👈 ADICIONADO: parâmetro attention
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    info_path = os.path.join(run_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Arquivo dataset_info.json não encontrado em {run_dir}")
        
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    base_data_dir = info["BASE_DATASET_DIR"]
    test_img_dir  = os.path.join(base_data_dir, "images", "test")
    test_mask_dir = os.path.join(base_data_dir, "labels", "test")
    
    out_preds_dir    = os.path.join(run_dir, "preds")
    out_images_dir   = os.path.join(out_preds_dir, "images")
    out_overlays_dir = os.path.join(out_preds_dir, "overlays")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_overlays_dir, exist_ok=True)

    # 👈 CORRIGIDO: Passando o parâmetro de atenção recebido para a reconstrução do modelo
    model = load_trained_model(run_dir, device, cmd_attention=attention) 
    
    test_dataset = CoffeeDataset(test_img_dir, test_mask_dir, augmentations=None)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\n📊 [EVAL] Lendo dados de teste da pasta: {base_data_dir}")
    print(f"📊 Iniciando inferência global ({len(test_dataset)} imagens)...")
    
    monai_iou  = MeanIoU(include_background=True, reduction="mean_batch")
    monai_prec = ConfusionMatrixMetric(include_background=True, metric_name="precision", reduction="mean_batch")
    monai_rec  = ConfusionMatrixMetric(include_background=True, metric_name="recall",    reduction="mean_batch")
    monai_f1   = ConfusionMatrixMetric(include_background=True, metric_name="f1 score",  reduction="mean_batch")
    monai_acc  = ConfusionMatrixMetric(include_background=True, metric_name="accuracy",  reduction="mean_batch")

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Inference", colour='cyan')
        for image_tensor, mask_tensor, filenames in test_loop:
            image_tensor = image_tensor.to(device)
            mask_tensor  = mask_tensor.to(device)
            with autocast('cuda'):
                mask_predictions = model(image_tensor)
                if mask_predictions.shape[-2:] != mask_tensor.shape[-2:]:
                    mask_predictions = F.interpolate(mask_predictions, size=mask_tensor.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(mask_predictions)
            conf_threshold = 0.5
            preds_bin   = (probs > conf_threshold).float()
            targets_bin = (mask_tensor > conf_threshold).float()
            preds_onehot = torch.cat([1 - preds_bin, preds_bin], dim=1)
            targets_onehot = torch.cat([1 - targets_bin, targets_bin], dim=1)
            monai_iou(y_pred=preds_onehot, y=targets_onehot)
            monai_prec(y_pred=preds_onehot, y=targets_onehot)
            monai_rec(y_pred=preds_onehot, y=targets_onehot)
            monai_f1(y_pred=preds_onehot, y=targets_onehot)
            monai_acc(y_pred=preds_onehot, y=targets_onehot)
            for i, fname in enumerate(filenames):
                orig_path = os.path.join(test_img_dir, fname)
                dest_path = os.path.join(out_images_dir, fname)
                shutil.copy2(orig_path, dest_path)
                orig_img_bgr = cv2.imread(orig_path)
                pred_np = preds_bin[i].cpu().squeeze().numpy()
                mask_bool = pred_np.astype(bool)
                overlay_bgr = orig_img_bgr.copy()
                overlay_bgr[mask_bool] = np.clip(overlay_bgr[mask_bool] * 0.5 + np.array([0, 0, 255]) * 0.5, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(out_overlays_dir, fname), overlay_bgr)

    # Cálculo do IoU por classe
    iou_per_class = monai_iou.aggregate()
    iou_bg        = iou_per_class[0].item()
    iou_coffee    = iou_per_class[1].item()
    miou          = iou_per_class.mean().item()
    
    # Cálculo do F1/Dice
    f1_tensor      = monai_f1.aggregate()[0]
    mdice          = f1_tensor.mean().item()
    
    # 🌟 NOVO: Cálculo de Precision Geral e por Classe (Fundo=0, Café=1)
    prec_tensor      = monai_prec.aggregate()[0]
    mprecision       = prec_tensor.mean().item()
    precision_bg     = prec_tensor[0].item()
    precision_coffee = prec_tensor[1].item()
    
    # 🌟 NOVO: Cálculo de Recall Geral e por Classe (Fundo=0, Café=1)
    rec_tensor       = monai_rec.aggregate()[0]
    mrecall          = rec_tensor.mean().item()
    recall_bg        = rec_tensor[0].item()
    recall_coffee    = rec_tensor[1].item()
    
    # Cálculo de Pixel Accuracy
    acc_tensor     = monai_acc.aggregate()[0]
    pixel_acc      = acc_tensor.mean().item()
    
    # 👈 ADICIONADO: Inclusão das métricas separadas por classe no dicionário do JSON
    metrics_report = {
        "Test_Samples": len(test_dataset), 
        "mIoU": float(miou), 
        "IoU_Background": float(iou_bg),
        "IoU_Coffee": float(iou_coffee), 
        "mDice_Global": float(mdice), 
        "mPrecision_Global": float(mprecision),
        "Precision_Background": float(precision_bg),
        "Precision_Coffee": float(precision_coffee),
        "mRecall_Global": float(mrecall), 
        "Recall_Background": float(recall_bg),
        "Recall_Coffee": float(recall_coffee),
        "PixelAcc": float(pixel_acc)
    }
    
    with open(os.path.join(run_dir, "test_eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
        
    print("\n" + "="*50)
    print(" 🚀 RELATÓRIO DE AVALIAÇÃO CONSOLIDADO (UNET) 🚀")
    print("="*50)
    print(f" 🏆 mIoU Global          : {miou:.4f}\n 🟤 IoU Fundo (Classe 0)  : {iou_bg:.4f}\n 🟢 IoU Café (Classe 1)   : {iou_coffee:.4f}\n" + "-"*50)
    print(f" 🎯 Precision Fundo      : {precision_bg:.4f}\n 🎯 Precision Café       : {precision_coffee:.4f}\n 🔍 Recall Fundo         : {recall_bg:.4f}\n 🔍 Recall Café          : {recall_coffee:.4f}\n" + "-"*50)
    print(f" 🌟 mDice Global (Macro)  : {mdice:.4f}\n 🎯 mPrecision (2 Classes): {mprecision:.4f}\n 🔍 mRecall (2 Classes)   : {mrecall:.4f}\n Pixel Accuracy: {pixel_acc:.4f}\n" + "="*50)


# =============================================================================
# BLOCO MAIN EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia um modelo salvo.")
    parser.add_argument("--run_dir", type=str, required=True, help="Caminho para a pasta da run")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamanho do batch")
    parser.add_argument("--attention", type=str, default=None, help="Tipo de atenção caso não esteja no JSON")
    
    args = parser.parse_args()
    
    evaluate_and_infer(args.run_dir, args.batch_size, args.attention)