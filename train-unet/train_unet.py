import os
import sys
import json
import time
import datetime
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast  # Mantido apenas o autocast, GradScaler removido para bf16
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# ── IMPORTAÇÃO DA CLASSE DE DADOS ── #
from coffee_data_loader import CoffeeDataset

# ── IMPORTAÇÃO DE LOSSES E MÉTRICAS ── #
from monai.losses import DiceLoss as MonaiDiceLoss
from monai.losses import TverskyLoss as MonaiTverskyLoss
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric

# ─────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO U-NET
# ─────────────────────────────────────────────────────────────────────────────
def validate_unet(
    model, val_loader, device,
    loss_type, bce_fn, dice_fn, tversky_fn,
    writer=None, vis_filenames=None, epoch=None,
):
    model.eval()

    val_loss_total   = 0.0
    val_loss_bce      = 0.0
    val_loss_dice     = 0.0
    val_loss_tversky = 0.0
    num_val_batches  = 0

    monai_dice = DiceMetric(include_background=True, reduction="mean")
    monai_iou  = MeanIoU(include_background=True, reduction="mean")
    monai_prec = ConfusionMatrixMetric(include_background=True, metric_name="precision", reduction="mean")
    monai_rec  = ConfusionMatrixMetric(include_background=True, metric_name="recall", reduction="mean")
    monai_f1   = ConfusionMatrixMetric(include_background=True, metric_name="f1 score", reduction="mean")
    monai_acc  = ConfusionMatrixMetric(include_background=True, metric_name="accuracy", reduction="mean")

    monai_dice.reset(); monai_iou.reset(); monai_prec.reset(); monai_rec.reset(); monai_f1.reset(); monai_acc.reset()

    vis_filenames_set = set(vis_filenames) if vis_filenames else set()
    vis_collected = {}  

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Validation", leave=False, colour='green')
        
        for image_tensor, mask_tensor, batch_filenames in val_loop:
            image_tensor = image_tensor.to(device)
            mask_tensor  = mask_tensor.to(device)

            # Ativação do bf16 na validação
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                mask_predictions = model(image_tensor)
                if mask_predictions.shape[-2:] != mask_tensor.shape[-2:]:
                    mask_predictions = F.interpolate(
                        mask_predictions, size=mask_tensor.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                l_bce     = bce_fn(mask_predictions, mask_tensor)
                l_dice    = dice_fn(mask_predictions, mask_tensor)
                l_tversky = tversky_fn(mask_predictions, mask_tensor)

            active_total = l_bce + (l_dice if loss_type == "bce_dice" else l_tversky)

            val_loss_total   += active_total.item()
            val_loss_bce      += l_bce.item()
            val_loss_dice     += l_dice.item()
            val_loss_tversky += l_tversky.item()
            num_val_batches  += 1

            val_loop.set_postfix(loss=active_total.item())

            preds_bin   = (mask_predictions.float() > 0).float()
            targets_bin = (mask_tensor.float() > 0.5).float()

            monai_dice(y_pred=preds_bin, y=targets_bin)
            monai_iou(y_pred=preds_bin, y=targets_bin)
            monai_prec(y_pred=preds_bin, y=targets_bin)
            monai_rec(y_pred=preds_bin, y=targets_bin)
            monai_f1(y_pred=preds_bin, y=targets_bin)
            monai_acc(y_pred=preds_bin, y=targets_bin)

            if vis_filenames_set:
                for j, fname in enumerate(batch_filenames):
                    if fname in vis_filenames_set and fname not in vis_collected:
                        vis_collected[fname] = (
                            image_tensor[j].cpu().float(),      
                            mask_predictions[j].cpu().float(),  
                        )

    if writer is not None and epoch is not None and vis_filenames:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for fname in vis_filenames:
            if fname not in vis_collected:
                continue
            img_t, pred_t = vis_collected[fname]

            orig_img = img_t.permute(1, 2, 0).numpy()  
            orig_img = (orig_img * std) + mean
            orig_img = np.clip(orig_img, 0.0, 1.0)
            
            H, W = orig_img.shape[:2]

            prob = torch.sigmoid(pred_t[0])  
            prob_resized = F.interpolate(
                prob.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode='bilinear', align_corners=False
            )[0, 0]

            mask_np = (prob_resized > 0.5).numpy()

            orig_img = (orig_img * 255).astype(np.uint8)
            overlay  = orig_img.copy()
            
            overlay[mask_np] = np.clip(
                overlay[mask_np] * 0.5 + np.array([255, 0, 0]) * 0.5, 0, 255
            ).astype(np.uint8)

            overlay_tensor = torch.from_numpy(overlay).permute(2, 0, 1)
            writer.add_image(f'Predições/{fname}', overlay_tensor, epoch + 1, dataformats='CHW')

    n = max(1, num_val_batches)
    metrics_summary = {
        "loss_total":   val_loss_total   / n,
        "loss_bce":      val_loss_bce     / n,
        "loss_dice":    val_loss_dice    / n,
        "loss_tversky": val_loss_tversky / n,
        "iou":          monai_iou.aggregate().item(),
        "f1":           monai_f1.aggregate()[0].item(),
        "dice_coef":    monai_dice.aggregate().item(),
        "precision":    monai_prec.aggregate()[0].item(),
        "recall":       monai_rec.aggregate()[0].item(),
        "pixel_acc":    monai_acc.aggregate()[0].item(),
    }
    return metrics_summary


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUÇÃO DO MODELO
# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_type, encoder_name, in_channels, unfreeze_encoder_blocks, unfreeze_decoder_blocks, device, encoder_weights="imagenet", attention_type=None):
    arch = smp.UnetPlusPlus if model_type == "unet++" else smp.Unet
    
    model = arch(encoder_name=encoder_name, encoder_weights=encoder_weights,
                 in_channels=in_channels, decoder_upsampling="bilinear",
                 classes=1, activation=None,
                 decoder_attention_type=attention_type).to(device)

    for param in model.parameters(): param.requires_grad = False

    if unfreeze_decoder_blocks == -1:
        for param in model.decoder.parameters(): param.requires_grad = True
    elif unfreeze_decoder_blocks > 0:
        blocks = list(model.decoder.blocks.values()) if isinstance(model.decoder.blocks, torch.nn.ModuleDict) else list(model.decoder.blocks)
        target_dec_blocks = blocks[-unfreeze_decoder_blocks:]
        for block in target_dec_blocks:
            for param in block.parameters(): param.requires_grad = True

    for param in model.segmentation_head.parameters(): param.requires_grad = True

    encoder_children = list(model.encoder.children())
    if unfreeze_encoder_blocks == -1 or unfreeze_encoder_blocks >= len(encoder_children):
        target_enc_blocks = encoder_children
    elif unfreeze_encoder_blocks > 0:
        target_enc_blocks = encoder_children[-unfreeze_encoder_blocks:]
    else:
        target_enc_blocks = []

    for child in target_enc_blocks:
        for param in child.parameters(): param.requires_grad = True

    return model


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING U-NET
# ─────────────────────────────────────────────────────────────────────────────
def finetune_unet(model, train_loader, val_loader, val_dataset, output_dir, epochs=300,
                 warmup_epochs=10, lr_decoder=1e-4, lr_encoder=1e-5, weight_decay=1e-4,
                 patience=40, loss_type="bce_tversky", alpha=0.5, beta=0.5):

    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_unet_model.pth")

    bce_fn     = torch.nn.BCEWithLogitsLoss()
    dice_fn    = MonaiDiceLoss(sigmoid=True)
    tversky_fn = MonaiTverskyLoss(sigmoid=True, alpha=alpha, beta=beta)

    writer = SummaryWriter(log_dir=output_dir)

    encoder_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
    decoder_params = filter(lambda p: p.requires_grad, list(model.decoder.parameters()) + list(model.segmentation_head.parameters()))

    optimizer_groups = [{'params': list(decoder_params), 'lr': lr_decoder}]
    encoder_params_list = list(encoder_params)
    if len(encoder_params_list) > 0:
        optimizer_groups.append({'params': encoder_params_list, 'lr': lr_encoder})

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=weight_decay)
    
    # ======= SCHEDULING SEGUIDO ESTRITAMENTE POR ÉPOCA ======= #
    # warmup_epochs define diretamente o número bruto de chamadas de step()
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01,  # Começa com 1% do LR nominal
        end_factor=1.0,     # Cresce linearmente até 100%
        total_iters=warmup_epochs
    )

    # O Cosseno assume o controle pelas épocas restantes do experimento
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs, 
        eta_min=1e-7
    )

    # Conexão sequencial mudando a chave na virada exata da época estipulada
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    print(f"[INFO] 🎯 SequentialLR por ÉPOCA Ativado:")
    print(f"       -> Épocas 1 a {warmup_epochs}: Warm-up Linear puro (Subindo até LR cheio)")
    print(f"       -> Épocas {warmup_epochs + 1} a {epochs}: Decaimento por Cosseno Nativo (T_max={epochs - warmup_epochs})")
    # ========================================================= #
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    num_vis = min(8, len(val_dataset))
    vis_indices = np.random.choice(len(val_dataset), num_vis, replace=False)
    vis_filenames = []
    for idx in vis_indices:
        _, _, fname = val_dataset[idx]
        vis_filenames.append(fname)

    try:
        for epoch in range(epochs):
            start_time = time.time()

            model.train()
            train_loss_total  = 0.0
            num_train_batches = 0

            train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1:03d}/{epochs} [Train]", leave=False, colour='blue')

            for image_tensor, mask_tensor, _ in train_loop:
                image_tensor = image_tensor.to(device)
                mask_tensor  = mask_tensor.to(device)

                optimizer.zero_grad()

                # Ativação do bf16 no loop de treinamento
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    mask_predictions = model(image_tensor)
                    if mask_predictions.shape[-2:] != mask_tensor.shape[-2:]:
                        mask_predictions = F.interpolate(
                            mask_predictions, size=mask_tensor.shape[-2:],
                            mode="bilinear", align_corners=False
                        )
                    l_bce     = bce_fn(mask_predictions, mask_tensor)
                    l_dice    = dice_fn(mask_predictions, mask_tensor)
                    l_tversky = tversky_fn(mask_predictions, mask_tensor)
                    active_total = l_bce + (l_dice if loss_type == "bce_dice" else l_tversky)

                # Retropropagação limpa e direta
                active_total.backward()
                optimizer.step()

                train_loss_total += active_total.item()
                num_train_batches += 1
                
                train_loop.set_postfix(loss=active_total.item())

            avg_train_total = train_loss_total / max(1, num_train_batches)

            val_metrics = validate_unet(
                model=model, val_loader=val_loader, device=device,
                loss_type=loss_type, bce_fn=bce_fn, dice_fn=dice_fn, tversky_fn=tversky_fn,
                writer=writer, vis_filenames=vis_filenames, epoch=epoch,
            )

            # ── PASSO DO SCHEDULER EXECUTADO UMA ÚNICA VEZ NO FINAL DA ÉPOCA ── #
            scheduler.step()

            end_time = time.time()
            mins, secs = int((end_time - start_time) // 60), int((end_time - start_time) % 60)

            print(f"Epoch {epoch + 1:03d}/{epochs} | Time: {mins}m {secs}s | T-Loss: {avg_train_total:.4f} | V-Loss: {val_metrics['loss_total']:.4f} | mIoU: {val_metrics['iou']:.4f}")

            history["train_loss"].append(avg_train_total)
            history["val_loss"].append(val_metrics["loss_total"])
            history["val_iou"].append(val_metrics["iou"])

            writer.add_scalar('Loss/Train', avg_train_total, epoch + 1)
            writer.add_scalar('Loss/Validation', val_metrics["loss_total"], epoch + 1)
            writer.add_scalar('Metrics/mIoU', val_metrics["iou"], epoch + 1)

            # Gravação do Learning Rate atual para checagem da curva no TensorBoard
            current_lrs = [g['lr'] for g in optimizer.param_groups]
            writer.add_scalar('Params/LR_Decoder', current_lrs[0], epoch + 1)
            if len(current_lrs) > 1:
                writer.add_scalar('Params/LR_Encoder', current_lrs[1], epoch + 1)

            if val_metrics["loss_total"] < best_val_loss:
                best_val_loss = val_metrics["loss_total"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"🌟 Novo melhor modelo salvo! (Val Loss reduziu para {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n🛑 EARLY STOPPING ACIONADO!")
                    break

    except KeyboardInterrupt:
        print("\n🛑 O treinamento foi abortado manualmente (Ctrl+C). Salvando gráficos e encerrando...")

    finally:
        writer.close()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history["train_loss"]) + 1)
        if len(epochs_range) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="blue", lw=2)
            plt.plot(epochs_range, history["val_loss"],   label="Val Loss",   color="red",  lw=2, linestyle="--")
            plt.title("Total Loss: Train vs Validation")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(True, linestyle=":")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "loss_train_vs_val.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(epochs_range, history["val_iou"], label="Validation mIoU", color="green", lw=2)
            plt.title("Validation mIoU Over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("mIoU")
            plt.grid(True, linestyle=":")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "validation_miou.png"), dpi=150)
            plt.close()
            print(f"📊 Gráficos salvos em: {output_dir}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# BLOCO MAIN COM CONFIGURAÇÕES E PASTAS (STANDALONE RUN)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DATA_DIR = "/homeLocal/kayo-lage/SIBGRAPI_CAFE_DATA"
    TILE_SZ       = 1024  
    OVP           = 0.8
    BG_OVP        = 0.6 if TILE_SZ==1024 else 0.4
    MANUAL_DATA   = True

    ENCODER                 = "resnet101" 
    MODEL_TYPE              = "unet++"  
    IN_CHANNELS             = 3
    UNFREEZE_ENCODER_BLOCKS = 0
    UNFREEZE_DECODER_BLOCKS = -1 

    if UNFREEZE_DECODER_BLOCKS != -1 and UNFREEZE_ENCODER_BLOCKS > 0:
        print(f'UNFREEZE_DECODER_BLOCKS = {UNFREEZE_DECODER_BLOCKS} < UNFREEZE_ENCODER_BLOCKS = {UNFREEZE_ENCODER_BLOCKS}\n')
        sys.exit(-1)

    MODEL_TYPE = MODEL_TYPE.lower()
    ENCODER    = ENCODER.lower()

    LOSS_TYPE = "bce_tversky"
    ALPHA     = 0.4
    BETA      = 0.6

    EPOCHS        = 300
    WARMUP_EPOCHS = 10  
    LR_DECODER   = 1e-4
    LR_ENCODER   = 1e-5
    BATCH_SIZE   = 8
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 40

    AUGMENTATIONS = {
        "hflip":  0.5,
        "vflip":  0.5,
        "rotate": [0, 90, 180, 270]
    }

    for TYPE_DATA in ['cluster']:

        print(f"\n{'='*80}")
        print(f" INICIANDO EXPERIMENTO U-NET PARA: {TYPE_DATA.upper()}")
        print(f"{'='*80}\n")

        NAME_DATASET_DIR  = f'dataset_semantico_{TYPE_DATA}'
        POS_FIX_MANUAL    = '_manual' if MANUAL_DATA else ''
        NAME_DATASET_DIR += POS_FIX_MANUAL
        POS_FIX           = f'_tilesz1024_ovp{OVP}_bgovp{BG_OVP}'
        NAME_DATASET_DIR += POS_FIX

        DATASET_DIR = os.path.join(BASE_DATA_DIR, NAME_DATASET_DIR)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        aug_str   = "Aug_" + "".join([k[0].upper() for k in AUGMENTATIONS.keys()]) if AUGMENTATIONS else "AugOff"

        run_name = (
            f"run_{timestamp}_{MODEL_TYPE}_{ENCODER}_{TYPE_DATA}{POS_FIX_MANUAL}"
            f"_tilesz{TILE_SZ}_ovp{OVP}_bgovp{BG_OVP}_{LOSS_TYPE}_{aug_str}"
            f"_ep{EPOCHS}_lrDec{LR_DECODER}_lrEnc{LR_ENCODER}_ufE{UNFREEZE_ENCODER_BLOCKS}_ufD{UNFREEZE_DECODER_BLOCKS}_wd{WEIGHT_DECAY}_bs{BATCH_SIZE}_wu{WARMUP_EPOCHS}_pat{PATIENCE}"
        )
        RUN_OUTPUT_DIR = os.path.join("./runs", run_name)
        os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

        dataset_info_path = os.path.join(RUN_OUTPUT_DIR, "dataset_info.json")
        dataset_info = {
            "DATASET_NAME": NAME_DATASET_DIR,
            "BASE_DATASET_DIR": DATASET_DIR,
            "ENCODER": ENCODER,
            "MODEL_TYPE": MODEL_TYPE,
            "IN_CHANNELS": IN_CHANNELS,
            "UNFREEZE_ENCODER_BLOCKS": UNFREEZE_ENCODER_BLOCKS,
            "UNFREEZE_DECODER_BLOCKS": UNFREEZE_DECODER_BLOCKS,
            "TYPE_DATA": TYPE_DATA,
            "OVP": OVP,
            "BG_OVP": BG_OVP,
            "MANUAL_DATA": MANUAL_DATA,
            "timestamp": timestamp
        }
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=4)

        train_img_dir  = os.path.join(DATASET_DIR, "images", "train")
        train_mask_dir = os.path.join(DATASET_DIR, "labels", "train")
        val_img_dir    = os.path.join(DATASET_DIR, "images", "val")
        val_mask_dir   = os.path.join(DATASET_DIR, "labels", "val")

        train_dataset = CoffeeDataset(train_img_dir, train_mask_dir, augmentations=AUGMENTATIONS)
        val_dataset   = CoffeeDataset(val_img_dir, val_mask_dir, augmentations=None)

        num_train    = len(train_dataset)
        num_val      = len(val_dataset)
        total_images = num_train + num_val

        if total_images > 0:
            train_pct = (num_train / total_images) * 100
            val_pct   = (num_val   / total_images) * 100
            print(f"\n [ESTATÍSTICAS DO DATASET - {TYPE_DATA.upper()}]")
            print(f"    ▶ Total de amostras : {total_images}")
            print(f"    ▶ Treino            : {num_train} imagens ({train_pct:.1f}%)")
            print(f"    ▶ Validação         : {num_val} imagens ({val_pct:.1f}%)\n")
        else:
            sys.exit(-1)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_model(
            model_type=MODEL_TYPE, encoder_name=ENCODER, in_channels=IN_CHANNELS,
            unfreeze_encoder_blocks=UNFREEZE_ENCODER_BLOCKS, unfreeze_decoder_blocks=UNFREEZE_DECODER_BLOCKS,
            device=device
        )

        trained_model = finetune_unet(
            model=model, train_loader=train_loader, val_loader=val_loader, val_dataset=val_dataset,
            output_dir=RUN_OUTPUT_DIR, loss_type=LOSS_TYPE, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
            lr_decoder=LR_DECODER, lr_encoder=LR_ENCODER, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
            alpha=ALPHA, beta=BETA,
        )

        del trained_model, model, train_loader, val_loader, train_dataset, val_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(" 🎉 TODOS OS TREINAMENTOS FORAM CONCLUÍDOS COM SUCESSO! 🎉")
    print("=" * 80) 