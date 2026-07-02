# main.py

import os
import sys
import json
import datetime
import traceback
import gc
import torch
from torch.utils.data import DataLoader

from coffee_data_loader import CoffeeDataset
import train_unet
import eval_unet

if __name__ == "__main__":
    # ================= CONFIGURAÇÕES DE DIRETÓRIO E DADOS ================= #
    BASE_DATA_DIR = "/homeLocal/kayo-lage/SIBGRAPI_CAFE_DATA"
    TILE_SZ       = 1024  
    OVP           = 0.8
    BG_OVP        = 0.4
    MANUAL_DATA   = True

    # ================= CONFIGURAÇÕES DO MODELO ============================ #
    IN_CHANNELS             = 3
    UNFREEZE_DECODER_BLOCKS = -1

    # ============================== LOSSES E HIPERPARÂMETROS ============== #
    LOSS_TYPE     = "bce_tversky"
    ALPHA         = 0.4
    BETA          = 0.6

    EPOCHS        = 300
    WARMUP_EPOCHS = 10

    LR_DECODER   = 1e-4
    LR_ENCODER   = 1e-5

    WEIGHT_DECAY = 1e-4
    PATIENCE     = 40

    AUGMENTATIONS = {
        "hflip":  0.5,
        "vflip":  0.5,
        "rotate": [0, 90, 180, 270]
    }

    # ================= FILA DE EXPERIMENTOS CONTROLADA ==================== #
    # Definimos explicitamente o tipo de dado (cluster ou ndvi) por modelo
    EXPERIMENTOS = [
        {
            "type_data": "cluster",
            "model_type": "unet++", 
            "encoder": "resnet101", 
            "weights": "imagenet",
            "attention": "scse",
            "unfreeze_enc": 2, 
            "batch_size": 3
        },

        {
            "type_data": "cluster",
            "model_type": "unet++", 
            "encoder": "resnext101_32x16d", 
            "weights": "ssl",
            "attention": "scse",
            "unfreeze_enc": 1, 
            "batch_size": 3  
        },

        {
            "type_data": "cluster",
            "model_type": "unet++", 
            "encoder": "timm-efficientnet-l2", 
            "weights": "noisy-student", 
            "attention": "scse",
            "unfreeze_enc": 1, 
            "batch_size": 2   
        },

        {
            "type_data": "cluster",
            "model_type": "unet", 
            "encoder": "resnext101_32x16d", 
            "weights": "ssl",
            "attention": "scse",
            "unfreeze_enc": 2, 
            "batch_size": 8   
        },

        {
            "type_data": "cluster",
            "model_type": "unet", 
            "encoder": "timm-efficientnet-l2", 
            "weights": "noisy-student", 
            "attention": "scse",
            "unfreeze_enc": 2, 
            "batch_size": 8   
        }
    ]

    # ================= LOOP DE ORQUESTRAÇÃO DIRETO ======================== #
    # Agora iteramos diretamente pela lista de experimentos sem duplicações
    for exp in EXPERIMENTOS:
        
        TYPE_DATA = exp["type_data"].lower()
        MODEL_TYPE = exp["model_type"].lower()
        ENCODER = exp["encoder"].lower()
        ENCODER_WEIGHTS = exp["weights"]
        ATTENTION = exp["attention"] 
        BATCH_SIZE = exp["batch_size"]
        UNFREEZE_ENCODER_BLOCKS = exp["unfreeze_enc"]

        # Verificação estrita de sanidade de congelamento
        if UNFREEZE_DECODER_BLOCKS != -1 and UNFREEZE_ENCODER_BLOCKS > 0:
            print(f'ERRO: UNFREEZE_DECODER_BLOCKS = {UNFREEZE_DECODER_BLOCKS} < UNFREEZE_ENCODER_BLOCKS = {UNFREEZE_ENCODER_BLOCKS}\n')
            sys.exit(-1)

        print(f"\n{'='*80}")
        print(f" 🚀 PIPELINE ATIVADA | ORIGEM DA MÁSCARA: {TYPE_DATA.upper()}")
        print(f" 🛠️  ARQUITETURA: {MODEL_TYPE.upper()} + {ENCODER.upper()} ({ENCODER_WEIGHTS})")
        print(f" 💡 ATENÇÃO: {ATTENTION} | UNFREEZE ENCODER: {UNFREEZE_ENCODER_BLOCKS} | BATCH SIZE: {BATCH_SIZE}")
        print(f"{'='*80}\n")

        # 1. Configuração Dinâmica de Caminhos baseada no TYPE_DATA do experimento
        NAME_DATASET_DIR  = f'dataset_semantico_{TYPE_DATA}'
        POS_FIX_MANUAL    = '_manual' if MANUAL_DATA else ''
        NAME_DATASET_DIR += POS_FIX_MANUAL
        POS_FIX           = f'_tilesz{TILE_SZ}_ovp{OVP}_bgovp{BG_OVP}'
        NAME_DATASET_DIR += POS_FIX

        DATASET_DIR = os.path.join(BASE_DATA_DIR, NAME_DATASET_DIR)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        aug_str   = "Aug_" + "".join([k[0].upper() for k in AUGMENTATIONS.keys()]) if AUGMENTATIONS else "AugOff"

        run_name = (
            f"run_{timestamp}_{MODEL_TYPE}_{ENCODER}_{TYPE_DATA}{POS_FIX_MANUAL}"
            f"_tilesz{TILE_SZ}_ovp{OVP}_bgovp{BG_OVP}_{LOSS_TYPE}_{aug_str}"
            f"_ep{EPOCHS}_lrDec{LR_DECODER}_lrEnc{LR_ENCODER}_ufE{UNFREEZE_ENCODER_BLOCKS}_ufD{UNFREEZE_DECODER_BLOCKS}_wd{WEIGHT_DECAY}_bs{BATCH_SIZE}_wu{WARMUP_EPOCHS}_pat{PATIENCE}_att{ATTENTION}"
        )
        RUN_OUTPUT_DIR = os.path.join("./runs", run_name)
        os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

        # Salvando os metadados do experimento atual
        dataset_info_path = os.path.join(RUN_OUTPUT_DIR, "dataset_info.json")
        dataset_info = {
            "DATASET_NAME": NAME_DATASET_DIR,
            "BASE_DATASET_DIR": DATASET_DIR,
            "ENCODER": ENCODER,
            "ENCODER_WEIGHTS": ENCODER_WEIGHTS,
            "MODEL_TYPE": MODEL_TYPE,
            "IN_CHANNELS": IN_CHANNELS,
            "UNFREEZE_ENCODER_BLOCKS": UNFREEZE_ENCODER_BLOCKS,
            "UNFREEZE_DECODER_BLOCKS": UNFREEZE_DECODER_BLOCKS,
            "TYPE_DATA": TYPE_DATA,
            "OVP": OVP,
            "BG_OVP": BG_OVP,
            "MANUAL_DATA": MANUAL_DATA,
            "BATCH_SIZE": BATCH_SIZE,
            "timestamp": timestamp,
            "ATTENTION": ATTENTION  # 👈 ADICIONE ESTA LINHA AQUI
        }
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=4)

        # 2. Resolução de caminhos internos do dataset
        train_img_dir  = os.path.join(DATASET_DIR, "images", "train")
        train_mask_dir = os.path.join(DATASET_DIR, "labels", "train")
        val_img_dir    = os.path.join(DATASET_DIR, "images", "val")
        val_mask_dir   = os.path.join(DATASET_DIR, "labels", "val")

        train_dataset = CoffeeDataset(train_img_dir, train_mask_dir, augmentations=AUGMENTATIONS)
        val_dataset   = CoffeeDataset(val_img_dir, val_mask_dir, augmentations=None)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ====================================================================
        # FASE 1: TREINAMENTO NATIVO EM BF16
        # ====================================================================
        try:
            print(f"⏳ [FASE 1] Construindo rede e carregando pesos pré-treinados...")
            model = train_unet.build_model(
                model_type=MODEL_TYPE, encoder_name=ENCODER, in_channels=IN_CHANNELS,
                unfreeze_encoder_blocks=UNFREEZE_ENCODER_BLOCKS, unfreeze_decoder_blocks=UNFREEZE_DECODER_BLOCKS,
                device=device, encoder_weights=ENCODER_WEIGHTS, attention_type=ATTENTION
            )

            trained_model = train_unet.finetune_unet(
                model=model, train_loader=train_loader, val_loader=val_loader, val_dataset=val_dataset,
                output_dir=RUN_OUTPUT_DIR, loss_type=LOSS_TYPE, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
                lr_decoder=LR_DECODER, lr_encoder=LR_ENCODER, weight_decay=WEIGHT_DECAY,
                patience=PATIENCE, alpha=ALPHA, beta=BETA,
            )
        except (Exception, KeyboardInterrupt) as e:
            print(f"\n⚠️ Ocorreu uma interrupção ou falha no treino do {ENCODER}: {e}")
            traceback.print_exc()
            print("Avançando diretamente para a fase de testes na tentativa de ler o melhor checkpoint...")

        # ====================================================================
        # FASE 2: AVALIAÇÃO E INFERÊNCIA NO TESTE MANUAL
        # ====================================================================
        try:
            print(f"⏳ [FASE 2] Rodando avaliação completa no conjunto de teste...")
            eval_unet.evaluate_and_infer(
                run_dir=RUN_OUTPUT_DIR,
                batch_size=BATCH_SIZE
            )
            print(f"✅ SUCESSO: Rotina concluída com êxito para {ENCODER} | Dataset: {TYPE_DATA}!")
            
        except Exception as e:
            print(f"\n❌ ERRO CRÍTICO ao avaliar o encoder {ENCODER}:")
            traceback.print_exc()
            print("Ignorando falha e passando para o próximo experimento para proteger a fila...")

        # ====================================================================
        # LIMPEZA FORÇADA DE MEMÓRIA (VRAM)
        # ====================================================================
        finally:
            if 'trained_model' in locals(): del trained_model
            if 'model' in locals(): del model
            del train_loader, val_loader, train_dataset, val_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    print("\n" + "=" * 80)
    print(" 🎉 ORQUESTRAÇÃO SELETIVA DE EXPERIMENTOS CONCLUÍDA COM SUCESSO! 🎉")
    print("=" * 80)