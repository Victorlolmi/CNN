"""
============================================================
 Treinamento YOLO26 SEMANTIC sobre dataset Semântico (NDVI)
============================================================

Assume que o gerador (gerador_dataset_yolo_semantico.py) já produziu:

  dataset/
    images/{train,val,test}/*.png   (RGB 8-bit)
    masks/{train,val,test}/*.png    (class-map 1 canal: 0=fundo, 1=cafe, 255=ignore)
    data.yaml                        (com a chave masks_dir: masks)

Tarefa: SEMANTIC . Métricas = mIoU + pixel accuracy.

============================================================
"""

import os
from pathlib import Path

import yaml
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO


# ==========================================
# CONFIGURAÇÕES
# ==========================================
ARQUIVO_YAML = "dataset/data.yaml"

MODELO_BASE = "yolo26x-sem.pt"
NOME_EXPERIMENTO = "semantico_cluster_final_1024"
PASTA_PROJETO = "resultados_cafe"


# ==========================================
# 1. VALIDAÇÃO DO DATASET
# ==========================================
def verificar_dataset() -> None:
    """Confere o dataset semântico: imagens + máscaras PNG pareadas + masks_dir no yaml."""
    yaml_path = Path(ARQUIVO_YAML)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"❌ data.yaml não encontrado em: {yaml_path.resolve()}\n"
            "   Rode primeiro o gerador semântico."
        )

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", yaml_path.parent))
    masks_dir = cfg.get("masks_dir")
    print(f"📂 Dataset em: {base.resolve()}")
    print(f"   classes: {cfg.get('names')}")
    print(f"   masks_dir: {masks_dir!r}  (modo PNG semântico ativo: {masks_dir is not None})")

    if masks_dir is None:
        print("   [warn] 'masks_dir' ausente — o loader cairia no modo de polígonos.\n"
              "          Para máscara PNG lossless, mantenha masks_dir: masks no yaml.")

    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if rel is None:
            print(f"   [warn] split '{split}' ausente no yaml")
            continue
        img_dir = base / rel
        if not img_dir.exists():
            raise FileNotFoundError(f"❌ diretório de imagens não existe: {img_dir}")
        n_imgs = sum(1 for _ in img_dir.glob("*.png")) + \
                 sum(1 for _ in img_dir.glob("*.jpg"))

        # máscaras: substitui o componente 'images' por masks_dir
        if masks_dir:
            msk_dir = Path(str(img_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}{masks_dir}{os.sep}"))
            n_msks = sum(1 for _ in msk_dir.glob("*.png")) if msk_dir.exists() else 0
            estado = "ok" if n_imgs == n_msks else "‼ DESPAREADO"
            print(f"   {split:<5}: {n_imgs:>5} imagens | {n_msks:>5} máscaras  [{estado}]  ({img_dir})")
        else:
            print(f"   {split:<5}: {n_imgs:>5} imagens  ({img_dir})")

    print("✅ Dataset verificado.\n")


# ==========================================
# 2. TENSORBOARD
# ==========================================
def configurar_tensorboard() -> None:
    print("📊 Inicializando TensorBoard...")
    try:
        os.makedirs(PASTA_PROJETO, exist_ok=True)
        writer = SummaryWriter(log_dir=PASTA_PROJETO)
        writer.add_text("Status", "TensorBoard forçado com sucesso!")
        writer.close()
        print(f"✅ Log do TensorBoard em: {Path(PASTA_PROJETO).resolve()}\n")
    except Exception as exc:
        print(f"❌ ERRO NO TENSORBOARD: {exc}\n")


# ==========================================
# 3. TREINAMENTO
# ==========================================
def treinar_modelo():
    print(f"🧠 Carregando arquitetura SEMANTIC {MODELO_BASE}...")
    # task='semantic' é inferido do checkpoint -sem; explícito p/ garantir.
    model = YOLO(MODELO_BASE, task="semantic")

    print("🔥 Iniciando o treinamento semântico. Pegue um café!\n")
    resultados = model.train(
        data=ARQUIVO_YAML,

        epochs=200,
        imgsz=640,                      
        batch=25,                        
        device=0,
        workers=6,
        patience=40,
        name=NOME_EXPERIMENTO,
        project=PASTA_PROJETO,
        save_period=10,

        # ---- Geometria (transforma a máscara junto) ----
        degrees=180.0,         
        fliplr=0.5,
        flipud=0.5,            

        translate=0.0,
        scale=0.0,             
        shear=0.0,
        perspective=0.0,       # ortofotos já ortorretificadas

        # ---- Composição ----
        mosaic=0.0,            

        # ---- Cor ----
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # ---- Regularização ----
        weight_decay=0.001,
        dropout=0.1,
        erasing=0.0,

        # ---- Sistema ----
        cache='ram',          
        amp=True,
        seed=42,
        deterministic=True,

        save=True,
        exist_ok=True,
        verbose=True,
        val=True,
        plots=True,
    )
    print("\n🏆 Treinamento semântico finalizado!")
    return resultados


# ==========================================
# 4. VALIDAÇÃO FINAL (mIoU / pixel accuracy)
# ==========================================
def validar_modelo():
    melhor = Path(PASTA_PROJETO) / "semantic" / NOME_EXPERIMENTO / "weights" / "best.pt"
    if not melhor.exists():
        candidatos = list(Path(PASTA_PROJETO).rglob("best.pt"))
        if candidatos:
            melhor = candidatos[-1]
        else:
            print("[warn] best.pt não encontrado para validação.")
            return
            
    print(f"\n📐 Validando {melhor} no conjunto de TESTE...")
    model = YOLO(str(melhor), task="semantic")
    
   
    metrics = model.val(
        data=ARQUIVO_YAML, 
        split="test",
        project=PASTA_PROJETO,                  
        name=f"{NOME_EXPERIMENTO}_teste_final_metrics"   
    ) 
    
    try:
        print(f"   mIoU (Teste)            = {metrics.miou:.4f}")
        print(f"   pixel accuracy (Teste)  = {metrics.pixel_accuracy:.4f}")
    except Exception:
        print(f"   métricas: {metrics}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    verificar_dataset()
    configurar_tensorboard()
    treinar_modelo()
    validar_modelo()