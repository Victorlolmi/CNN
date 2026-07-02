"""
============================================================
 Pipeline de Dataset de Segmentação SEMÂNTICA 
 (PNG class-map BINÁRIO + regiões proibidas + split por área
  + anti-vazamento + DEBUGS COMPLETOS + NORMALIZAÇÃO GLOBAL)
============================================================
ESTRUTURA GERADA:
  dataset/
    images/{train,val,test}/*.png   (RGB 8-bit)
    masks/{train,val,test}/*.png    (class-map 1 canal: 0/1)
    data.yaml                        (com masks_dir: masks)
============================================================
"""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from skimage.measure import find_contours, label, regionprops
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÃO
# ============================================================
PathLike = Optional[Union[str, List[str]]]

@dataclass
class Config:
    pasta_jovens: str = "/path/to/cafe_jovens"
    pasta_adultos: str = "/path/to/cafe_adultos"
    ortofoto: str = "/path/to/ortofoto_fazenda.tif"
    saida: str = "/path/to/dataset_saida"

    mascaras_jovens: Optional[Union[str, Dict[str, str]]] = None
    mascaras_adultos: Optional[Union[str, Dict[str, str]]] = None
    sufixos_mascara: Tuple[str, ...] = ("", "_mask", "_mascara", "_msk")

    regiao_proibida: PathLike = None

    # Tiling
    tile_size: int = 640
    overlap: float = 0.80
    min_cobertura_mask: float = 0.05

    # Filtro de área mínima de café (em pixels) para um tile valer a pena
    min_area_cafe_px: int = 10

    # Mascaramento (anti-vazamento)
    mascarar_outros_splits: bool = True
    mascarar_proibida: bool = True
    min_cobertura_pos_mascara: float = 0.15

    # Split por área (garante separação geográfica entre splits)
    proporcao: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 45
    tolerancia_area_log: float = 0.10

    # Normalização
    sample_to_calc_norm: float = 1.0

    # Mineração de backgrounds (negativos)
    salvar_backgrounds: bool = True
    overlap_background: float = 0.40
    limite_backgrounds: Optional[int] = None
    split_background: str = "train"
    prefixo_background: str = "bg"
    n_backgrounds_val: int = 40
    n_backgrounds_test: int = 40

    # Debug e Plots
    salvar_debug_supermosaico: bool = True
    salvar_debug_tiles_mapa: bool = True
    salvar_debug_distribuicao_areas: bool = True
    salvar_debug_csv_tiles: bool = True
    salvar_debug_distribuicao_instancias: bool = True
    salvar_debug_mapa_splits: bool = True
    salvar_debug_overlay_mascaras: bool = True   # NOVO: overlays img+máscara p/ inspeção
    n_overlays_por_split: int = 12               # NOVO
    debug_max_size_px: int = 4096

    # Mapa de splits
    mapa_splits_downsample: int = 4
    mapa_splits_anotar_nome: bool = True


# ============================================================
# CONSTANTES DE PLOT
# ============================================================
CORES_SPLIT = {
    "train": "#43A047",
    "val":   "#EC407A",
    "test":  "#1E88E5",
}
NOME_SPLIT_PT = {"train": "treino", "val": "validação", "test": "teste"}
HATCH_CLASSE = {"jovem": "", "adulto": "////"}

# Valores do class-map semântico (Binário)
SEM_FUNDO = 0
SEM_CAFE = 1


# ============================================================
# 1. MÁSCARAS SEMÂNTICAS DE CAFÉ (PNG)
# ============================================================
def encontrar_mascara_png(talhao_path: str,
                          mascaras: Optional[Union[str, Dict[str, str]]],
                          sufixos: Tuple[str, ...] = ("", "_mask")
                          ) -> Optional[str]:
    if mascaras is None:
        return None
    stem = Path(talhao_path).stem
    if isinstance(mascaras, dict):
        return mascaras.get(stem)
    pasta = Path(mascaras)
    if not pasta.exists():
        return None
    for sufixo in sufixos:
        nome = f"{stem}{sufixo}"
        for ext in (".png", ".PNG"):
            cand = pasta / f"{nome}{ext}"
            if cand.exists():
                return str(cand)
        matches = (list(pasta.rglob(f"{nome}.png"))
                   + list(pasta.rglob(f"{nome}.PNG")))
        if matches:
            return str(matches[0])
    return None

def carregar_mascara_cafe(mask_path: Optional[str],
                          target_shape: Tuple[int, int],
                          nome_talhao: str = "") -> Optional[np.ndarray]:
    if mask_path is None or not Path(mask_path).exists():
        return None
    img = Image.open(mask_path)
    if img.mode != "L":
        img = img.convert("L")
    H, W = target_shape
    if img.size != (W, H):
        print(f"[warn] {nome_talhao}: máscara PNG {img.size} != TIFF "
              f"({W}, {H}). Resize NEAREST aplicado.")
        img = img.resize((W, H), Image.NEAREST)
    return np.array(img) > 0


# ============================================================
# 1b. DIAGNÓSTICO DE GEORREFERÊNCIA
# ============================================================
def diagnosticar_alinhamento(ortofoto_path: str, talhoes: List[dict]) -> None:
    if not talhoes:
        return
    with rasterio.open(ortofoto_path) as src:
        gsd_o = (abs(src.transform.a), abs(src.transform.e))
        crs_o = str(src.crs)
    print(f"\n=== DIAGNÓSTICO DE GEORREFERÊNCIA ===")
    print(f"  ortofoto: gsd_x={gsd_o[0]:.6f}  gsd_y={gsd_o[1]:.6f}  CRS={crs_o}")

    diffs_gsd, diffs_crs, mask_size_mismatch = [], [], []
    for t in talhoes:
        a, _, _, _, e, _ = t["transform"]
        gsd_t = (abs(a), abs(e))
        if not (abs(gsd_t[0] - gsd_o[0]) < 1e-6 and abs(gsd_t[1] - gsd_o[1]) < 1e-6):
            diffs_gsd.append((Path(t["path"]).name, gsd_t))
        if str(t["crs"]) != crs_o:
            diffs_crs.append((Path(t["path"]).name, t["crs"]))
        if t.get("mask_path"):
            try:
                with Image.open(t["mask_path"]) as im:
                    if im.size != (t["width"], t["height"]):
                        mask_size_mismatch.append((Path(t["path"]).name, im.size, (t["width"], t["height"])))
            except Exception:
                pass

    if diffs_gsd: print(f"  [!] {len(diffs_gsd)} talhão(ões) com gsd diferente")
    if diffs_crs: print(f"  [!] {len(diffs_crs)} talhão(ões) com CRS diferente")
    if mask_size_mismatch: print(f"  [!] {len(mask_size_mismatch)} máscara(s) PNG com tamanho diferente do TIFF")
    if not (diffs_gsd or diffs_crs or mask_size_mismatch):
        print(f"  ok: talhões e máscaras alinhados com a ortofoto.")


# ============================================================
# 2. LISTAGEM DOS TALHÕES
# ============================================================
def _mascara_valida_de_data(data: np.ndarray, nodata) -> np.ndarray:
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        m = (np.all(data != nodata, axis=0) & ~np.isnan(data).any(axis=0))
    elif nodata is not None:
        m = ~np.isnan(data).any(axis=0)
    else:
        m = np.any(data > 0, axis=0) & ~np.isnan(data).any(axis=0)
    return m

def listar_talhoes(pastas_classes: Dict[str, str],
                   mascaras_por_classe: Optional[Dict[str, Optional[Union[str, Dict[str, str]]]]] = None,
                   sufixos_mascara: Tuple[str, ...] = ("", "_mask")
                   ) -> List[dict]:
    mascaras_por_classe = mascaras_por_classe or {}
    talhoes = []
    for classe, pasta in pastas_classes.items():
        mascaras = mascaras_por_classe.get(classe)
        for fp in sorted(Path(pasta).rglob("*.tif")):
            mask_path = encontrar_mascara_png(str(fp), mascaras, sufixos=sufixos_mascara)

            if mascaras is not None and mask_path is None:
                print(f"[warn] sem máscara PNG para {fp.name}. Talhão será ignorado.")
                continue

            with rasterio.open(fp) as src:
                data = src.read()
                nodata = src.nodata
                gsd_x, gsd_y = abs(src.transform.a), abs(src.transform.e)
                m_valida = _mascara_valida_de_data(data, nodata)
                area_m2 = int(m_valida.sum()) * gsd_x * gsd_y
                talhoes.append({
                    "path": str(fp),
                    "mask_path": mask_path,
                    "classe": classe,
                    "width": src.width,
                    "height": src.height,
                    "n_bands": src.count,
                    "transform": list(src.transform)[:6],
                    "crs": str(src.crs),
                    "bounds": tuple(src.bounds),
                    "area_m2": float(area_m2),
                })
    return talhoes


# ============================================================
# 3. REGIÕES PROIBIDAS
# ============================================================
def _resolver_lista_tifs(entrada: PathLike) -> List[Path]:
    if entrada is None: return []
    if isinstance(entrada, (list, tuple)): return [Path(p) for p in entrada]
    p = Path(entrada)
    if not p.exists(): return []
    if p.is_file(): return [p]
    if p.is_dir(): return sorted(p.rglob("*.tif"))
    return []

def carregar_regioes_proibidas(entrada: PathLike) -> List[dict]:
    arquivos = _resolver_lista_tifs(entrada)
    proibidas: List[dict] = []
    for fp in arquivos:
        with rasterio.open(fp) as src:
            data = src.read()
            valid = _mascara_valida_de_data(data, src.nodata)
            if not valid.any(): continue
            proibidas.append({
                "path": str(fp),
                "bounds": tuple(src.bounds),
                "transform": src.transform,
                "valid_mask": valid,
                "crs": str(src.crs),
            })
    return proibidas

def tile_intersecta_proibida(bounds_tile, regioes_proibidas) -> bool:
    if not regioes_proibidas: return False
    minx_t, miny_t, maxx_t, maxy_t = bounds_tile
    for r in regioes_proibidas:
        minx_r, miny_r, maxx_r, maxy_r = r["bounds"]
        if maxx_t <= minx_r or minx_t >= maxx_r or maxy_t <= miny_r or miny_t >= maxy_r:
            continue
        try:
            win = from_bounds(max(minx_t, minx_r), max(miny_t, miny_r),
                              min(maxx_t, maxx_r), min(maxy_t, maxy_r), transform=r["transform"])
            col_off = max(0, int(np.floor(win.col_off)))
            row_off = max(0, int(np.floor(win.row_off)))
            col_end = min(r["valid_mask"].shape[1], int(np.ceil(win.col_off + win.width)))
            row_end = min(r["valid_mask"].shape[0], int(np.ceil(win.row_off + win.height)))
            if col_end <= col_off or row_end <= row_off: continue
            if r["valid_mask"][row_off:row_end, col_off:col_end].any(): return True
        except Exception:
            return True
    return False

def mascara_proibida(bounds_tile: Tuple[float, float, float, float],
                     tile_size: int,
                     regioes_proibidas: List[dict]) -> np.ndarray:
    mascara = np.zeros((tile_size, tile_size), dtype=bool)
    if not regioes_proibidas: return mascara
    minx_t, miny_t, maxx_t, maxy_t = bounds_tile
    for r in regioes_proibidas:
        minx_r, miny_r, maxx_r, maxy_r = r["bounds"]
        if maxx_t <= minx_r or minx_t >= maxx_r or maxy_t <= miny_r or miny_t >= maxy_r:
            continue
        try:
            with rasterio.open(r["path"]) as src:
                fill = src.nodata if src.nodata is not None else 0
                data = src.read(
                    window=from_bounds(minx_t, miny_t, maxx_t, maxy_t, transform=src.transform),
                    boundless=True, fill_value=fill, out_shape=(src.count, tile_size, tile_size),
                    resampling=Resampling.nearest,
                )
                m_proibida = _mascara_valida_de_data(data, src.nodata)
                mascara |= m_proibida
        except Exception:
            mascara[:] = True
            return mascara
    return mascara


# ============================================================
# 4. SPLIT POR ÁREA  (separação geográfica = anti-vazamento)
# ============================================================
def _selecao_gulosa_por_area(candidatos: List[dict], alvo_area: float, rng: np.random.Generator) -> Tuple[List[dict], float]:
    candidatos = list(candidatos)
    if not candidatos: return [], 0.0
    selecionados: List[dict] = []
    area_acum = 0.0
    rng.shuffle(candidatos)
    for t in candidatos:
        area_novo = area_acum + t["area_m2"]
        if abs(area_novo - alvo_area) <= abs(area_acum - alvo_area):
            selecionados.append(t)
            area_acum = area_novo
    return selecionados, area_acum

def split_talhoes_por_area(talhoes: List[dict], proporcao: Tuple[float, float, float], seed: int) -> Dict[str, List[dict]]:
    p_train, p_val, p_test = proporcao
    classes = ("jovem", "adulto")
    area_total_classe = {c: sum(t["area_m2"] for t in talhoes if t["classe"] == c) for c in classes}
    alvos = {
        "train": {c: area_total_classe[c] * p_train for c in classes},
        "val":   {c: area_total_classe[c] * p_val   for c in classes},
        "test":  {c: area_total_classe[c] * p_test  for c in classes},
    }

    teste: List[dict] = []
    teste_ids = set()
    for i, c in enumerate(classes):
        pool_c = [t for t in talhoes if t["classe"] == c]
        rng_c = np.random.default_rng(seed + 200 + i)
        sub, _ = _selecao_gulosa_por_area(pool_c, alvos["test"][c], rng_c)
        teste.extend(sub)
        teste_ids.update(id(t) for t in sub)

    resto = [t for t in talhoes if id(t) not in teste_ids]
    val: List[dict] = []
    val_ids = set()
    for i, c in enumerate(classes):
        pool_c = [t for t in resto if t["classe"] == c]
        rng_c = np.random.default_rng(seed + 300 + i)
        val_ratio = p_val / (p_train + p_val) if (p_train + p_val) > 0 else 0.0
        alvo = sum(t["area_m2"] for t in pool_c) * val_ratio
        sub, _ = _selecao_gulosa_por_area(pool_c, alvo, rng_c)
        val.extend(sub)
        val_ids.update(id(t) for t in sub)

    train = [t for t in resto if id(t) not in val_ids]
    return {"train": train, "val": val, "test": teste}


# ============================================================
# 5. ROI + SUPERMOSAICO
# ============================================================
def _bbox_da_mascara(m: np.ndarray):
    if not m.any(): return None
    y_idx, x_idx = np.where(np.any(m, axis=1))[0], np.where(np.any(m, axis=0))[0]
    return int(y_idx[0]), int(y_idx[-1]) + 1, int(x_idx[0]), int(x_idx[-1]) + 1

def criar_supermosaico(talhoes: List[dict], ortofoto_path: str):
    with rasterio.open(ortofoto_path) as src_orto:
        orto_transform = src_orto.transform
        orto_inv = ~orto_transform

    info_talhoes = []
    for i, t in enumerate(talhoes):
        with rasterio.open(t["path"]) as src:
            data, nodata = src.read(), src.nodata

        m_full = _mascara_valida_de_data(data, nodata)
        bbox = _bbox_da_mascara(m_full)
        if bbox is None: continue
        y0, y1, x0, x1 = bbox

        data = np.nan_to_num(data, nan=0)[:, y0:y1, x0:x1]
        m_roi = m_full[y0:y1, x0:x1]

        m_cafe_full = carregar_mascara_cafe(t.get("mask_path"), (t["height"], t["width"]))
        m_cafe_roi = m_roi.copy() if m_cafe_full is None else (m_cafe_full[y0:y1, x0:x1] & m_roi)

        talhao_transform = Affine(*t["transform"])
        x_geo, y_geo = talhao_transform * (x0, y0)
        col_orto, row_orto = int(round((orto_inv * (x_geo, y_geo))[0])), int(round((orto_inv * (x_geo, y_geo))[1]))

        info_talhoes.append({
            "talhao": t, "data": data, "mask_valida": m_roi, "mask_cafe": m_cafe_roi,
            "h_roi": y1 - y0, "w_roi": x1 - x0, "col_orto": col_orto, "row_orto": row_orto,
        })

    if not info_talhoes: raise ValueError("Nenhum talhão válido no split.")

    min_col = min(info["col_orto"] for info in info_talhoes)
    min_row = min(info["row_orto"] for info in info_talhoes)
    max_col = max(info["col_orto"] + info["w_roi"] for info in info_talhoes)
    max_row = max(info["row_orto"] + info["h_roi"] for info in info_talhoes)

    bin_w, bin_h = max_col - min_col, max_row - min_row
    canvas = np.zeros((info_talhoes[0]["data"].shape[0], bin_h, bin_w), dtype=info_talhoes[0]["data"].dtype)
    mask_valida = np.zeros((bin_h, bin_w), dtype=bool)
    mask_cafe = np.zeros((bin_h, bin_w), dtype=bool)
    mapping = []

    for info in info_talhoes:
        t, h, w = info["talhao"], info["h_roi"], info["w_roi"]
        x, y = info["col_orto"] - min_col, info["row_orto"] - min_row

        for b in range(canvas.shape[0]):
            canvas[b, y:y + h, x:x + w] = np.where(info["mask_valida"], info["data"][b], canvas[b, y:y + h, x:x + w])
        mask_valida[y:y + h, x:x + w] |= info["mask_valida"]
        mask_cafe[y:y + h, x:x + w]   |= info["mask_cafe"]

        mapping.append({
            "talhao_path": t["path"], "classe": t["classe"],
            "sm_x": x, "sm_y": y, "width": w, "height": h,
        })

    sm_transform = orto_transform * Affine.translation(min_col, min_row)
    return canvas, mask_valida, mask_cafe, mapping, sm_transform


# ============================================================
# 6. TILES, RECORTES E NORMALIZAÇÃO GLOBAL
# ============================================================
def gerar_tiles_coords(shape_hw, tile_size, overlap):
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    H, W = shape_hw
    ys = list(range(0, H - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < H: ys.append(max(0, H - tile_size))
    xs = list(range(0, W - tile_size + 1, stride))
    if not xs or xs[-1] + tile_size < W: xs.append(max(0, W - tile_size))
    return [(y, x) for y in ys for x in xs]

def recortar_da_ortofoto(ortofoto_path, bounds_tile, tile_size):
    with rasterio.open(ortofoto_path) as src:
        win = from_bounds(*bounds_tile, transform=src.transform)
        return src.read(window=win, boundless=True, fill_value=0, out_shape=(src.count, tile_size, tile_size), resampling=Resampling.bilinear)

def calcular_limites_globais_rgb(ortofoto_path, talhoes_treino, p_low=2, p_high=98, sample_frac=0.05):
    """Amostra pixels apenas do split de treino para calcular limites globais."""
    print(f"\n=== CALCULANDO LIMITES GLOBAIS DE NORMALIZAÇÃO (Treino) ===")
    pixels_validos = []
    
    with rasterio.open(ortofoto_path) as src:
        for t in tqdm(talhoes_treino, desc="Amostrando talhões de treino", unit="talhão"):
            win = from_bounds(*t["bounds"], transform=src.transform)
            data = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
            data_flat = data.reshape(3, -1)
            
            mask_validos = np.any(data_flat > 0, axis=0)
            valid_pixels = data_flat[:, mask_validos]
            
            if valid_pixels.size == 0: 
                continue
                
            n_validos = valid_pixels.shape[1]
            n_amostras = max(1, int(n_validos * sample_frac))
            idx_amostra = np.random.choice(n_validos, n_amostras, replace=False)
            
            pixels_validos.append(valid_pixels[:, idx_amostra])
            
    if not pixels_validos:
        print("Aviso: Nenhum pixel válido encontrado no treino. Usando 0-255.")
        return 0.0, 255.0
        
    todos_pixels = np.concatenate(pixels_validos, axis=1)
    todos_pixels_1d = todos_pixels.flatten()
    lo, hi = np.percentile(todos_pixels_1d, [p_low, p_high])
    
    return float(lo), float(hi)

def to_8bit_rgb_global(data: np.ndarray, global_lo: float, global_hi: float) -> np.ndarray:
    """Normaliza para 8-bit usando limites globais pré-calculados."""
    rgb = data[:3] if data.shape[0] >= 3 else np.repeat(data[:1], 3, axis=0)
    mask_nodata = np.all(rgb == 0, axis=0)
    rgb_float = rgb.astype(np.float32)
    
    if global_hi > global_lo:
        normalizado = ((rgb_float - global_lo) / (global_hi - global_lo) * 255)
    else:
        normalizado = rgb_float
        
    out = np.clip(normalizado, 0, 255).astype(np.uint8)
    out[:, mask_nodata] = 0
    return np.moveaxis(out, 0, -1)

def salvar_imagem_png(data_8bit, path):
    # Recebe a imagem já normalizada em (H, W, 3) 8-bits
    Image.fromarray(data_8bit).save(path)


# ============================================================
# 7. EXPORTAÇÃO SEMÂNTICA (máscara binária -> PNG class-map)
# ============================================================
def construir_mascara_semantica(mask_cafe_tile: np.ndarray) -> np.ndarray:
    """
    Class map de 1 canal para YOLO semantic — PURAMENTE BINÁRIO (sem 255):
      1 = café
      0 = FUNDO/CHÃO -> TUDO que não é café, incluindo:
            - solo, estradas, construções, mata
            - NODATA/preto (zero-fill da ortofoto)
            - região proibida e talhão de OUTRO split (que já foram
              ZERADOS na imagem antes de salvar)
    """
    sem = np.zeros(mask_cafe_tile.shape, dtype=np.uint8)   # tudo fundo (0)
    sem[mask_cafe_tile] = SEM_CAFE                          # café (1)
    return sem


def contar_instancias_por_classe(mask_cafe_tile: np.ndarray, cfg: Config) -> int:
    return sum(1 for r in regionprops(label(mask_cafe_tile))
               if r.area >= cfg.min_area_cafe_px)

def mascara_outros_splits(bounds_tile, tile_size, outros_talhoes) -> np.ndarray:
    mascara = np.zeros((tile_size, tile_size), dtype=bool)
    minx_t, miny_t, maxx_t, maxy_t = bounds_tile
    for t in outros_talhoes:
        if "bounds" not in t: continue
        minx_r, miny_r, maxx_r, maxy_r = t["bounds"]
        if maxx_t <= minx_r or minx_t >= maxx_r or maxy_t <= miny_r or miny_t >= maxy_r: continue
        try:
            with rasterio.open(t["path"]) as src:
                fill = src.nodata if src.nodata is not None else 0
                data = src.read(window=from_bounds(minx_t, miny_t, maxx_t, maxy_t, transform=src.transform),
                                boundless=True, fill_value=fill, out_shape=(src.count, tile_size, tile_size),
                                resampling=Resampling.nearest)
                mascara |= _mascara_valida_de_data(data, src.nodata)
        except Exception:
            mascara[:] = True; return mascara
    return mascara


# ============================================================
# 8. PROCESSAMENTO DE UM SPLIT (train/val/test)
# ============================================================
def processar_split(nome: str, talhoes_split: List[dict], cfg: Config,
                    regioes_proibidas: List[dict], outros_talhoes: List[dict],
                    stats_globais: dict, global_lo: float, global_hi: float) -> None:
    print(f"\n=== {nome.upper()} : {len(talhoes_split)} talhões (MÁSCARAS SEMÂNTICAS PNG) ===")
    img_dir   = Path(cfg.saida) / "images" / nome
    msk_dir   = Path(cfg.saida) / "masks"  / nome
    map_dir   = Path(cfg.saida) / "mapping"
    debug_dir = Path(cfg.saida) / "debug"
    for d in (img_dir, msk_dir, map_dir, debug_dir):
        d.mkdir(parents=True, exist_ok=True)

    canvas, mask_valid, mask_cafe, mapping, sm_transform = criar_supermosaico(talhoes_split, cfg.ortofoto)
    print(f"  [FASE 1] supermosaico: {canvas.shape} cobertura_talhao={mask_valid.mean():.2%} cobertura_cafe={mask_cafe.mean():.2%}")

    with open(map_dir / f"{nome}.json", "w") as f: json.dump(mapping, f, indent=2)

    if cfg.salvar_debug_supermosaico:
        salvar_debug_supermosaico(canvas, mask_cafe, mask_valid, mapping, Path(cfg.saida), nome, cfg.debug_max_size_px, global_lo, global_hi)

    coords = gerar_tiles_coords(canvas.shape[1:], cfg.tile_size, cfg.overlap)
    salvos = desc_cobertura = desc_erro = desc_sem_cafe = desc_proibida = desc_outro = 0
    total_inst_split = {"jovem": 0, "adulto": 0}
    total_px_cafe_split = 0
    tiles_registry = []
    overlays_salvos = 0

    for y, x in coords:
        ts = cfg.tile_size
        mask_valid_tile = mask_valid[y:y + ts, x:x + ts]
        cobertura_tile = float(mask_valid_tile.mean())
        if cobertura_tile < cfg.min_cobertura_mask: desc_cobertura += 1; continue

        left, top = sm_transform * (x, y)
        right, bottom = sm_transform * (x + ts, y + ts)
        bounds_tile = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))

        # --- ANTI-VAZAMENTO: regiões proibidas ---
        mask_proibida = mascara_proibida(bounds_tile, ts, regioes_proibidas) if cfg.mascarar_proibida else None
        frac_proibida = float(mask_proibida.mean()) if mask_proibida is not None else 0.0
        if frac_proibida >= 1.0 or (not cfg.mascarar_proibida and tile_intersecta_proibida(bounds_tile, regioes_proibidas)):
            desc_proibida += 1; continue

        # --- ANTI-VAZAMENTO: talhões de OUTROS splits ---
        mask_outros = mascara_outros_splits(bounds_tile, ts, outros_talhoes) if cfg.mascarar_outros_splits else None
        frac_outros = float(mask_outros.mean()) if mask_outros is not None else 0.0
        if frac_outros >= 1.0 or (not cfg.mascarar_outros_splits and frac_outros > 0):
            desc_outro += 1; continue

        masks_invasoras = [m for m in (mask_proibida, mask_outros) if m is not None]
        mask_total = masks_invasoras[0].copy() if masks_invasoras else None
        if len(masks_invasoras) > 1: mask_total |= masks_invasoras[1]

        cobertura_pos = float((mask_valid_tile & ~mask_total).mean()) if mask_total is not None else cobertura_tile
        if cobertura_pos < cfg.min_cobertura_pos_mascara:
            if frac_proibida >= frac_outros: desc_proibida += 1
            else: desc_outro += 1
            continue

        try:
            imagem_raw = recortar_da_ortofoto(cfg.ortofoto, bounds_tile, ts)
        except Exception:
            desc_erro += 1; continue

        # Normalização GLOBAL 16bits -> 8bits RGB
        imagem_8bit = to_8bit_rgb_global(imagem_raw, global_lo, global_hi)

        # Zera a imagem nas regiões invasoras (defesa visual contra vazamento)
        if mask_total is not None and mask_total.any():
            imagem_8bit = imagem_8bit.copy()
            imagem_8bit[mask_total] = 0

        # Máscara de café do tile, removendo invasores
        mask_cafe_tile = mask_cafe[y:y + ts, x:x + ts].copy()
        if mask_total is not None and mask_total.any():
            mask_cafe_tile[mask_total] = False

        # Gate: tile precisa ter um mínimo de café para valer a pena
        n_cafe_px = int(mask_cafe_tile.sum())
        if n_cafe_px < cfg.min_area_cafe_px:
            desc_sem_cafe += 1; continue

        n_inst = contar_instancias_por_classe(mask_cafe_tile, cfg)

        fname = f"{nome}_{salvos:06d}"
        salvar_imagem_png(imagem_8bit, img_dir / f"{fname}.png")

        # >>> EXPORTAÇÃO SEMÂNTICA DIRETA BINÁRIA <<<
        sem = construir_mascara_semantica(mask_cafe_tile)
        Image.fromarray(sem, mode="L").save(msk_dir / f"{fname}.png")

        # Overlay de inspeção (debug)
        if cfg.salvar_debug_overlay_mascaras and overlays_salvos < cfg.n_overlays_por_split:
            salvar_overlay_inspecao(imagem_8bit, sem, debug_dir, f"overlay_{nome}_{overlays_salvos:03d}")
            overlays_salvos += 1

        talhao_nome, classe_predominante = "multiplos_mosaico", "misto"
        for m in mapping:
            if m["sm_x"] <= x + ts/2 <= m["sm_x"] + m["width"] and m["sm_y"] <= y + ts/2 <= m["sm_y"] + m["height"]:
                talhao_nome, classe_predominante = Path(m["talhao_path"]).stem, m["classe"]; break

        if classe_predominante in total_inst_split:
            total_inst_split[classe_predominante] += n_inst
        total_px_cafe_split += n_cafe_px

        tiles_registry.append({
            "fname": fname, "talhao": talhao_nome, "classe": classe_predominante,
            "bounds_minx": bounds_tile[0], "bounds_miny": bounds_tile[1],
            "bounds_maxx": bounds_tile[2], "bounds_maxy": bounds_tile[3],
            "cobertura_valida": cobertura_tile, "cobertura_pos_mascara": cobertura_pos,
            "frac_mascara_total": float(mask_total.mean()) if mask_total is not None else 0.0,
            "n_instancias": n_inst, "n_pixels_cafe": n_cafe_px,
        })
        salvos += 1

    print(f"  [FASE 2] tiles salvos: {salvos} "
          f"(desc_cobertura={desc_cobertura} desc_proibida={desc_proibida} "
          f"desc_outro={desc_outro} desc_sem_cafe={desc_sem_cafe})")

    stats_globais[nome] = {
        "talhoes": {c: sum(1 for t in talhoes_split if t["classe"] == c) for c in ("jovem", "adulto")},
        "area_m2": sum(t["area_m2"] for t in talhoes_split),
        "tiles_salvos": salvos,
        "tiles_descartados_proibida": desc_proibida,
        "pixels_cafe_total": total_px_cafe_split,
        "instancias_por_classe": {
            "jovem": total_inst_split["jovem"],
            "adulto": total_inst_split["adulto"],
            "cafe_total": total_inst_split["jovem"] + total_inst_split["adulto"]
        },
        "tiles_registry": tiles_registry,
    }

    if cfg.salvar_debug_csv_tiles:
        _salvar_csv_tiles(tiles_registry, debug_dir / f"tiles_{nome}.csv")


# ============================================================
# 9. MINERAÇÃO DE BACKGROUNDS (negativos puros)
# ============================================================
def processar_backgrounds(cfg, todos_talhoes, regioes_proibidas, stats_globais, global_lo: float, global_hi: float):
    print("\n=== BACKGROUNDS ===")
    img_dir   = Path(cfg.saida) / "images" / cfg.split_background
    msk_dir   = Path(cfg.saida) / "masks"  / cfg.split_background
    debug_dir = Path(cfg.saida) / "debug"
    for d in (img_dir, msk_dir, debug_dir): d.mkdir(parents=True, exist_ok=True)

    formas_talhoes = []
    for t in todos_talhoes:
        try:
            with rasterio.open(t["path"]) as src:
                valid = _mascara_valida_de_data(src.read(), src.nodata)
                if valid.any():
                    formas_talhoes.append({
                        "path": t["path"],
                        "bounds": t["bounds"],
                        "transform": src.transform,
                        "valid_mask": valid
                    })
        except Exception as e:
            print(f"  [warn] falha lendo {t['path']} para background: {e}")

    def _tile_toca_forma(b, formas):
        minx_t, miny_t, maxx_t, maxy_t = b
        for f in formas:
            minx_r, miny_r, maxx_r, maxy_r = f["bounds"]
            if maxx_t <= minx_r or minx_t >= maxx_r or maxy_t <= miny_r or miny_t >= maxy_r: continue
            win = from_bounds(max(minx_t, minx_r), max(miny_t, miny_r),
                              min(maxx_t, maxx_r), min(maxy_t, maxy_r), transform=f["transform"])
            col_off, row_off = max(0, int(np.floor(win.col_off))), max(0, int(np.floor(win.row_off)))
            col_end, row_end = min(f["valid_mask"].shape[1], int(np.ceil(win.col_off + win.width))), min(f["valid_mask"].shape[0], int(np.ceil(win.row_off + win.height)))
            if col_end > col_off and row_end > row_off and f["valid_mask"][row_off:row_end, col_off:col_end].any(): return True
        return False

    ts = cfg.tile_size
    salvos = desc_bate_talhao = desc_proibida = desc_invalido = 0
    tiles_registry = []

    with rasterio.open(cfg.ortofoto) as src:
        H, W, transform = src.height, src.width, src.transform
        coords = gerar_tiles_coords((H, W), ts, cfg.overlap_background)

        for y, x in coords:
            if cfg.limite_backgrounds is not None and salvos >= cfg.limite_backgrounds: break

            bounds_tile = tuple(np.concatenate((transform * (x, y), transform * (x + ts, y + ts))))
            bounds_tile = (min(bounds_tile[0], bounds_tile[2]), min(bounds_tile[1], bounds_tile[3]), max(bounds_tile[0], bounds_tile[2]), max(bounds_tile[1], bounds_tile[3]))

            if _tile_toca_forma(bounds_tile, formas_talhoes): desc_bate_talhao += 1; continue
            if tile_intersecta_proibida(bounds_tile, regioes_proibidas): desc_proibida += 1; continue

            try:
                window = Window(x, y, ts, ts)
                validade = src.read_masks(1, window=window)
                if validade.shape[0] < ts or validade.shape[1] < ts or not np.all(validade == 255): desc_invalido += 1; continue

                dados = src.read([1, 2, 3], window=window).astype(np.float32)
                if dados.shape[1] < ts or dados.shape[2] < ts or np.any(np.sum(dados, axis=0) == 0): desc_invalido += 1; continue

                fname = f"{cfg.prefixo_background}_{salvos:06d}_BG"
                
                # Normalização GLOBAL aplicada nos backgrounds
                imagem_8bit = to_8bit_rgb_global(dados, global_lo, global_hi)
                salvar_imagem_png(imagem_8bit, img_dir / f"{fname}.png")

                sem_bg = np.full((ts, ts), SEM_FUNDO, dtype=np.uint8)
                Image.fromarray(sem_bg, mode="L").save(msk_dir / f"{fname}.png")

                tiles_registry.append({"fname": fname, "talhao": "background", "bounds_minx": bounds_tile[0], "bounds_miny": bounds_tile[1], "bounds_maxx": bounds_tile[2], "bounds_maxy": bounds_tile[3], "n_instancias": 0, "n_pixels_cafe": 0})
                salvos += 1
            except Exception: pass

    tiles_por_split = {cfg.split_background: salvos, "val": 0, "test": 0}
    if salvos > 0:
        p_train, p_val, p_test = cfg.proporcao
        total_bg = len(tiles_registry)
        n_val = min(int(round(total_bg * p_val)), total_bg)
        n_test = min(int(round(total_bg * p_test)), max(0, total_bg - n_val))
        indices = np.random.default_rng(cfg.seed + 999).permutation(total_bg)
        idx_val, idx_test = set(indices[:n_val]), set(indices[n_val:n_val + n_test])

        for i, entry in enumerate(tiles_registry):
            if i in idx_val or i in idx_test:
                destino = "val" if i in idx_val else "test"
                img_dst = Path(cfg.saida) / "images" / destino
                msk_dst = Path(cfg.saida) / "masks"  / destino
                img_dst.mkdir(parents=True, exist_ok=True)
                msk_dst.mkdir(parents=True, exist_ok=True)

                fname = entry["fname"]
                src_img, src_msk = img_dir / f"{fname}.png", msk_dir / f"{fname}.png"
                if src_img.exists(): src_img.replace(img_dst / f"{fname}.png")
                if src_msk.exists(): src_msk.replace(msk_dst / f"{fname}.png")

                tiles_por_split[destino] += 1
                tiles_por_split[cfg.split_background] -= 1

    print(f"  BG salvos: {salvos} (bate_talhao={desc_bate_talhao} proibida={desc_proibida} invalido={desc_invalido})  distribuição={tiles_por_split}")

    stats_globais["background"] = {
        "talhoes": {"jovem": 0, "adulto": 0},
        "tiles_salvos": salvos,
        "tiles_descartados_proibida": desc_proibida,
        "pixels_cafe_total": 0,
        "instancias_por_classe": {"jovem": 0, "adulto": 0, "cafe_total": 0},
        "tiles_registry": tiles_registry,
        "tiles_por_split": tiles_por_split
    }
    if cfg.salvar_debug_csv_tiles: _salvar_csv_tiles(tiles_registry, debug_dir / "tiles_background.csv")


# ============================================================
# 10. DEBUGS: GRÁFICOS E MAPAS
# ============================================================
def salvar_overlay_inspecao(imagem_8bit: np.ndarray, sem: np.ndarray, debug_dir: Path, nome: str):
    """Overlay RGB + máscara semântica binária: verde=café."""
    rgb = imagem_8bit.astype(np.float32)
    over = rgb.copy()
    cafe = (sem == SEM_CAFE)
    over[cafe] = 0.55 * over[cafe] + 0.45 * np.array([60, 220, 60], dtype=np.float32)
    out = over.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(imagem_8bit); axes[0].set_title("imagem"); axes[0].axis("off")
    axes[1].imshow(out); axes[1].set_title("verde=café (Binário)"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(debug_dir / f"{nome}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

def salvar_debug_supermosaico(canvas, mask_cafe, mask_valida, mapping, saida_dir, nome_split, max_size_px, global_lo, global_hi):
    rgb = to_8bit_rgb_global(canvas, global_lo, global_hi) * mask_valida[..., None].astype(np.uint8)
    overlay = rgb.astype(np.float32)
    overlay[mask_cafe] = ((1 - 0.45) * overlay[mask_cafe] + 0.45 * np.array([100, 230, 100], dtype=np.float32))
    overlay_u8 = overlay.clip(0, 255).astype(np.uint8)

    for m in mapping:
        x0, y0 = m["sm_x"], m["sm_y"]
        x1, y1 = min(x0 + m["width"] - 1, overlay_u8.shape[1] - 1), min(y0 + m["height"] - 1, overlay_u8.shape[0] - 1)
        overlay_u8[y0, x0:x1+1] = overlay_u8[y1, x0:x1+1] = [220, 60, 60]
        overlay_u8[y0:y1+1, x0] = overlay_u8[y0:y1+1, x1] = [220, 60, 60]

    img = Image.fromarray(overlay_u8)
    if max(img.size) > max_size_px:
        scale = max_size_px / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    out_path = saida_dir / "debug" / f"supermosaico_{nome_split}.png"
    img.save(out_path, optimize=True)

def _salvar_csv_tiles(tiles_registry, out_path):
    if not tiles_registry: return
    campos = ["fname", "talhao", "classe", "bounds_minx", "bounds_miny", "bounds_maxx", "bounds_maxy",
              "cobertura_valida", "cobertura_pos_mascara", "frac_mascara_total", "n_instancias", "n_pixels_cafe"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(tiles_registry)

def plotar_distribuicao_areas(stats: dict, proporcao: Tuple[float, float, float], saida_path: Path) -> None:
    splits = ["train", "val", "test"]
    presentes = [s for s in splits if s in stats]
    if not presentes: return
    p_map = dict(zip(splits, proporcao))

    areas_ha = [stats[s].get("area_m2", 0) / 1e4 for s in presentes]
    area_total = sum(areas_ha) if sum(areas_ha) > 0 else 1
    perc = [a / area_total * 100 for a in areas_ha]
    alvos = [p_map[s] * 100 for s in presentes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(presentes))

    axes[0].bar(x, areas_ha, color=[CORES_SPLIT[s] for s in presentes])
    for i, a in enumerate(areas_ha):
        axes[0].text(i, a, f"{a:.2f} ha", ha="center", va="bottom")
    axes[0].set_xticks(x); axes[0].set_xticklabels(presentes)
    axes[0].set_ylabel("área (ha)")
    axes[0].set_title("Área absoluta por split")

    largura = 0.35
    axes[1].bar(x - largura / 2, perc, largura, label="real", color=[CORES_SPLIT[s] for s in presentes])
    axes[1].bar(x + largura / 2, alvos, largura, label="alvo", color="#888888", alpha=0.6)
    for i, (p, a) in enumerate(zip(perc, alvos)):
        axes[1].text(i - largura / 2, p, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
        axes[1].text(i + largura / 2, a, f"{a:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(presentes)
    axes[1].set_ylabel("% da área total")
    axes[1].set_title("Distribuição real vs alvo")
    axes[1].legend()

    fig.suptitle("Distribuição de área entre splits", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(saida_path / "distribuicao_areas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plotar_distribuicao(stats: dict, saida_path: Path) -> None:
    splits = ["train", "val", "test"]
    presentes = [s for s in splits if s in stats]
    x = np.arange(len(presentes))
    largura = 0.35
    jovem_col, adulto_col = "#7CB342", "#5D4037"

    talhoes_j = [stats[s]["talhoes"]["jovem"] for s in presentes]
    talhoes_a = [stats[s]["talhoes"]["adulto"] for s in presentes]
    inst_j = [stats[s]["instancias_por_classe"]["jovem"] for s in presentes]
    inst_a = [stats[s]["instancias_por_classe"]["adulto"] for s in presentes]
    tiles = [stats[s]["tiles_salvos"] for s in presentes]
    proibida = [stats[s]["tiles_descartados_proibida"] for s in presentes]
    px_cafe = [stats[s].get("pixels_cafe_total", 0) for s in presentes]

    fig, axes = plt.subplots(1, 5, figsize=(27, 5))

    # 1. Talhões
    axes[0].bar(x - largura / 2, talhoes_j, largura, label="jovem", color=jovem_col)
    axes[0].bar(x + largura / 2, talhoes_a, largura, label="adulto", color=adulto_col)
    for i, (vj, va) in enumerate(zip(talhoes_j, talhoes_a)):
        axes[0].text(i - largura / 2, vj, str(vj), ha="center", va="bottom")
        axes[0].text(i + largura / 2, va, str(va), ha="center", va="bottom")
    axes[0].set_xticks(x); axes[0].set_xticklabels(presentes)
    axes[0].set_ylabel("nº de talhões")
    axes[0].set_title("Talhões por classe e split")
    axes[0].legend()

    # 2. Componentes conexas
    axes[1].bar(x - largura / 2, inst_j, largura, label="originado de jovem", color=jovem_col)
    axes[1].bar(x + largura / 2, inst_a, largura, label="originado de adulto", color=adulto_col)
    for i, (vj, va) in enumerate(zip(inst_j, inst_a)):
        axes[1].text(i - largura / 2, vj, str(vj), ha="center", va="bottom")
        axes[1].text(i + largura / 2, va, str(va), ha="center", va="bottom")
    axes[1].set_xticks(x); axes[1].set_xticklabels(presentes)
    axes[1].set_ylabel("nº de componentes de café")
    axes[1].set_title("Componentes conexas de café")
    axes[1].legend()

    # 3. Tiles (café + BG)
    bgs_por_split = {s: 0 for s in presentes}
    if "background" in stats:
        bgs_dict = stats["background"].get("tiles_por_split", {})
        for s in presentes:
            bgs_por_split[s] = int(bgs_dict.get(s, 0))
    bgs = [bgs_por_split[s] for s in presentes]

    cafe_col, bg_col = "#1976D2", "#9E9E9E"
    axes[2].bar(x, tiles, color=cafe_col, label="tiles com café")
    axes[2].bar(x, bgs, bottom=tiles, color=bg_col, label="tiles BG (fundo)")
    for i, (vc, vb) in enumerate(zip(tiles, bgs)):
        if vc > 0:
            axes[2].text(i, vc / 2, str(vc), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if vb > 0:
            axes[2].text(i, vc + vb / 2, str(vb), ha="center", va="center", color="black", fontsize=9, fontweight="bold")
        total = vc + vb
        axes[2].text(i, total, f"= {total}", ha="center", va="bottom", fontsize=9)
    axes[2].set_xticks(x); axes[2].set_xticklabels(presentes)
    axes[2].set_ylabel("nº de tiles (PNG)")
    axes[2].set_title("Tiles por split (café + BG)")
    axes[2].legend(loc="upper right", fontsize=8)

    # 4. Pixels de café
    axes[3].bar(x, [p / 1e6 for p in px_cafe], color="#2E7D32")
    for i, p in enumerate(px_cafe):
        axes[3].text(i, p / 1e6, f"{p/1e6:.2f}M", ha="center", va="bottom", fontsize=9)
    axes[3].set_xticks(x); axes[3].set_xticklabels(presentes)
    axes[3].set_ylabel("pixels de café (milhões)")
    axes[3].set_title("Volume de pixels classe 'café'")

    # 5. Proibida
    axes[4].bar(x, proibida, color="#D32F2F")
    for i, v in enumerate(proibida):
        axes[4].text(i, v, str(v), ha="center", va="bottom")
    axes[4].set_xticks(x); axes[4].set_xticklabels(presentes)
    axes[4].set_ylabel("nº de tiles descartados")
    axes[4].set_title("Tiles barrados por região proibida")

    fig.suptitle("Distribuição do dataset semântico", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(saida_path / "distribuicao_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def _contornos_geo_da_mascara(mask: np.ndarray, transform: Affine, downsample: int = 1) -> List[np.ndarray]:
    if downsample > 1:
        mask_ds = mask[::downsample, ::downsample]
        trans_ds = Affine(transform.a * downsample, transform.b, transform.c,
                          transform.d, transform.e * downsample, transform.f)
    else:
        mask_ds, trans_ds = mask, transform

    padded = np.pad(mask_ds.astype(np.uint8), 1, mode="constant", constant_values=0)
    contornos_px = find_contours(padded, level=0.5)
    contornos_geo = []
    for c in contornos_px:
        rows, cols = c[:, 0] - 1, c[:, 1] - 1
        xs = trans_ds.a * cols + trans_ds.b * rows + trans_ds.c
        ys = trans_ds.d * cols + trans_ds.e * rows + trans_ds.f
        pts = np.column_stack([xs, ys])
        if len(pts) >= 3: contornos_geo.append(pts)
    return contornos_geo

def plotar_mapa_splits(splits_dict: Dict[str, List[dict]], regioes_proibidas: List[dict],
                       saida_path: Path, downsample: int = 4, anotar_nome: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))

    for nome_split, talhoes in splits_dict.items():
        cor = CORES_SPLIT[nome_split]
        for t in talhoes:
            try:
                with rasterio.open(t["path"]) as src:
                    m = _mascara_valida_de_data(src.read(), src.nodata)
                    if not m.any(): continue
                    contornos = _contornos_geo_da_mascara(m, src.transform, downsample)
                    hatch = HATCH_CLASSE.get(t["classe"], "")
                    for pts in contornos:
                        ax.fill(pts[:, 0], pts[:, 1], facecolor=cor, edgecolor=cor, alpha=0.55, hatch=hatch, linewidth=1.2)
                    if anotar_nome and contornos:
                        all_pts = np.vstack(contornos)
                        cx, cy = all_pts[:, 0].mean(), all_pts[:, 1].mean()
                        ax.annotate(Path(t["path"]).stem, xy=(cx, cy), fontsize=7, ha="center", va="center", color="black",
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
            except Exception: pass

    for r in regioes_proibidas:
        try:
            contornos = _contornos_geo_da_mascara(r["valid_mask"], r["transform"], downsample)
            for pts in contornos:
                ax.fill(pts[:, 0], pts[:, 1], facecolor="#D32F2F", edgecolor="#B71C1C", alpha=0.50, hatch="xxxx", linewidth=1.5)
            if anotar_nome and contornos:
                all_pts = np.vstack(contornos)
                cx, cy = all_pts[:, 0].mean(), all_pts[:, 1].mean()
                ax.annotate(f"[X] {Path(r['path']).stem}", xy=(cx, cy), fontsize=7, ha="center", va="center", color="#5b0000",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
        except Exception: pass

    legenda = []
    for split_name in ["train", "val", "test"]:
        cor = CORES_SPLIT[split_name]
        nome_pt = NOME_SPLIT_PT[split_name]
        legenda.append(Patch(facecolor=cor, edgecolor=cor, alpha=0.55, label=f"{nome_pt} — jovem"))
        legenda.append(Patch(facecolor=cor, edgecolor=cor, alpha=0.55, hatch="////", label=f"{nome_pt} — adulto"))
    legenda.append(Patch(facecolor="#D32F2F", edgecolor="#B71C1C", alpha=0.50, hatch="xxxx", label="região proibida"))

    ax.legend(handles=legenda, loc="best", fontsize=9, framealpha=0.95, ncol=2)
    ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
    ax.set_title("Mapa de talhões — distribuição geográfica (Semântico)\n"
                 "Cor = split · Padrão = classe (sólido=jovem, ////=adulto)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout(); plt.savefig(saida_path / "mapa_splits.png", dpi=180, bbox_inches="tight"); plt.close(fig)

def plotar_mapa_tiles(stats: dict, splits_dict: Dict[str, List[dict]], regioes_proibidas: List[dict], saida_path: Path, downsample: int = 4) -> None:
    fig, ax = plt.subplots(figsize=(16, 12))

    for nome_split, talhoes in splits_dict.items():
        cor = CORES_SPLIT[nome_split]
        for t in talhoes:
            try:
                with rasterio.open(t["path"]) as src:
                    m = _mascara_valida_de_data(src.read(), src.nodata)
                    if not m.any(): continue
                    for pts in _contornos_geo_da_mascara(m, src.transform, downsample):
                        ax.fill(pts[:, 0], pts[:, 1], facecolor=cor, alpha=0.15, edgecolor=cor, linewidth=0.6)
            except Exception: pass

    for nome_split in ["train", "val", "test"]:
        if nome_split not in stats: continue
        cor = CORES_SPLIT[nome_split]
        tiles_reg = stats[nome_split].get("tiles_registry", [])
        patches = [Rectangle((r["bounds_minx"], r["bounds_miny"]), r["bounds_maxx"] - r["bounds_minx"], r["bounds_maxy"] - r["bounds_miny"]) for r in tiles_reg]
        if patches: ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=cor, linewidth=0.3, alpha=0.55))

    if "background" in stats:
        tiles_reg_bg = stats["background"].get("tiles_registry", [])
        patches_bg = [Rectangle((r["bounds_minx"], r["bounds_miny"]), r["bounds_maxx"] - r["bounds_minx"], r["bounds_maxy"] - r["bounds_miny"]) for r in tiles_reg_bg]
        if patches_bg: ax.add_collection(PatchCollection(patches_bg, facecolor="#9E9E9E", edgecolor="#616161", linewidth=0.2, alpha=0.35))

    for r in regioes_proibidas:
        try:
            for pts in _contornos_geo_da_mascara(r["valid_mask"], r["transform"], downsample):
                ax.fill(pts[:, 0], pts[:, 1], facecolor="#D32F2F", alpha=0.30, edgecolor="#B71C1C", hatch="xxxx", linewidth=1.0)
        except Exception: pass

    legenda = [
        Patch(facecolor=CORES_SPLIT["train"], alpha=0.5, edgecolor=CORES_SPLIT["train"], label="tiles treino"),
        Patch(facecolor=CORES_SPLIT["val"], alpha=0.5, edgecolor=CORES_SPLIT["val"], label="tiles val"),
        Patch(facecolor=CORES_SPLIT["test"], alpha=0.5, edgecolor=CORES_SPLIT["test"], label="tiles teste"),
        Patch(facecolor="#9E9E9E", alpha=0.35, edgecolor="#616161", label="tiles background (_BG)"),
        Patch(facecolor="#D32F2F", alpha=0.30, hatch="xxxx", label="região proibida"),
    ]
    ax.legend(handles=legenda, loc="best", fontsize=9, framealpha=0.95)
    ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
    ax.set_title("Mapa de recortes da janela deslizante\nCada retângulo é um tile efetivamente exportado")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout(); plt.savefig(saida_path / "mapa_tiles.png", dpi=180, bbox_inches="tight"); plt.close(fig)


# ============================================================
# YAML  (modo semântico PNG — chave masks_dir + 2 classes)
# ============================================================
def escrever_yaml(cfg: Config):
    with open(Path(cfg.saida) / "data.yaml", "w") as f:
        f.write(
            f"path: {cfg.saida}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n\n"
            f"masks_dir: masks\n\n"
            f"names:\n"
            f"  0: fundo\n"
            f"  1: cafe\n"
        )


# ============================================================
# MAIN
# ============================================================
def main(cfg: Config):
    Path(cfg.saida).mkdir(parents=True, exist_ok=True)
    pastas = {"jovem": cfg.pasta_jovens, "adulto": cfg.pasta_adultos}
    mascaras = {"jovem": cfg.mascaras_jovens, "adulto": cfg.mascaras_adultos}

    talhoes = listar_talhoes(pastas, mascaras, sufixos_mascara=cfg.sufixos_mascara)
    if len(talhoes) < 3: raise RuntimeError("Poucos talhões encontrados.")

    diagnosticar_alinhamento(cfg.ortofoto, talhoes)
    regioes_proibidas = carregar_regioes_proibidas(cfg.regiao_proibida)

    splits = split_talhoes_por_area(talhoes, cfg.proporcao, cfg.seed)

    if cfg.salvar_debug_mapa_splits:
        plotar_mapa_splits(splits, regioes_proibidas, Path(cfg.saida),
                           downsample=cfg.mapa_splits_downsample,
                           anotar_nome=cfg.mapa_splits_anotar_nome)

    # =========================================================================
    # CÁLCULO DOS LIMITES RADIOMÉTRICOS GLOBAIS (BASEADO APENAS NO TREINO)
    # =========================================================================
    talhoes_treino_objs = splits.get("train", [])
    if not talhoes_treino_objs:
        raise RuntimeError("Nenhum talhão de treino foi gerado no split.")
        
    print(f"\n=== CALCULANDO LIMITES RADIOMÉTRICOS GLOBAIS (Apenas Treino - Sample: {cfg.sample_to_calc_norm:.0%}) ===")
    global_lo, global_hi = calcular_limites_globais_rgb(
        cfg.ortofoto, 
        talhoes_treino_objs,
        sample_frac=cfg.sample_to_calc_norm 
    )
    print(f"  Limites globais calculados -> LO: {global_lo:.1f}, HI: {global_hi:.1f}")

    # =========================================================================
    # PROCESSAMENTO DOS SPLITS E BACKGROUND
    # =========================================================================
    stats = {}
    for nome, grupo in splits.items():
        if not grupo: continue
        outros_talhoes = [t for n, g in splits.items() if n != nome for t in g]
        processar_split(nome, grupo, cfg, regioes_proibidas, outros_talhoes, stats, global_lo, global_hi)

    if cfg.salvar_backgrounds:
        processar_backgrounds(cfg, talhoes, regioes_proibidas, stats, global_lo, global_hi)

    escrever_yaml(cfg)

    if cfg.salvar_debug_distribuicao_areas:
        plotar_distribuicao_areas(stats, cfg.proporcao, Path(cfg.saida))
    if cfg.salvar_debug_distribuicao_instancias:
        plotar_distribuicao(stats, Path(cfg.saida))
    if cfg.salvar_debug_tiles_mapa:
        plotar_mapa_tiles(stats, splits, regioes_proibidas, Path(cfg.saida))

    print("\n=== RESUMO (SEMÂNTICO PNG) ===")
    for nome, s in stats.items():
        linha_extra = ""
        if nome == "background":
            linha_extra = f"  distribuição={s.get('tiles_por_split', {})}"
        print(f"  {nome}: talhões={s['talhoes']} "
              f"área={s.get('area_m2', 0)/1e4:.2f} ha "
              f"tiles={s['tiles_salvos']} "
              f"px_cafe={s.get('pixels_cafe_total', 0)/1e6:.2f}M "
              f"componentes={s['instancias_por_classe']} "
              f"barrados_proib={s['tiles_descartados_proibida']}{linha_extra}")

    print(f"\ndata.yaml em: {Path(cfg.saida) / 'data.yaml'}")
    print("Pipeline Finalizado (modo semântico PNG BINÁRIO: 0=fundo, 1=cafe).")


if __name__ == "__main__":
    PASTA_JOVENS  = "talhoes_jovens_tif"
    PASTA_ADULTOS = "talhoes_adultos_tif"
    ORTOFOTO      = "Paula_Candido.tif"

    MASCARAS_JOVENS  = "antes/mascaras_jovens_cluster_manual"
    MASCARAS_ADULTOS = "antes/mascaras_adultos_cluster_manual"

    REGIAO_PROIBIDA = "regiao_proibida/"
    SAIDA = "dataset_cnn_1024"

    cfg = Config(
        pasta_jovens=PASTA_JOVENS,
        pasta_adultos=PASTA_ADULTOS,
        ortofoto=ORTOFOTO,
        mascaras_jovens=MASCARAS_JOVENS,
        mascaras_adultos=MASCARAS_ADULTOS,
        regiao_proibida=REGIAO_PROIBIDA,
        saida=SAIDA,
        tile_size=1024,
        overlap=0.80,
        mascarar_outros_splits=True,   
        mascarar_proibida=True,        
        min_cobertura_pos_mascara=0.15,
        
        # Norm
        sample_to_calc_norm=1.0,
        
        salvar_backgrounds=True,
        salvar_debug_supermosaico=True,
        salvar_debug_tiles_mapa=True,
        salvar_debug_distribuicao_areas=True,
        salvar_debug_csv_tiles=True,
        salvar_debug_distribuicao_instancias=True,
        salvar_debug_mapa_splits=True,
        salvar_debug_overlay_mascaras=True,
    )
    main(cfg)