#!/usr/bin/env python3
# coding: utf-8
"""
watermark.py
- DCT-based blind watermark embed & extract for color images (Y channel)
- Robustness tests: flip, translate, crop, contrast, noise, JPEG
Author: (you)
Usage:
    python watermark.py --cover cover.jpg --watermark wm.png
"""

import cv2
import numpy as np
import argparse
import os
from scipy.fftpack import dct, idct
import math

# ---------------------------
# Utilities
# ---------------------------
def to_gray_uint8(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b)**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# ---------------------------
# DCT helpers (8x8 blocks)
# ---------------------------
def block_process(img, block_size, func):
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            block = img[by:by+block_size, bx:bx+block_size]
            if block.shape[0] != block_size or block.shape[1] != block_size:
                # skip partial block (we force image dims multiple of block_size)
                continue
            out_block = func(block)
            out[by:by+block_size, bx:bx+block_size] = out_block
    return out

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ---------------------------
# Watermark embedding/extraction (blind)
# Algorithm:
# - Convert image to YCrCb, take Y channel (float)
# - Resize watermark (binary) to (m, n) based on number of 8x8 blocks
# - For each 8x8 block, pick two mid-frequency coeff positions (p1,p2), ensure they are not DC and not very high freq
# - If watermark bit == 1: make coeff(p1) > coeff(p2) + alpha
#   else: coeff(p1) < coeff(p2) - alpha
# - Reconstruct Y with IDCT blocks and merge back to BGR
# Extraction:
# - Same block grid, compute DCT, check sign of coeff(p1)-coeff(p2) to infer bit
# This is blind (doesn't require original cover).
# ---------------------------

# choose mid-frequency coefficient positions within 8x8 (zero-based)
# avoid (0,0). Common choice: (2,1) & (1,2) or (3,2)&(2,3)
P1 = (2, 1)
P2 = (1, 2)
BLOCK = 8

def embed_watermark(cover_bgr, watermark_bin, alpha=10.0):
    """
    cover_bgr: BGR uint8 image
    watermark_bin: binary numpy array (values 0/1) to embed
    alpha: embedding strength
    returns: watermarked_bgr (uint8)
    """
    # ensure cover dims divisible by BLOCK
    h, w = cover_bgr.shape[:2]
    h2 = (h // BLOCK) * BLOCK
    w2 = (w // BLOCK) * BLOCK
    cover = cover_bgr[:h2, :w2].copy()

    # convert to YCrCb and use Y
    ycrcb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]

    # number of blocks
    n_by = h2 // BLOCK
    n_bx = w2 // BLOCK
    total_blocks = n_by * n_bx

    # resize watermark to fit block grid
    wm_h, wm_w = watermark_bin.shape
    # target size = (n_by, n_bx)
    wm_resized = cv2.resize(watermark_bin.astype(np.uint8) * 255, (n_bx, n_by), interpolation=cv2.INTER_NEAREST)
    wm_bits = (wm_resized > 127).astype(np.uint8)

    # process blocks
    def proc(block):
        B = block.astype(np.float32)
        C = dct2(B)
        b = wm_bits[proc.by, proc.bx]
        # positions
        p1 = P1; p2 = P2
        c1 = C[p1]
        c2 = C[p2]
        if b == 1:
            if c1 <= c2 + alpha:
                # increase c1 or decrease c2
                C[p1] = c2 + alpha + 1.0
        else:
            if c1 >= c2 - alpha:
                C[p1] = c2 - alpha - 1.0
        out = idct2(C)
        return out

    # need to provide indices inside proc
    outY = np.zeros_like(Y, dtype=np.float32)
    for by in range(n_by):
        for bx in range(n_bx):
            proc.by = by
            proc.bx = bx
            y_block = Y[by*BLOCK:(by+1)*BLOCK, bx*BLOCK:(bx+1)*BLOCK]
            out_block = proc(y_block)
            outY[by*BLOCK:(by+1)*BLOCK, bx*BLOCK:(bx+1)*BLOCK] = out_block

    # clip and combine
    ycrcb[:, :, 0] = np.clip(outY, 0, 255)
    out_bgr = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out_bgr, wm_bits

def extract_watermark(watermarked_bgr, wm_shape):
    """
    watermarked_bgr: BGR uint8 image
    wm_shape: (h_blocks, w_blocks) i.e. target watermark grid (n_by, n_bx)
    returns: extracted_bits (array of 0/1)
    """
    h, w = watermarked_bgr.shape[:2]
    h2 = (h // BLOCK) * BLOCK
    w2 = (w // BLOCK) * BLOCK
    img = watermarked_bgr[:h2, :w2].copy()
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]

    n_by = h2 // BLOCK
    n_bx = w2 // BLOCK

    bits = np.zeros((n_by, n_bx), dtype=np.uint8)
    for by in range(n_by):
        for bx in range(n_bx):
            B = Y[by*BLOCK:(by+1)*BLOCK, bx*BLOCK:(bx+1)*BLOCK]
            C = dct2(B)
            c1 = C[P1]
            c2 = C[P2]
            bits[by, bx] = 1 if (c1 - c2) > 0 else 0

    # resize bits to requested wm_shape
    extracted = cv2.resize(bits.astype(np.uint8) * 255, (wm_shape[1], wm_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (extracted > 127).astype(np.uint8)

# ---------------------------
# Attacks
# ---------------------------
def attack_flip(img, mode='horizontal'):
    if mode == 'horizontal':
        return cv2.flip(img, 1)
    elif mode == 'vertical':
        return cv2.flip(img, 0)
    else:
        return img.copy()

def attack_translate(img, tx=10, ty=5):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def attack_crop(img, crop_ratio=0.8):
    # center crop then pad back to original size (so extraction grid aligns)
    h, w = img.shape[:2]
    ch = int(h * crop_ratio)
    cw = int(w * crop_ratio)
    y0 = (h - ch)//2
    x0 = (w - cw)//2
    crop = img[y0:y0+ch, x0:x0+cw]
    # pad to original (pad reflect)
    top = y0; left = x0; bottom = h - (y0+ch); right = w - (x0+cw)
    return cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REFLECT)

def attack_contrast(img, alpha=1.2, beta=0):
    # new = alpha*img + beta
    out = img.astype(np.float32) * alpha + beta
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def attack_noise(img, sigma=5.0):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def attack_jpeg(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return decimg

# ---------------------------
# Evaluation harness
# ---------------------------
def bit_accuracy(orig_bits, extracted_bits):
    # resize orig_bits to extracted size to compare
    if orig_bits.shape != extracted_bits.shape:
        # assume orig_bits was original watermark in pixels (HxW)
        # convert to 0/1 arrays with same size
        pass
    total = orig_bits.size
    same = np.sum(orig_bits == extracted_bits)
    return float(same) / float(total)

def run_demo(cover_path, wm_path, out_dir='out', alpha=10.0):
    os.makedirs(out_dir, exist_ok=True)
    cover = cv2.imread(cover_path)
    if cover is None:
        raise ValueError("Cannot read cover image")
    wm_img = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    if wm_img is None:
        raise ValueError("Cannot read watermark image")
    # binarize watermark
    _, wm_bin = cv2.threshold(wm_img, 127, 1, cv2.THRESH_BINARY)
    # embed
    watermarked, wm_grid = embed_watermark(cover, wm_bin, alpha=alpha)
    cv2.imwrite(os.path.join(out_dir, 'watermarked.png'), watermarked)
    print("Saved watermarked.png")
    # extract from clean
    extracted_clean = extract_watermark(watermarked, wm_bin.shape)
    cv2.imwrite(os.path.join(out_dir, 'extracted_clean.png'), extracted_clean*255)
    acc_clean = bit_accuracy(wm_bin, extracted_clean)
    print(f"Extraction accuracy (clean): {acc_clean*100:.2f}%  PSNR cover->watermarked: {psnr(cover[:watermarked.shape[0],:watermarked.shape[1]], watermarked):.2f} dB")

    # perform attacks
    attacks = [
        ('flip_h', lambda img: attack_flip(img, 'horizontal')),
        ('flip_v', lambda img: attack_flip(img, 'vertical')),
        ('translate', lambda img: attack_translate(img, tx=6, ty=4)),
        ('crop80', lambda img: attack_crop(img, 0.8)),
        ('contrast_high', lambda img: attack_contrast(img, alpha=1.3)),
        ('contrast_low', lambda img: attack_contrast(img, alpha=0.7)),
        ('noise_sigma5', lambda img: attack_noise(img, sigma=5.0)),
        ('jpeg_q70', lambda img: attack_jpeg(img, quality=70)),
        ('jpeg_q50', lambda img: attack_jpeg(img, quality=50)),
    ]
    results = []
    for name, fn in attacks:
        attacked = fn(watermarked)
        path = os.path.join(out_dir, f'attacked_{name}.png')
        cv2.imwrite(path, attacked)
        extracted = extract_watermark(attacked, wm_bin.shape)
        cv2.imwrite(os.path.join(out_dir, f'extracted_{name}.png'), extracted*255)
        acc = bit_accuracy(wm_bin, extracted)
        print(f"Attack {name}: accuracy = {acc*100:.2f}%  saved attacked image & extracted map.")
        results.append((name, acc))
    # summary
    print("\nSummary:")
    for name, acc in results:
        print(f"{name:12s} : {acc*100:6.2f}%")
    return results

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SM4-like watermark demo (DCT 8x8 blind)")
    parser.add_argument('--cover', type=str, required=True, help="cover image path")
    parser.add_argument('--watermark', type=str, required=True, help="watermark binary image path (prefer BW)")
    parser.add_argument('--out', type=str, default='out', help="output directory")
    parser.add_argument('--alpha', type=float, default=10.0, help="embedding strength (float, default 10.0)")
    args = parser.parse_args()
    run_demo(args.cover, args.watermark, out_dir=args.out, alpha=args.alpha)
