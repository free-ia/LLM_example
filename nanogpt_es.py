"""
============================================================================
 nanogpt_es.py - GPT estilo nanoGPT, adaptado al corpus Spanish Billion Words
============================================================================

Este archivo es el "siguiente paso" natural despues de microgpt.py:
sigue siendo EL MISMO algoritmo (la formula que vimos en EXPLICACION.md
seccion 11), pero implementado con PyTorch para que pueda procesar texto
real en cantidades serias (gigabytes), aprovechando GPU si esta disponible.

Esta inspirado en `nanoGPT` de Andrej Karpathy
    https://github.com/karpathy/nanoGPT
adaptado para entrenar sobre el "Spanish Billion Words Corpus" (SBWC) de
Cristian Cardellino, Universidad Nacional de Cordoba:
    http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/

----------------------------------------------------------------------------
 DIFERENCIAS RESPECTO A microgpt.py
----------------------------------------------------------------------------
                          microgpt.py            nanogpt_es.py
  Dependencias            ninguna                torch + numpy
  Operaciones             escalares (1 a 1)      tensores (paralelas)
  Hardware                CPU, 1 hilo            GPU si hay, sino CPU
  Datos de entrenamiento  ~30 KB de nombres      ~10 GB de espanol
  Parametros              ~4 000                  ~10 000 000
  Tamanio de contexto     16 caracteres          256 caracteres
  Velocidad por step      ~segundos              ~milisegundos
  Normalizacion           RMSNorm                LayerNorm  (estandar GPT-2)
  No-linealidad           ReLU                   GELU       (estandar GPT-2)
  Atencion                manual                 Flash Attention (PyTorch 2.0)
  Algoritmo conceptual    IDENTICO               IDENTICO

Es decir: si entiendes microgpt.py, entiendes este. Lo unico que cambia es
la implementacion para que sea practica a escala real.

----------------------------------------------------------------------------
 USO (3 modos)
----------------------------------------------------------------------------

  1) PREPARAR los datos (tokeniza el corpus y crea train.bin / val.bin):

       python nanogpt_es.py prepare --corpus_dir spanish_billion_words

     Para hacer una prueba rapida con solo unos pocos ficheros:
       python nanogpt_es.py prepare --corpus_dir spanish_billion_words --max_files 2

  2) ENTRENAR el modelo:

       python nanogpt_es.py train

     (Lee data/train.bin, data/val.bin y data/vocab.pkl)

  3) GENERAR texto con el modelo entrenado:

       python nanogpt_es.py sample --prompt "Habia una vez"

----------------------------------------------------------------------------
 LAS 5 PIEZAS (las mismas que en microgpt.py)
----------------------------------------------------------------------------
  1. DATASET     -> spanish_billion_words/*.txt              (cmd_prepare)
  2. TOKENIZER   -> CharTokenizer (vocabulario de caracteres) (build_vocab)
  3. MODELO      -> clase GPT (embeddings + N x Block + lm_head)
  4. ENTRENAMIENTO -> bucle con AdamW                          (cmd_train)
  5. INFERENCIA  -> GPT.generate() con temperatura y top-k    (cmd_sample)
============================================================================
"""

import os
import sys
import math
import time
import json
import pickle
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# 1) CONFIGURACION GLOBAL
# ============================================================================
# Toda la configuracion en un sitio. En proyectos serios suele estar en un
# fichero YAML/JSON aparte; aqui lo dejamos inline para mayor legibilidad.
#
# RECORDATORIO de la formula (seccion 11 de EXPLICACION.md):
#     N_params  ~=  12 * n_layer * n_embd^2   (mas embeddings)
#
# Con los valores por defecto:
#     12 * 6 * 384^2  =  10 616 832  ->  ~10.6M parametros
# ----------------------------------------------------------------------------

# ---- Directorios ----
DATA_DIR  = Path("data")              # binarios tokenizados (train.bin, val.bin)
CKPT_DIR  = Path("checkpoints")       # checkpoints del modelo durante entrenamiento

# ---- Hiperparametros del modelo (afectan al numero de parametros) ----
N_LAYER    = 6      # profundidad: numero de bloques transformer apilados
N_HEAD     = 6      # cabezas de atencion (n_embd debe ser divisible por n_head)
N_EMBD     = 384    # anchura: dimension de los embeddings y del residual stream
BLOCK_SIZE = 256    # contexto maximo: caracteres que ve el modelo de una vez
DROPOUT    = 0.0    # regularizacion (0 = sin dropout). Subir si hay overfitting.
BIAS       = False  # sin biases en LayerNorm/Linear (estilo Llama, mas simple)

# ---- Hiperparametros del entrenamiento ----
BATCH_SIZE       = 32       # secuencias procesadas en paralelo por step
GRAD_ACCUM_STEPS = 4        # acumular gradientes antes de actualizar.
                            # Batch efectivo = BATCH_SIZE * GRAD_ACCUM_STEPS
                            # Permite simular batches grandes con poca VRAM.
MAX_ITERS        = 5000     # numero total de iteraciones de entrenamiento
EVAL_INTERVAL    = 250      # cada cuantas iteraciones validamos
EVAL_ITERS       = 50       # cuantos batches usamos para estimar val loss

# Learning rate schedule: warmup + cosine decay (estandar en LLMs modernos)
LEARNING_RATE    = 3e-4     # LR maximo
WARMUP_ITERS     = 100      # iteraciones de calentamiento (LR sube linealmente)
LR_DECAY_ITERS   = 5000     # cuando termina el cosine decay
MIN_LR           = 3e-5     # LR minimo al final del decay (10% de LR maximo)

# Optimizador AdamW
WEIGHT_DECAY     = 0.1      # regularizacion L2 (solo en parametros 2D)
BETA1, BETA2     = 0.9, 0.95
GRAD_CLIP        = 1.0      # clipping de la norma del gradiente (estabilidad)

# ---- Hardware: deteccion automatica ----
# CUDA (GPU NVIDIA) > MPS (Apple Silicon) > CPU
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
# bfloat16 acelera mucho en GPUs modernas (Ampere+) sin perdida apreciable.
# En CPU usamos float32 (no hay aceleracion de half precision).
DTYPE = (
    "bfloat16" if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
    else "float16" if DEVICE == "cuda"
    else "float32"
)

# ---- Reproducibilidad ----
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# 2) PREPARACION DE DATOS
# ============================================================================
# Antes de entrenar necesitamos transformar el corpus crudo (gigabytes de
# texto plano) en algo eficiente de leer durante el entrenamiento. El plan:
#
#   1) Recorrer todos los .txt del corpus
#   2) Construir el vocabulario (caracteres unicos que superen un umbral)
#   3) Tokenizar TODO el corpus (cada caracter -> id entero)
#   4) Guardar el resultado como un array binario en disco (uint16)
#   5) Reservar un trozo para validacion (val.bin = 10% del total)
#
# Usamos uint16 porque el vocabulario es < 65536 (2 bytes por token).
# Si el vocabulario fuese mayor (BPE de 100K tokens), usariamos uint32.
#
# El array final se lee con np.memmap durante el entrenamiento: el SO se
# encarga de cargar paginas a demanda, asi no necesitamos cargar 10GB en RAM.
# ----------------------------------------------------------------------------

def build_vocab(corpus_dir: Path, max_files=None, min_count=100):
    """
    Recorre los .txt del corpus y construye el vocabulario.

    Filtra caracteres que aparezcan menos de `min_count` veces (basura,
    caracteres exoticos sueltos): los reemplazaremos por '?' al tokenizar.
    Asi mantenemos el vocabulario manejable (~150-200 chars en lugar de
    miles de caracteres Unicode raros que aparecen 1-2 veces).

    Devuelve un dict {caracter: id}.
    """
    print(f"[prepare] escaneando vocabulario en {corpus_dir} ...")
    char_counts = {}
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        # El corpus podria venir como un unico .txt o sin extension
        files = sorted(p for p in corpus_dir.iterdir() if p.is_file())
    if max_files:
        files = files[:max_files]
    if not files:
        sys.exit(f"ERROR: no se encontraron ficheros en {corpus_dir}")

    for fpath in files:
        size_mb = fpath.stat().st_size / 1e6
        print(f"  scanning {fpath.name} ({size_mb:.1f} MB)")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            # Lectura en bloques de 1MB para no agotar RAM con ficheros grandes
            while chunk := f.read(1024 * 1024):
                for ch in chunk:
                    char_counts[ch] = char_counts.get(ch, 0) + 1

    # Aseguramos siempre un token '?' como fallback para caracteres descartados
    chars = sorted([c for c, n in char_counts.items() if n >= min_count])
    if '?' not in chars:
        chars.append('?')
    stoi = {ch: i for i, ch in enumerate(chars)}
    descartados = sum(1 for n in char_counts.values() if n < min_count)
    print(f"  vocabulario: {len(stoi)} tokens (descartados {descartados} chars con <{min_count} apariciones)")
    return stoi


def tokenize_corpus(corpus_dir: Path, stoi: dict, out_path: Path, max_files=None):
    """
    Tokeniza el corpus completo y lo escribe a un fichero binario uint16.
    Procesa en streaming, escribiendo a disco bloque a bloque.
    """
    print(f"[prepare] tokenizando -> {out_path}")
    fallback = stoi.get('?', 0)
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        files = sorted(p for p in corpus_dir.iterdir() if p.is_file())
    if max_files:
        files = files[:max_files]

    total_tokens = 0
    t0 = time.time()
    with open(out_path, "wb") as out:
        for fpath in files:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                while chunk := f.read(1024 * 1024):
                    # Mapear cada caracter a su id (o al fallback si no esta)
                    ids = np.fromiter(
                        (stoi.get(ch, fallback) for ch in chunk),
                        dtype=np.uint16, count=len(chunk)
                    )
                    out.write(ids.tobytes())
                    total_tokens += len(ids)
            elapsed = time.time() - t0
            rate = total_tokens / elapsed / 1e6 if elapsed > 0 else 0
            print(f"  tokenized {fpath.name}: {total_tokens:,} tokens total ({rate:.1f}M tok/s)")
    return total_tokens


def cmd_prepare(args):
    """Comando 'prepare': tokeniza el corpus y guarda train.bin / val.bin / vocab.pkl."""
    DATA_DIR.mkdir(exist_ok=True)
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        sys.exit(f"ERROR: no existe el directorio {corpus_dir}")

    # 1) Vocabulario
    stoi = build_vocab(corpus_dir, max_files=args.max_files)
    itos = {i: ch for ch, i in stoi.items()}

    # 2) Tokenizar todo a un binario temporal
    tmp_path = DATA_DIR / "_tokens.bin"
    total = tokenize_corpus(corpus_dir, stoi, tmp_path, max_files=args.max_files)

    # 3) Split 90/10 en train/val (copia en bloques con memmap)
    print(f"[prepare] dividiendo train (90%) / val (10%) ...")
    src = np.memmap(tmp_path, dtype=np.uint16, mode='r')
    split = int(0.9 * total)
    chunk = 1 << 22  # 4M tokens por bloque (8 MB)

    with open(DATA_DIR / "train.bin", "wb") as f:
        for i in range(0, split, chunk):
            f.write(src[i:min(i + chunk, split)].tobytes())
    with open(DATA_DIR / "val.bin", "wb") as f:
        for i in range(split, total, chunk):
            f.write(src[i:min(i + chunk, total)].tobytes())

    del src
    tmp_path.unlink()

    # 4) Guardar vocabulario
    with open(DATA_DIR / "vocab.pkl", "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos, "vocab_size": len(stoi)}, f)

    print(f"[prepare] OK")
    print(f"  train.bin: {split:,} tokens")
    print(f"  val.bin:   {total - split:,} tokens")
    print(f"  vocab:     {len(stoi)} tokens -> {DATA_DIR / 'vocab.pkl'}")


# ============================================================================
# 3) EL MODELO: GPT en PyTorch
# ============================================================================
# La arquitectura es exactamente la misma que en microgpt.py, pero PyTorch
# se encarga de:
#   - paralelizar todo en GPU (lo que en microgpt eran bucles for)
#   - propagar gradientes automaticamente (lo que en microgpt era la clase Value)
#   - usar Flash Attention si esta disponible (atencion mucho mas eficiente
#     en memoria, equivalente matematicamente al softmax(QK^T/sqrt(d))V manual)
#
# Sigue dividida en las mismas piezas:
#   - CausalSelfAttention: la atencion (con mascara causal: solo mira hacia atras)
#   - MLP:                 la red feed-forward (expand -> activation -> contract)
#   - Block:               atencion + MLP con conexiones residuales y LayerNorm
#   - GPT:                 embeddings + N bloques + lm_head
# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Atencion multi-cabeza CAUSAL (cada token solo puede mirar tokens anteriores).

    En microgpt.py teniamos 4 matrices separadas (Wq, Wk, Wv, Wo). Aqui
    fusionamos Wq, Wk, Wv en una sola (`c_attn`) que produce los 3 vectores
    de un solo matmul (3x mas rapido).
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, bias=False):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd debe ser divisible por n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        # Proyeccion combinada: x (n_embd) -> Q,K,V (3 * n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Proyeccion final que mezcla las cabezas (la "Wo" de microgpt)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Si no tenemos Flash Attention (PyTorch < 2.0), creamos la mascara causal
        # manualmente como buffer (no es parametro entrenable).
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        # x: (B, T, C) donde B=batch, T=tokens, C=n_embd
        B, T, C = x.shape

        # 1) Calcular Q, K, V en un solo matmul y separarlos
        qkv = self.c_attn(x)                            # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)         # cada uno (B, T, C)

        # 2) Reorganizar para multi-head:
        #    (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        # Cada "cabeza" trabaja en un subespacio de tamanio head_dim.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3) Atencion: softmax(Q @ K^T / sqrt(d)) @ V, con mascara causal
        if self.flash:
            # PyTorch 2.0+ tiene una implementacion super optimizada que evita
            # materializar la matriz de atencion completa (NxN) en memoria.
            # Internamente hace lo mismo que microgpt.py pero ~10x mas rapido.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Implementacion manual (mismo algoritmo que microgpt.py)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # 4) Concatenar las cabezas: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5) Proyeccion final (Wo)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Mini-red feed-forward por token: expand (4x) -> GELU -> contract.

    Es la misma estructura que en microgpt.py, pero usamos GELU en vez de
    ReLU. GELU es mas suave que ReLU y suele dar mejor rendimiento en
    practicas de NLP (es el estandar en GPT-2, GPT-3, BERT, etc.).
    """
    def __init__(self, n_embd, dropout=0.0, bias=False):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=bias)   # expansion
        self.gelu   = nn.GELU()                                  # no-linealidad
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)   # contraccion
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm con bias opcional. PyTorch tiene `nn.LayerNorm` pero su bias
    no se puede desactivar facilmente. Esta version permite `bias=False`
    (estilo Llama: solo escala, sin shift).
    """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    """
    Un bloque transformer = LayerNorm + Atencion + LayerNorm + MLP, con
    conexiones residuales (sumamos la entrada despues de cada subbloque).

    Esquema (pre-norm, igual que microgpt.py):

         x ---> LN ---> Attn --> + ---> LN ---> MLP --> +  ---> salida
         |________________________|       |_______________|
            (residual conn 1)              (residual conn 2)
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp  = MLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # subblock 1: atencion
        x = x + self.mlp(self.ln_2(x))    # subblock 2: feed-forward
        return x


class GPT(nn.Module):
    """
    El modelo completo. Composicion de:
      - wte: token embedding   (vocab_size -> n_embd)
      - wpe: position embedding (block_size -> n_embd)
      - N x Block: bloques transformer
      - ln_f: LayerNorm final
      - lm_head: proyeccion a logits (n_embd -> vocab_size)

    TRUCO: weight tying. Compartimos los pesos entre `wte` y `lm_head`,
    porque ambas son matrices [vocab_size, n_embd] y se ha demostrado que
    funcionan mejor compartidas (papers de Press & Wolf 2017).
    """
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 dropout=0.0, bias=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer    = n_layer
        self.n_head     = n_head
        self.n_embd     = n_embd

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout, bias) for _ in range(n_layer)
        ])
        self.ln_f = LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying entre embedding de entrada y proyeccion de salida
        self.wte.weight = self.lm_head.weight

        # Inicializacion de pesos (importante para entrenamiento estable)
        self.apply(self._init_weights)
        # Para las proyecciones residuales, escalamos por 1/sqrt(2*n_layer)
        # (ver paper original GPT-2). Evita que la varianza explote al apilar capas.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        # Restamos los embeddings de posicion (no cuentan como "no-embedding params")
        print(f"[model] {n_params/1e6:.2f}M parametros totales")

    def _init_weights(self, module):
        """Inicializacion estandar de GPT-2: normal(0, 0.02)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) tensor de ids de tokens
        targets: (B, T) tensor con el token siguiente esperado en cada posicion

        Si targets es None, devolvemos solo logits (modo inferencia).
        Si no, calculamos cross-entropy y devolvemos (logits, loss).
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Secuencia de {T} tokens > block_size={self.block_size}"

        # Embeddings de token y de posicion
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)   # (B, T, n_embd)
        pos_emb = self.wpe(pos)   # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Pila de bloques transformer
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            # Modo entrenamiento: calculamos logits para todas las posiciones
            logits = self.lm_head(x)   # (B, T, vocab_size)
            # Cross-entropy: -log(prob asignada al token correcto)
            # F.cross_entropy aplica internamente softmax + log + nll
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Modo inferencia: solo nos importa el ultimo token (mas barato)
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Crea un AdamW separando los parametros en dos grupos:
          - 2D (matrices de pesos): aplican weight decay
          - 1D (biases, LayerNorm): NO aplican weight decay

        Esta es la receta estandar en GPT-2/3/Llama (ver paper de AdamW).
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() <  2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        # `fused=True` es una optimizacion en CUDA que junta las operaciones
        # del optimizador en un solo kernel GPU.
        use_fused = (DEVICE == 'cuda')
        extra = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generacion auto-regresiva: dado un prompt `idx` (B, T_prompt),
        genera `max_new_tokens` tokens nuevos y los va concatenando.

        - temperature: < 1.0 mas conservador, > 1.0 mas creativo
        - top_k: si se da, solo muestreamos entre los k tokens mas probables
                 (corta la cola de probabilidades pequenias = menos disparates)
        """
        for _ in range(max_new_tokens):
            # Si la secuencia es mas larga que block_size, recortamos al final
            # (el modelo solo puede ver block_size tokens hacia atras)
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# 4) ENTRENAMIENTO
# ============================================================================
# El bucle es conceptualmente identico al de microgpt.py:
#   forward -> loss -> backward -> step -> repeat
#
# La diferencia es que aqui:
#   - Procesamos BATCH_SIZE secuencias en paralelo (no una sola)
#   - Usamos memmap para no cargar el corpus entero en RAM
#   - Hacemos validacion periodica para detectar overfitting
#   - Usamos warmup + cosine decay para el learning rate
#   - Acumulamos gradientes (batch efectivo grande con poca VRAM)
#   - Usamos mixed precision (bfloat16) en GPU para acelerar
# ----------------------------------------------------------------------------

def get_batch(split, train_data, val_data, block_size, batch_size, device):
    """
    Saca un batch aleatorio del split correspondiente.

    Cogemos `batch_size` posiciones al azar y de cada una extraemos una
    ventana de `block_size + 1` tokens. Los primeros `block_size` son la
    entrada (x), y desplazados un token a la derecha son los targets (y).

    Asi entrenamos al modelo a predecir el siguiente token en cada posicion.
    """
    data = train_data if split == 'train' else val_data
    # Posiciones aleatorias (con margen para que la ventana entre)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Construimos los tensores de entrada y target
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))     for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin_memory + non_blocking acelera la transferencia CPU -> GPU
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it):
    """
    Schedule del learning rate: warmup lineal + cosine decay hasta MIN_LR.

    Curva:
        |  /----\
        | /      \____
        |/            ----.____
        +--warmup--decay--min_lr----> iteraciones

    Es el schedule estandar en LLMs modernos (GPT-3, Llama, etc.).
    """
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, autocast_ctx):
    """
    Estima la loss en train y val promediando varios batches.

    Importante: ponemos el modelo en `eval()` para desactivar dropout, y
    al final lo devolvemos a `train()`. Si no, los resultados serian ruidosos.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            with autocast_ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def cmd_train(args):
    """Comando 'train': entrena el modelo desde cero."""
    CKPT_DIR.mkdir(exist_ok=True)

    # ---- Cargar vocabulario y datos ----
    if not (DATA_DIR / 'vocab.pkl').exists():
        sys.exit(f"ERROR: no existe {DATA_DIR / 'vocab.pkl'}. Ejecuta 'prepare' primero.")
    with open(DATA_DIR / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = vocab['vocab_size']

    train_data = np.memmap(DATA_DIR / 'train.bin', dtype=np.uint16, mode='r')
    val_data   = np.memmap(DATA_DIR / 'val.bin',   dtype=np.uint16, mode='r')
    print(f"[train] device={DEVICE} dtype={DTYPE}")
    print(f"[train] train: {len(train_data):,} tokens  |  val: {len(val_data):,} tokens")
    print(f"[train] vocab_size: {vocab_size}")

    # ---- Crear modelo ----
    model = GPT(
        vocab_size=vocab_size, block_size=BLOCK_SIZE, n_layer=N_LAYER,
        n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT, bias=BIAS,
    ).to(DEVICE)

    # ---- Optimizador y mixed precision ----
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2))

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}[DTYPE]
    autocast_ctx = (
        nullcontext() if DEVICE == 'cpu'
        else torch.amp.autocast(device_type=DEVICE, dtype=ptdtype)
    )
    # GradScaler: solo necesario para float16. bfloat16 tiene rango similar a fp32.
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DTYPE == 'float16'))

    # ---- Bucle de entrenamiento ----
    print(f"[train] iniciando entrenamiento, {MAX_ITERS} iteraciones")
    best_val_loss = float('inf')
    t0 = time.time()
    X, Y = get_batch('train', train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)

    for it in range(MAX_ITERS + 1):

        # 1) Aplicar learning rate del schedule
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 2) Validacion periodica + checkpoint si mejora
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data, autocast_ctx)
            print(f"[eval] step {it}: train {losses['train']:.4f}  val {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if it > 0:
                    ckpt = {
                        'model_state': model.state_dict(),
                        'config': dict(
                            vocab_size=vocab_size, block_size=BLOCK_SIZE,
                            n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                            dropout=DROPOUT, bias=BIAS,
                        ),
                        'iter': it,
                        'val_loss': losses['val'],
                    }
                    torch.save(ckpt, CKPT_DIR / 'ckpt.pt')
                    print(f"  guardado checkpoint en {CKPT_DIR / 'ckpt.pt'} (val_loss={losses['val']:.4f})")

        if it == MAX_ITERS:
            break

        # 3) Forward + backward con gradient accumulation
        # Acumulamos GRAD_ACCUM_STEPS micro-batches antes de hacer optimizer.step()
        # para simular un batch mas grande sin gastar mas VRAM.
        for micro_step in range(GRAD_ACCUM_STEPS):
            with autocast_ctx:
                _, loss = model(X, Y)
                loss = loss / GRAD_ACCUM_STEPS
            # Pre-fetch del siguiente batch mientras backward calcula gradientes
            X, Y = get_batch('train', train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            scaler.scale(loss).backward()

        # 4) Clip de gradientes para estabilidad
        if GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # 5) Paso del optimizador
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # 6) Logging
        if it % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(1, it if it > 0 else 1)
            print(f"  it {it:5d}  loss {loss.item() * GRAD_ACCUM_STEPS:.4f}  "
                  f"lr {lr:.2e}  {dt:.1f} ms/it", end='\r')

    print(f"\n[train] entrenamiento terminado. mejor val_loss: {best_val_loss:.4f}")


# ============================================================================
# 5) INFERENCIA: generar texto con el modelo entrenado
# ============================================================================

def cmd_sample(args):
    """Comando 'sample': carga el checkpoint y genera texto desde un prompt."""
    if not (CKPT_DIR / 'ckpt.pt').exists():
        sys.exit(f"ERROR: no existe {CKPT_DIR / 'ckpt.pt'}. Entrena primero.")
    if not (DATA_DIR / 'vocab.pkl').exists():
        sys.exit(f"ERROR: no existe {DATA_DIR / 'vocab.pkl'}.")

    # Cargar vocabulario
    with open(DATA_DIR / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    stoi, itos = vocab['stoi'], vocab['itos']
    fallback = stoi.get('?', 0)

    # Cargar checkpoint y reconstruir modelo
    print(f"[sample] cargando checkpoint ...")
    ckpt = torch.load(CKPT_DIR / 'ckpt.pt', map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    model = GPT(**cfg).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[sample] modelo cargado (iter {ckpt['iter']}, val_loss {ckpt['val_loss']:.4f})")

    # Tokenizar prompt
    prompt = args.prompt or "El "
    prompt_ids = [stoi.get(c, fallback) for c in prompt]
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # Generar
    print(f"\n--- generando ({args.max_new_tokens} tokens, T={args.temperature}, top_k={args.top_k}) ---")
    print(prompt, end='', flush=True)
    with torch.no_grad():
        # Generamos token a token e imprimimos en streaming para feedback inmediato
        for _ in range(args.max_new_tokens):
            x_cond = x if x.size(1) <= cfg['block_size'] else x[:, -cfg['block_size']:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / args.temperature
            if args.top_k:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            ch = itos.get(x_next.item(), '?')
            print(ch, end='', flush=True)
            x = torch.cat((x, x_next), dim=1)
    print("\n")


# ============================================================================
# 6) CLI: punto de entrada
# ============================================================================
# Tres subcomandos: prepare / train / sample
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="nanoGPT en castellano (SBWC)")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_prep = sub.add_parser('prepare', help='Tokeniza el corpus y crea train.bin/val.bin')
    p_prep.add_argument('--corpus_dir', default='spanish_billion_words',
                        help='Directorio con los .txt del corpus')
    p_prep.add_argument('--max_files', type=int, default=None,
                        help='Limitar a los primeros N ficheros (para pruebas rapidas)')

    p_train = sub.add_parser('train', help='Entrena el modelo')

    p_sample = sub.add_parser('sample', help='Genera texto con el modelo entrenado')
    p_sample.add_argument('--prompt', type=str, default="El ",
                          help='Texto inicial')
    p_sample.add_argument('--max_new_tokens', type=int, default=500,
                          help='Cuantos tokens generar')
    p_sample.add_argument('--temperature', type=float, default=0.8,
                          help='Aleatoriedad: <1 conservador, >1 creativo')
    p_sample.add_argument('--top_k', type=int, default=50,
                          help='Muestrear solo entre los k tokens mas probables')

    args = parser.parse_args()
    if args.cmd == 'prepare':
        cmd_prepare(args)
    elif args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'sample':
        cmd_sample(args)


if __name__ == '__main__':
    main()
