# LLM_example — Un GPT en Python puro, explicado para profanos

Un modelo GPT (Large Language Model) **completo, funcional y entrenable**, escrito en Python con el objetivo de servir como material didáctico para entender desde cero cómo funcionan los modelos detrás de ChatGPT, Claude o Gemini.

El repo contiene **dos versiones del mismo algoritmo**, en orden creciente de realismo:

| Fichero | Dependencias | Hardware | Datos | Parámetros | Para qué sirve |
|---|---|---|---|---|---|
| `microgpt.py` | ninguna | CPU | ~30 KB de nombres | ~4 000 | **Entender el algoritmo** desde el escalar más básico |
| `nanogpt_es.py` | torch, numpy | GPU recomendada | ~10 GB de castellano | ~10 M | **Entrenar un modelo real** sobre texto en español |

## Documentación

- **`EXPLICACION.md`** — documento didáctico con diagramas (Mermaid + ASCII) que cubre desde tokenización hasta scaling laws y las técnicas modernas (Flash Attention, AdamW, mixed precision, etc.). Pensado para leerse antes o en paralelo al código.
- Los propios ficheros `.py` están **muy comentados en español**: cada bloque explica qué hace y por qué.

## Parte 1 — `microgpt.py`: el algoritmo desnudo

GPT en menos de 200 líneas de Python puro, **sin instalar nada**. Entrena sobre una lista de nombres y al final genera nombres "alucinados".

```bash
python microgpt.py
```

La primera ejecución descarga automáticamente el dataset (lista de nombres). El entrenamiento dura unos minutos en cualquier portátil.

## Parte 2 — `nanogpt_es.py`: GPT real sobre castellano

Versión PyTorch del mismo algoritmo, ya entrenable sobre el **Spanish Billion Words Corpus** (SBWC) de Cristian Cardellino (Universidad Nacional de Córdoba). Incluye todas las técnicas modernas: Flash Attention, mixed precision (bfloat16), gradient accumulation, warmup + cosine LR schedule, AdamW con grupos, top-k sampling, etc. — todas explicadas en `EXPLICACION.md` sección 12.

### Requisitos

```bash
pip install torch numpy
```

GPU NVIDIA recomendada (Apple Silicon también funciona vía MPS). En CPU funciona pero es **mucho** más lento.

### Paso 1 — Descargar y descomprimir el dataset

El corpus ocupa ~3 GB comprimido y ~10 GB descomprimido.

**Linux / macOS / WSL:**

```bash
wget http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/clean_corpus.tar.bz2
tar -xjf clean_corpus.tar.bz2
# crea el directorio spanish_billion_words/
```

**Windows (PowerShell):**

```powershell
Invoke-WebRequest -Uri http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/clean_corpus.tar.bz2 -OutFile clean_corpus.tar.bz2
tar -xjf clean_corpus.tar.bz2
```

(`tar` viene incluido en Windows 10+. Alternativamente: 7-Zip o WinRAR.)

El directorio resultante `spanish_billion_words/` debe estar al mismo nivel que `nanogpt_es.py`. Está incluido en `.gitignore` para que no se suba al repo.

### Paso 2 — Preparar los datos (tokenización)

Tokeniza el corpus carácter a carácter y crea `data/train.bin`, `data/val.bin` y `data/vocab.pkl`:

```bash
# Recomendado: empieza con 1-2 ficheros para comprobar que todo funciona
python nanogpt_es.py prepare --corpus_dir spanish_billion_words --max_files 2

# Cuando estés seguro, procesa el corpus entero (puede tardar unos minutos)
python nanogpt_es.py prepare --corpus_dir spanish_billion_words
```

### Paso 3 — Entrenar el modelo

```bash
python nanogpt_es.py train
```

El script detecta automáticamente CUDA / MPS / CPU. Imprime la pérdida cada 10 iteraciones y valida cada 250, guardando un checkpoint en `checkpoints/ckpt.pt` cuando la val loss mejora.

**Tiempos orientativos** con la configuración por defecto (~10 M parámetros, 5000 iteraciones):

| Hardware | Tiempo aproximado |
|---|---|
| GPU NVIDIA moderna (RTX 3060+) | ~30–60 minutos |
| Apple Silicon (M1/M2) | ~2–4 horas |
| CPU | ~días (no recomendado) |

Para iteraciones más rápidas en pruebas, edita las constantes al inicio del fichero (`MAX_ITERS`, `N_LAYER`, `N_EMBD`, etc.).

### Paso 4 — Generar texto

```bash
python nanogpt_es.py sample --prompt "Habia una vez"
```

Opciones útiles:

```bash
python nanogpt_es.py sample \
    --prompt "El gato" \
    --max_new_tokens 500 \
    --temperature 0.8 \
    --top_k 50
```

- `temperature` < 1 → texto más conservador y repetitivo
- `temperature` > 1 → texto más creativo y caótico
- `top_k` → limita el muestreo a los k tokens más probables (corta disparates)

## Fuente original

El código de `microgpt.py` está basado en el gist público de **Andrej Karpathy**:

> https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

`nanogpt_es.py` está inspirado en su proyecto **nanoGPT** (https://github.com/karpathy/nanoGPT), adaptado al corpus en castellano y con documentación exhaustiva en español.

Esta versión no modifica la lógica de los originales: **toda la aportación de este repositorio es documentación adicional en español** (comentarios en el código y el documento `EXPLICACION.md`) para hacerlo accesible a personas sin formación técnica avanzada.

## Sobre Andrej Karpathy

[Andrej Karpathy](https://karpathy.ai/) es una de las figuras más influyentes en la divulgación moderna de la inteligencia artificial:

- **Miembro fundador de OpenAI** (2015), donde participó en los primeros desarrollos que llevarían a GPT.
- **Director de IA en Tesla** (2017–2022), responsable del equipo de visión por computador del Autopilot.
- Volvió brevemente a OpenAI en 2023.
- En 2024 fundó **[Eureka Labs](https://eurekalabs.ai/)**, una empresa centrada en educación AI-nativa.
- Doctor en Stanford bajo la dirección de Fei-Fei Li, donde co-creó el curso **CS231n** (visión por computador), referencia mundial.

Es especialmente conocido por su capacidad pedagógica: proyectos como **micrograd**, **makemore**, **nanoGPT** y la serie de YouTube **"Neural Networks: Zero to Hero"** han enseñado a una generación entera de ingenieros cómo funcionan las redes neuronales modernas, construyéndolas desde cero. Este repositorio es un ejemplo perfecto de esa filosofía: el algoritmo completo, sin abstracciones que escondan lo esencial.

## Sobre el corpus

El **Spanish Billion Words Corpus (SBWC)** fue compilado por Cristian Cardellino (Universidad Nacional de Córdoba, Argentina) a partir de Wikipedia, ParaCrawl, OPUS, Europarl, etc. Contiene ~1.4 mil millones de palabras de texto plano en castellano ya limpio y tokenizado.

> Página oficial: http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/

## Licencia

- El código original de `microgpt.py` y la inspiración de `nanogpt_es.py` son de **Andrej Karpathy** (consultar [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) y [nanoGPT](https://github.com/karpathy/nanoGPT) para los términos).
- El corpus SBWC tiene su propia [licencia](http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/) (Cardellino, 2016).
- La documentación añadida en este repositorio (`EXPLICACION.md` y comentarios en español) se ofrece libremente con fines educativos.
