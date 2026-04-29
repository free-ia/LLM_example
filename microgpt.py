"""
============================================================================
 microgpt.py - Un GPT completo en Python puro, sin dependencias externas
============================================================================

Esta es la forma más atómica posible de entrenar y ejecutar inferencia con
un modelo GPT (un "Large Language Model" o LLM, como los que usan ChatGPT,
Claude, Gemini, etc.). Todo el algoritmo cabe en este único archivo.
Todo lo demás que ves en frameworks reales (PyTorch, JAX, etc.) son
optimizaciones de eficiencia: el corazón es esto.

Autor original: @karpathy

----------------------------------------------------------------------------
 ¿QUE ES UN LLM EN UNA FRASE?
----------------------------------------------------------------------------
Un LLM es una funcion matematica gigantesca que, dado un trozo de texto,
predice cual es el SIGUIENTE caracter (o "token") mas probable. Repitiendo
esa prediccion una y otra vez, va construyendo texto coherente.

Este archivo entrena un LLM en miniatura usando una lista de nombres de
personas. Despues de entrenar, el modelo "alucina" nombres nuevos que se
parecen a los que ha visto, pero que no existen en el dataset.

----------------------------------------------------------------------------
 LAS 5 PIEZAS DE UN LLM (y donde encontrarlas en este archivo)
----------------------------------------------------------------------------
  1. DATASET     -> los datos de entrenamiento (lista de nombres)
  2. TOKENIZER   -> convierte texto en numeros (tokens) y viceversa
  3. MODELO      -> la funcion `gpt(...)` con sus parametros (state_dict)
  4. ENTRENAMIENTO -> bucle que ajusta los parametros para minimizar el error
  5. INFERENCIA  -> usar el modelo entrenado para generar texto nuevo

Para una explicacion visual y mas extensa, lee EXPLICACION.md en este mismo
directorio.
============================================================================
"""

import os       # os.path.exists  -> comprobar si el dataset ya esta descargado
import math     # math.log, math.exp -> operaciones matematicas escalares
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Fijamos la semilla para que el experimento sea reproducible
                # (mismas inicializaciones aleatorias y mismo orden de datos
                # en cada ejecucion). 42 es la constante tradicional.

# ============================================================================
# 1) DATASET
# ============================================================================
# Un LLM aprende imitando datos. Aqui el "dataset" es simplemente una lista
# de strings (cada string es un "documento"). En este caso son nombres de
# personas en ingles, descargados desde el repositorio "makemore" de Karpathy.
#
# En modelos reales (GPT-3/4, Claude, etc.) el dataset son TRILLONES de
# tokens: paginas web, libros, codigo, articulos cientificos, etc. La idea
# es la misma: una lista de documentos de texto.
# ----------------------------------------------------------------------------
if not os.path.exists('input.txt'):
    # Si no existe el fichero local, lo descargamos. Solo se hace una vez.
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# Cargamos cada linea no vacia como un "documento" (un nombre).
docs = [line.strip() for line in open('input.txt') if line.strip()]
# Mezclamos para que el orden de presentacion al modelo no introduzca sesgos.
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ============================================================================
# 2) TOKENIZER
# ============================================================================
# Las redes neuronales no procesan texto, procesan numeros. El "tokenizer"
# es el traductor entre ambos mundos:
#
#     "ana"  --tokenizer-->  [1, 14, 1]   (cada caracter -> un id entero)
#
# En modelos reales el tokenizer trabaja con sub-palabras (BPE, SentencePiece)
# y tiene vocabularios de ~50.000 a ~200.000 tokens. Aqui simplificamos al
# maximo: cada CARACTER unico es un token. Por ejemplo si en los nombres
# aparecen las letras a-z, tendremos 26 tokens, mas un token especial.
# ----------------------------------------------------------------------------

# Caracteres unicos ordenados -> ids 0..n-1
uchars = sorted(set(''.join(docs)))

# Token especial "Beginning Of Sequence" (BOS): marca el inicio y el final
# de un documento. Sin esta marca el modelo no sabria cuando empezar a
# generar ni cuando parar. Su id es el siguiente disponible.
BOS = len(uchars)

# Tamanio total del vocabulario = caracteres unicos + el token BOS.
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ============================================================================
# 3) AUTOGRAD: el motor de derivadas automaticas
# ============================================================================
# "Entrenar" una red neuronal significa ajustar millones (o billones) de
# numeros llamados "parametros" para que el modelo cometa cada vez menos
# errores. Para saber como ajustarlos necesitamos el GRADIENTE: cuanto
# cambia el error si modificamos ligeramente cada parametro.
#
# Calcular gradientes a mano es imposible para un modelo grande. Por eso
# se usa "autograd" (auto-diferenciacion): cada operacion matematica se
# registra en un GRAFO COMPUTACIONAL, y luego se recorre hacia atras
# aplicando la regla de la cadena del calculo diferencial.
#
# En PyTorch o TensorFlow esto esta MUY optimizado y opera con tensores
# (matrices N-dimensionales) en GPU. Aqui implementamos la version mas
# basica posible: cada `Value` es un escalar (un solo numero) que recuerda
# de donde viene.
# ----------------------------------------------------------------------------
class Value:
    # __slots__ es una optimizacion de memoria: evita que cada instancia
    # tenga un dict, ahorrando bytes (importante porque crearemos millones).
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # Valor numerico (forward pass)
        self.grad = 0                   # Derivada del error respecto a este valor (backward pass)
        self._children = children       # Nodos de los que dependo en el grafo
        self._local_grads = local_grads # Derivada local respecto a cada hijo

    # ---- Operaciones aritmeticas: cada una crea un nuevo Value y guarda
    # ---- la derivada local frente a sus operandos. ----

    def __add__(self, other):
        # d(a+b)/da = 1, d(a+b)/db = 1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # d(a*b)/da = b, d(a*b)/db = a
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))      # d(ln x)/dx = 1/x
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),)) # d(e^x)/dx = e^x
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),)) # ReLU: max(0,x)
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """
        Backpropagation: calcula self.grad respecto a TODOS los nodos
        del grafo computacional usando la regla de la cadena.

        Pasos:
          1) Ordenacion topologica del grafo (de hojas a raiz).
          2) Iniciamos self.grad = 1 (derivada de uno mismo).
          3) Recorremos el grafo en orden inverso, propagando gradientes:
                child.grad += local_grad * v.grad
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        # Recorrer al reves = ir desde el output (loss) hacia los parametros
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ============================================================================
# 4) PARAMETROS DEL MODELO
# ============================================================================
# Estos son los "pesos" que el modelo va a aprender. Inicialmente son
# numeros aleatorios pequenios; durante el entrenamiento se ajustan para
# que el modelo prediga bien el siguiente token.
#
# La cantidad de parametros define la "capacidad" del modelo:
#   - GPT-2 small:  ~124 millones
#   - GPT-3:        ~175.000 millones (175B)
#   - Claude / GPT-4: cientos de miles de millones (no publicos)
#   - Este micro-GPT: unos miles
# ----------------------------------------------------------------------------
n_layer   = 1   # Profundidad: cuantos bloques transformer apilamos.
                # GPT-2 small=12, GPT-3=96. Aqui solo 1 (suficiente para nombres).
n_embd    = 16  # Anchura: dimension de los vectores que representan cada token.
                # GPT-3 usa 12288. Aqui 16.
block_size = 16 # Contexto maximo: cuantos tokens puede "ver" el modelo de una vez.
                # GPT-3 = 2048. Aqui 16 (el nombre mas largo tiene 15 letras).
n_head    = 4   # Numero de "cabezas" de atencion (ver explicacion abajo).
head_dim  = n_embd // n_head # Dimension por cabeza.

# Helper para crear matrices de parametros aleatorios con distribucion gausiana
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# state_dict = "diccionario de estado" del modelo. Contiene TODAS las matrices
# de pesos. En modelos reales este es el fichero ".bin" o ".safetensors" que
# se publica para compartir un modelo entrenado.
state_dict = {
    'wte':     matrix(vocab_size, n_embd),  # Word Token Embedding: token_id -> vector
    'wpe':     matrix(block_size, n_embd),  # Word Position Embedding: posicion -> vector
    'lm_head': matrix(vocab_size, n_embd),  # Language Model head: vector -> logits sobre vocab
}

# Por cada capa transformer, anadimos los pesos de Atencion y de la MLP
for i in range(n_layer):
    # ---- Pesos de Atencion (Q, K, V, O) ----
    # En la atencion cada token genera 3 vectores: Query, Key, Value.
    # Q ("que estoy buscando?"), K ("que ofrezco?"), V ("que informacion llevo?").
    # Wo ("output projection") combina las salidas de las cabezas.
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)

    # ---- Pesos de la MLP (Multi-Layer Perceptron) ----
    # Tras la atencion, cada vector pasa por una pequenia red feed-forward
    # que expande la dimension a 4x y luego la vuelve a comprimir.
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# Aplanamos todos los parametros en una sola lista para iterarlos facilmente
# durante el optimizador.
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ============================================================================
# 5) ARQUITECTURA DEL MODELO (la funcion gpt)
# ============================================================================
# Un GPT es una pila de "bloques transformer". Cada bloque tiene 2 piezas:
#
#   (a) ATENCION: cada token mira a los anteriores y decide cuanto "atender"
#       a cada uno. Es lo que permite que el modelo entienda dependencias
#       a larga distancia ("Juan... el dijo..." -> "el" se refiere a "Juan").
#
#   (b) MLP: una pequenia red densa por token que "procesa" la informacion
#       agregada por la atencion.
#
# Antes de las capas, el token se convierte en un vector via "embeddings"
# (token + posicion). Despues de las capas, otra capa lineal convierte el
# vector final en LOGITS: una puntuacion por cada posible siguiente token.
#
# Cuanto mas alto el logit -> mas probable que el modelo elija ese token.
# ----------------------------------------------------------------------------

def linear(x, w):
    """
    Capa lineal (multiplicacion matriz-vector): y = W * x
    `w` es una matriz [nout][nin], `x` es un vector [nin], devuelve [nout].
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """
    Convierte logits en probabilidades (suman 1).
    Restamos el max por estabilidad numerica (evita overflow al hacer exp).
        softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
    """
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """
    RMSNorm: normaliza el vector dividiendo por su "magnitud RMS".
    Mantiene los valores en una escala estable a traves de las capas, lo
    cual es CRUCIAL para que los gradientes no exploten ni se desvanezcan.
    Es como ajustar el volumen automaticamente entre canciones.
    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """
    Forward pass del modelo: dado un token y su posicion, calcula los
    "logits" (puntuaciones) sobre el vocabulario para predecir el siguiente.

    `keys` y `values` son CACHES: en inferencia no recalculamos las K,V de
    los tokens anteriores cada vez. Esto se llama "KV-cache".

    Sigue la arquitectura de GPT-2 con pequenias diferencias modernas:
       - LayerNorm reemplazado por RMSNorm (mas simple)
       - Sin biases en las capas lineales (Llama-style)
       - GeLU reemplazado por ReLU (mas facil de implementar a mano)
    """
    # ---- (1) EMBEDDINGS: token + posicion ----
    # Cada token_id se mapea a un vector (su "significado" aprendido).
    # Cada posicion tambien tiene su vector (el modelo necesita saber el
    # orden, ya que la atencion por si sola no lo captura).
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # vector inicial del token
    x = rmsnorm(x)  # normalizamos antes de entrar al primer bloque

    # ---- (2) BLOQUES TRANSFORMER ----
    for li in range(n_layer):

        # ---- (2a) BLOQUE DE ATENCION MULTI-CABEZA ----
        x_residual = x          # guardamos para la "conexion residual"
        x = rmsnorm(x)          # pre-norm: estabiliza el entrenamiento

        # Proyectamos x a tres espacios: Query, Key, Value
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # Almacenamos K y V en la cache (necesarios para tokens futuros)
        keys[li].append(k)
        values[li].append(v)

        # Multi-head attention: dividimos los vectores en `n_head` trozos
        # y cada trozo ("cabeza") atiende de forma independiente. Esto deja
        # que distintas cabezas se especialicen (ej: una en sintaxis, otra
        # en semantica, otra en posiciones, etc.).
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]                                     # query de esta cabeza
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]               # todas las keys pasadas
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]             # todas las values pasadas

            # ATENCION = "cuanto le importa a este token cada token anterior?"
            # Producto escalar Q.K -> medida de similitud, escalado por sqrt(d).
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(k_h))]

            # Convertimos esas similitudes en pesos que suman 1
            attn_weights = softmax(attn_logits)

            # Salida = combinacion ponderada de los Values usando esos pesos.
            # Es decir: "mezclamos" la informacion de tokens anteriores
            # segun lo importantes que sean para el token actual.
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)

        # Proyeccion final que mezcla las cabezas
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])

        # CONEXION RESIDUAL: sumamos la entrada original.
        # Asi el modelo puede "saltarse" capas si no las necesita, lo que
        # facilita el entrenamiento de redes muy profundas (ResNet, 2015).
        x = [a + b for a, b in zip(x, x_residual)]

        # ---- (2b) BLOQUE MLP (feed-forward) ----
        # Cada token pasa por una mini-red densa: expande -> ReLU -> contrae.
        # Aqui es donde el modelo guarda gran parte del "conocimiento"
        # (asociaciones tipo "Paris -> Francia").
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])     # expansion (n_embd -> 4*n_embd)
        x = [xi.relu() for xi in x]                          # no-linealidad
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])     # contraccion (4*n_embd -> n_embd)
        x = [a + b for a, b in zip(x, x_residual)]          # otra residual

    # ---- (3) CABEZA DE LENGUAJE: vector -> logits sobre el vocabulario ----
    # Convertimos el vector final (de tamanio n_embd) en una puntuacion por
    # cada posible token siguiente. El token con la puntuacion mas alta es
    # la prediccion del modelo.
    logits = linear(x, state_dict['lm_head'])
    return logits

# ============================================================================
# 6) OPTIMIZADOR ADAM
# ============================================================================
# Tras calcular los gradientes, hay que decidir COMO mover los parametros.
# El metodo basico es "gradient descent": p = p - lr * grad
#
# Adam es una version mas inteligente que adapta el "tamanio del paso" para
# cada parametro usando dos buffers:
#   - m: media movil del gradiente (momentum, "inercia")
#   - v: media movil del cuadrado del gradiente (escala adaptativa)
#
# Es el optimizador estandar de facto en deep learning desde 2015.
# ----------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # primer momento (media de gradientes)
v = [0.0] * len(params)  # segundo momento (media de gradientes^2)

# ============================================================================
# 7) BUCLE DE ENTRENAMIENTO
# ============================================================================
# La idea es repetir muchas veces:
#   1) Coger un documento del dataset
#   2) Pasarlo por el modelo (forward pass) y medir cuanto se equivoca (LOSS)
#   3) Calcular gradientes (backward pass)
#   4) Actualizar parametros con Adam
#   5) Volver a empezar
#
# La LOSS (perdida) usada es "cross-entropy": -log(probabilidad asignada al
# token correcto). Si el modelo asigna probabilidad 1 al token correcto,
# loss=0. Cuanto menos probable lo crea, mayor sera la loss.
# ----------------------------------------------------------------------------
num_steps = 1000  # numero de pasos de entrenamiento
for step in range(num_steps):

    # ---- 7.1) TOKENIZAMOS UN DOCUMENTO ----
    # Lo rodeamos con BOS al principio y al final: marca explicita de
    # inicio/fin para que el modelo aprenda donde empieza y termina un nombre.
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # ---- 7.2) FORWARD: pasamos cada posicion por el modelo y medimos la loss ----
    # Para cada posicion `pos_id`, el modelo intenta predecir tokens[pos_id+1]
    # a partir de tokens[pos_id]. Esto se llama "next-token prediction".
    # Es la tarea fundamental de TODOS los LLMs.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        # Cross-entropy: penalizamos al modelo en proporcion a lo poco
        # probable que considero el token correcto.
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)  # loss media sobre el documento

    # ---- 7.3) BACKWARD: calcula gradientes de la loss respecto a TODOS los params ----
    loss.backward()

    # ---- 7.4) ADAM: actualizamos los parametros ----
    # Decay lineal del learning rate: empieza alto y baja a 0 al final.
    # Esto ayuda a converger mejor (al final queremos pasos pequenios).
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad             # momentum
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2        # escala adaptativa
        m_hat = m[i] / (1 - beta1 ** (step + 1))               # bias correction
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)     # paso de actualizacion
        p.grad = 0                                              # reset para el siguiente step

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# ============================================================================
# 8) INFERENCIA: usamos el modelo entrenado para generar nombres nuevos
# ============================================================================
# Empezamos con BOS y vamos pidiendo al modelo el siguiente token, una y
# otra vez, hasta que devuelve BOS (= "fin del nombre").
#
# La "temperatura" controla la aleatoriedad:
#   - temperature -> 0   : el modelo es deterministico (siempre el mas probable)
#   - temperature = 1    : muestreamos segun las probabilidades nativas
#   - temperature alta   : mas creatividad y caos
# Dividir los logits por T antes del softmax hace exactamente esto.
# ----------------------------------------------------------------------------
temperature = 0.5  # un equilibrio entre coherencia y variedad
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # arrancamos con el token de inicio
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        # Muestreamos un token segun la distribucion de probabilidades
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break  # el modelo dice "fin del nombre"
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
