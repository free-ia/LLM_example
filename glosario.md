### Glosario: Conceptos Clave de Redes Neuronales y LLMs 

**Pesos, Sesgos y Capas Lineales (Weights, Biases & Linear Layers)**
*   **Relación con tu código:**
    *   Los pesos se inicializan como números aleatorios pequeños que representan la capacidad del modelo[cite: 1].
    *   En la función lineal, los datos de entrada se multiplican como un vector y una matriz para proyectar resultados[cite: 1].
    *   En PyTorch, estos parámetros se manejan dentro de módulos como `nn.Linear` que pueden tener la opción `bias=False`[cite: 2].
*   **Videos de 3Blue1Brown recomendados:**
    1. *But what is a neural network? | Deep learning, chapter 1* (Explica qué es un parámetro y cómo las capas construyen conocimiento).

**Funciones de Activación (ReLU y GELU)**
*   **Relación con tu código:**
    *   La activación ReLU se implementa evaluando si el valor de los datos es mayor que cero[cite: 1].
    *   En el modelo escalado, la clase de mini-red `MLP` usa `nn.GELU()` para generar suavidad en el entrenamiento[cite: 2].
*   **Videos de 3Blue1Brown recomendados:**
    1. *But what is a neural network? | Deep learning, chapter 1* (Al final del capítulo se detalla cómo ReLU reemplazó a las funciones Sigmoides para ser más fácil de entrenar).

**Función de Pérdida o Coste (Cross-Entropy Loss)**
*   **Relación con tu código:**
    *   La pérdida se calcula penalizando al modelo en proporción a lo poco probable que consideró el token correcto[cite: 1].
    *   En la primera implementación, esto equivale matemáticamente a calcular el logaritmo negativo sobre las probabilidades[cite: 1].
    *   En la implementación avanzada, la función `F.cross_entropy` maneja la penalización aplicando internamente las operaciones softmax y log[cite: 2].
*   **Videos de 3Blue1Brown recomendados:**
    1. *Gradient descent, how neural networks learn | Deep learning, chapter 2* (Explica el concepto universal de minimizar la "falta de precisión").

**Optimizadores y Descenso de Gradiente (Adam / AdamW)**
*   **Relación con tu código:**
    *   Adam adapta los tamaños de paso calculando una media móvil del gradiente y otra media de su cuadrado[cite: 1].
    *   El optimizador `torch.optim.AdamW` en el segundo script separa las matrices de pesos para aplicarles regularización mediante decaimiento[cite: 2].
*   **Videos de 3Blue1Brown recomendados:**
    1. *Gradient descent, how neural networks learn | Deep learning, chapter 2* (Explicación visual de cómo el algoritmo baja "la colina" del error).

**Retropropagación (Backpropagation)**
*   **Relación con tu código:**
    *   El modelo de Python puro llama a la función `backward()` para propagar el gradiente empezando con un valor de uno[cite: 1].
    *   El modelo en PyTorch también ejecuta la retropropagación llamando a `backward()` después de escalar la pérdida en la GPU[cite: 2].
*   **Videos de 3Blue1Brown recomendados:**
    1. *What is backpropagation really doing? | Deep learning, chapter 3* (Explica de forma intuitiva cómo se ajusta la red).

**Autograd y la Regla de la Cadena (Chain Rule)**
*   **Relación con tu código:**
    *   El objeto `Value` actúa como un nodo que almacena operandos hijos y aplica derivadas locales[cite: 1].
    *   Las dependencias topológicas del grafo computacional se resuelven de las hojas a la raíz y propagan los valores usando la regla de la cadena[cite: 1].
*   **Videos de 3Blue1Brown recomendados:**
    1. *Backpropagation calculus | Deep learning, chapter 4* (Profundiza en la matemática exacta de la regla de la cadena, ideal para entender tu clase `Value`).

**Embeddings (Posición y Tokens)**
*   **Relación con tu código:**
    *   El token se mapea a un vector inicial sumando su embedding de significado aprendido y su embedding de posición espacial[cite: 1].
    *   El código de PyTorch emplea el objeto `nn.Embedding` para transformar el tamaño del vocabulario y del bloque a dimensiones continuas[cite: 2].
*   **Videos de 3Blue1Brown recomendados:**
    1. *Attention in neural networks | Deep learning, chapter 5* (Explica el concepto de espacios latentes y vectores de palabras).

**Atención (Queries, Keys, Values)**
*   **Relación con tu código:**
    *   Cada cabeza procesa independientemente la información generando vectores Query, Key y Value para realizar una combinación ponderada en la salida[cite: 1].
    *   La versión causal en PyTorch crea una máscara que impide a la atención mirar hacia adelante[cite: 2].
    *   Esta atención multi-cabeza se optimiza enormemente usando `scaled_dot_product_attention` internamente[cite: 2].
*   **Videos de 3Blue1Brown recomendados:**
    1. *Attention in neural networks | Deep learning, chapter 5* (Explicación magistral sobre qué pregunto, qué ofrezco y qué entrego).

cite1: -> microgpt.py
cite2: -> nanogpt_es.py
