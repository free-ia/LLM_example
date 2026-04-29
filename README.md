# LLM_example — Un GPT en Python puro, explicado para profanos

Un modelo GPT (Large Language Model) **completo, funcional y entrenable**, escrito en menos de 200 líneas de Python sin dependencias externas. Sirve como material didáctico para entender desde cero cómo funcionan los modelos detrás de ChatGPT, Claude o Gemini.

## Contenido

- **`microgpt.py`** — el código del modelo (entrenamiento + inferencia) con comentarios extensos en español que explican cada pieza: dataset, tokenizer, autograd, atención multi-cabeza, MLP, optimizador Adam y muestreo.
- **`EXPLICACION.md`** — documento didáctico complementario con diagramas (Mermaid + ASCII) que cubre los conceptos fundamentales: tokenización, embeddings, atención Q/K/V, bloques transformer, función de pérdida, backpropagation, leyes de escalado, etc.

## Cómo ejecutarlo

```bash
python microgpt.py
```

No requiere `pip install` de nada — sólo Python 3. La primera ejecución descarga automáticamente el dataset (lista de nombres) y entrena un mini-GPT durante 1000 pasos. Al final imprime 20 nombres "alucinados" por el modelo.

## Fuente original

El código está basado en el gist público de **Andrej Karpathy**:

> https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Esta versión no modifica la lógica del original: **toda la aportación de este repositorio es documentación adicional en español** (comentarios en el código y el documento `EXPLICACION.md`) para hacerlo accesible a personas sin formación técnica avanzada.

## Sobre Andrej Karpathy

[Andrej Karpathy](https://karpathy.ai/) es una de las figuras más influyentes en la divulgación moderna de la inteligencia artificial:

- **Miembro fundador de OpenAI** (2015), donde participó en los primeros desarrollos que llevarían a GPT.
- **Director de IA en Tesla** (2017–2022), responsable del equipo de visión por computador del Autopilot.
- Volvió brevemente a OpenAI en 2023.
- En 2024 fundó **[Eureka Labs](https://eurekalabs.ai/)**, una empresa centrada en educación AI-nativa.
- Doctor en Stanford bajo la dirección de Fei-Fei Li, donde co-creó el curso **CS231n** (visión por computador), referencia mundial.

Es especialmente conocido por su capacidad pedagógica: proyectos como **micrograd**, **makemore**, **nanoGPT** y la serie de YouTube **"Neural Networks: Zero to Hero"** han enseñado a una generación entera de ingenieros cómo funcionan las redes neuronales modernas, construyéndolas desde cero. Este `microgpt.py` es un ejemplo perfecto de esa filosofía: el algoritmo completo, sin abstracciones que escondan lo esencial.

## Licencia

El código original es de Andrej Karpathy (consultar el [gist fuente](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) para los términos). La documentación añadida en este repositorio se ofrece libremente con fines educativos.
