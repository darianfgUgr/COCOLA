# CoCoLa & Open-CoCoLa: Pipelines de Etiquetado Conceptual Guiado y de Conjunto Abierto

Este repositorio implementa dos pipelines complementarios para anotación conceptual de objetos y partes utilizando modelos multimodales alojados en **Ollama**:

1. **CoCoLa (Closed-Set Concept Labeling)**  
   Selección cerrada de categorías mediante prompting restrictivo y diccionarios jerárquicos de conceptos.

2. **Open-CoCoLa (Open-Set Concept Labeling)**  
   Etiquetado libre seguido de proyección del concepto mediante **FAISS + embeddings**, permitiendo operar en escenarios *open-set*.

Ambos métodos están diseñados para experimentación *zero-shot* con modelos multimodales (Visual-Language Models, VLMs) y para análisis de granularidad múltiple (objetos y partes).

## Inserción de imágenes
Las imágenes proporcionadas deben colocarse en `examples/`:

- `examples/open_cocola_diagram.png`  
- `examples/cocola_diagram.png`

(En el README final se usarán marcadores como:  
`![CoCoLa Pipeline](examples/cocola_diagram.png)`  
`![Open-CoCoLa Pipeline](examples/open_cocola_diagram.png)`)

## Dependencias principales

```
pip install -r requirements.txt
```

## Ejecución CoCoLa

```
python cocola/cocola.py     --model "llava:7b"     --port 11434     --labels data/concepts.json     --input_json data/json/     --input_images data/images/     --output results/
```

## Construcción FAISS (Open-CoCoLa)

```
python open_cocola/embeddings_builder.py data/classes.txt
```

## Ejecución Open-CoCoLa

```
python open_cocola/open_cocola.py     --model "llava:7b"     --port 11434     --embed_model "nomic-embed-text:latest"     --concepts data/concepts.json     --faiss_index data/concepts_index.faiss     --faiss_metadata data/concepts_metadata.pkl     --input_json data/json/     --input_images data/images/     --output results/
```

