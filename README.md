---
title: Coredrill — Rhizome Text Classifier
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: Geometry-native multilingual text classification, 238-node rhizomatic taxonomy
---

# Coredrill API

**Geometry-native multilingual text classification over a 238-node rhizomatic taxonomy.**

No trained classifiers. Classification by the geometric structure of the [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embedding manifold.

## Interactive docs

Once the Space is running, open:

```
https://your-username-coredrill.hf.space/docs
```

This is the auto-generated Swagger UI. You can send test requests directly from the browser.

## Quick test

```bash
curl -X POST https://your-username-coredrill.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Algebra is a branch of mathematics that generalizes arithmetic."}'
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/info` | Taxonomy info |
| POST | `/predict` | Classify one text |
| POST | `/predict/batch` | Classify up to 64 texts |
| GET | `/taxonomy/nodes` | Browse the 238-node tree |
| GET | `/taxonomy/node/{name}` | Node detail |

## Source code

[github.com/your-username/coredrill](https://github.com/your-username/coredrill)

## Paper

*Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis* (Lucas, 2026)
