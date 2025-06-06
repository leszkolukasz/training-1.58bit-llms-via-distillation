## Training 1.58bit LLMs via Distillation

### Example run

```bash
uv run python -m src.main fit --model TestModel --data WikiText2DataModule --trainer.max_epochs 1
```