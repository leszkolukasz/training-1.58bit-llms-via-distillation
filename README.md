## Training 1.58bit LLMs via Distillation

### Training

To train, execute the following command:

```sh
uv run python -m src.main fit
  --model <model_name>
  --data <dataset_name>
  --model.quantization <quantization>
  --model.bitlinear_implementation <impl>
  --model.loss_function <loss>
  --ckpt_path <path_to_checkpoint> (optional, to resume training)
```

Available options are:
- `<model_name>`: `ExampleModel`, `QuantizedQwenModel`, `QuantizedSmolModel`
- `<dataset_name>`: `WikiText2DataModule`, `AmberDataModule`
- `<quantization>`: `1b`, `1_58b`, `1b_no_shift`
- `<impl>`: `FBI`, `OneBit`, `BitNet`
- `<loss>`: `CrossEntropy`, `KL`, `CAKL`, `Wasserstein`

#### Example

```sh
uv run python -m src.main fit \
  --model ExampleModel
  --data WikiText2DataModule
  --trainer.max_epochs 1
```

```sh
uv run python -m src.main fit \
  --model QuantizedQwenModel \
  --data AmberDataModule \
  --trainer.max_epochs 1 \
  --model.quantization 1b \
  --model.bitlinear_implementation FBI \
  --model.loss_function CrossEntropy
```

### Chat

To chat with the trained model, execute the following command:

```sh
uv run python -m src.main chat \
  --model <model_name> \
  --ckpt_path <path_to_checkpoint>
  --simple (optional, does not apply chat template)
```

#### Example

```sh
uv run python -m src.main chat \
  --model QuantizedSmolModel \
  --ckpt_path "./mlruns/807781062401719461/a9b88f905cd94554ab873f37c21c25be/checkpoints/epoch=0-step=7.ckpt"
```