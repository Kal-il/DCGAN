# DCGAN

## Configurando o Ambiente

### Windows:

1. Intale o Conda:
2. Crie um ambiente Conda:
   ```bash
   conda create -n dcgan_env python=3.8 -y
   ```

### Estrutura do projeto:
``` plaintext
DCGAN/
├── 102flowers/
├     └── jpg/
├── experiments/
├     ├── dcgan_experiment_1/
├     ├     ├── checkpoints/
├     ├     ├── images/
├     ├     └── training_logs.csv
├     ├── dcgan_experiment_2/
├     └── ...
├── config.py
├── dataset.py
├── model.py
├── train.py
├── generate_image.py
├── utils.py
├── requirements.txt
└── README.md
```