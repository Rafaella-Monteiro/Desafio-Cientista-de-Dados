# Desafio Cientista de Dados - Previsão de Nota IMDB

## Instalação e Configuração

Para configurar e executar este projeto localmente, siga estas etapas:

# 1. Clonar o Repositório

Clone este repositório para o seu ambiente local:

```bash
git clone https://github.com/Rafaella-Monteiro/Desafio-Cientista-de-Dados.git
cd Desafio-Cientista-de-Dados

```

# 2. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

# 3. Execute a análise:
```bash
python desafio.py
```
# 4. Carregar o modelo salvo:
## Você pode carregar o modelo previamente treinado usando o seguinte código Python:

import pickle

# Carregar o modelo salvo: 
with open('modelo_imdb.pkl', 'rb') as file:
    model = pickle.load(file)


