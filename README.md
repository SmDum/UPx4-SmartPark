# UPx4-SmartPark

Projeto para detecção e notificação em tempo real do estado de vagas de estacionamento usando YOLO, Redis e FastAPI + WebSocket.

## Instalação (Windows PowerShell)

Abra um PowerShell na raiz do projeto e execute:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuração

- `app.py` atualmente lê a URL do Redis direto do código (`REDIS_URL`). Você pode editar esse valor no arquivo ou definir a variável equivalente no lugar onde sua aplicação procura configuração.
- `main.py` usa a camada de conexão Redis definida em `src/redis/models/connection`.

Recomendações:
- Definir acesso seguro ao Redis e evitar expor credenciais em código (usar variáveis de ambiente ou um gerenciador de segredos).

## Executando

1) Iniciar o servidor FastAPI (WebSocket):

```powershell
# opção 1 (rápida, já presente no app.py)
python app.py

```

2) Iniciar o detector YOLO (publica atualizações no Redis):

```powershell
python main.py
```

3) Conectar um cliente WebSocket (exemplo do console do navegador):

# Execute antes no Console do DevTools -> allow pasting
```javascript

const ws = new WebSocket('ws://localhost:8000/ws');
ws.onopen = () => console.log('WS aberto');
ws.onmessage = e => console.log('Recebeu:', e.data);
```

Formato esperado das mensagens publicadas pelo detector (exemplo):

```json
{"status": [{"vaga": "A1", "status": 1}, ...], "ts": 169...}
```

## Visão geral

Fluxo principal:
- Câmera -> `main.py` (YOLO) -> processamento de vagas
- `main.py` publica atualizações no Redis (PUB)
- `app.py` (FastAPI) subscreve o canal Redis e envia eventos aos clientes via WebSocket
- Frontend (React ou outro) mantém conexão WebSocket e atualiza o dashboard em tempo real

Arquitetura: FastAPI + Redis (pub/sub) para entrega em tempo real; separação de responsabilidades facilita escalar o processamento e o frontend.

## Principais arquivos
- `main.py` — módulo que usa o modelo YOLO para detectar veículos e publica o status das vagas no Redis.
- `app.py` — servidor FastAPI que aceita conexões WebSocket e repassa mensagens do Redis para os clientes.
- `requiriments.txt` — lista de dependências Python.
- `slots.json` — dados de vagas e status de exemplo.

## Requisitos
- Python 3.10+ (recomendado)
- Redis (serviço acessível; o projeto usa pub/sub)

Observação: o arquivo de dependências no repositório está nomeado `requiriments.txt` (ortografia existente). Use esse arquivo ao instalar.


