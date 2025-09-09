# Arquitetura e Decisões de Processo

Data: 2025-09-08

Este documento reúne registros de decisões de arquitetura (ADR) e de processo (PDR) para o projeto UPx4-SmartPark.


## ADR-001: Arquitetura de comunicação em tempo real — FastAPI + Redis (pub/sub) + WebSocket

Data: 2025-09-08

Contexto
- Existe um fluxo de inferência (YOLO) que detecta vagas a partir de frames de câmera; essa informação precisa ser entregue em tempo real ao dashboard UI.
- O diagrama de arquitetura propõe uso de Redis para armazenamento/propagação de estado e FastAPI como ponte de comunicação com clientes WebSocket.

Decisão
- Adotar Redis como mecanismo pub/sub para eventos de atualização de vaga, FastAPI como serviço HTTP/WebSocket que escuta eventos do Redis e publica atualizações ao cliente React via WebSocket.

Racional
- Redis é rápido e adequado para operações simples de SET/GET e pub/sub em tempo real.
- FastAPI fornece endpoints leves, integração com WebSocket e facilidade de implementação em Python existente.
- Separação de responsabilidades: detecção/proc. de imagem atualiza estado; FastAPI entrega ao cliente.

Alternativas consideradas
- Usar apenas WebSocket sem Redis (estado em memória): menos resiliência e dificuldade com múltiplas instâncias.
- Broker dedicado (RabbitMQ/Kafka): maior complexidade e overhead para a necessidade atual.

Consequências
- Facilita entrega em tempo real para múltiplos clientes; permite escalar o processamento e o frontend separadamente.
- Requer deploy e gerenciamento do Redis; cuidado com persistência e milion-scale se o projeto crescer.


Status: accepted

---

## PDR-001: Fluxo runtime (camera -> YOLO -> Redis -> FastAPI -> React)

Data: 2025-09-08

Contexto
- Baseado no diagrama fornecido: camera -> YOLO (detecção) -> processamento -> atualização de vaga (JSON) -> Redis (SET + PUB) -> FastAPI (subscriber/listener) -> WebSocket -> React client.

Processo / passos operacionais
1. A câmera fornece frames ao módulo de inferência (YOLO).
2. O módulo de processamento converte a detecção em (VagaID, Status)
3. O backend publica a atualização no Redis (PUBLISH no canal).
4. Um cliente Redis no FastAPI consome mensagens do Redis e envia eventos via WebSocket para clientes conectados.
5. O cliente React mantém uma conexão WebSocket e atualiza o dashboard em tempo real ao receber eventos.

Critérios de aceitação
- A latência end-to-end (frame -> atualização no cliente) deve ficar dentro dos requisitos do projeto (por exemplo, <1s para detecção e notificação em condições normais).
- Tolerância a reconexões: cliente deve reconectar ao WebSocket e receber estado atual via leitura do Redis ao reconectar.

Status: accepted

---
