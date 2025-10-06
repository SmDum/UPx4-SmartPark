import { useEffect, useRef, useState } from 'react'
import type { Vaga } from '../types/park.type'

function listasIguais(a: Vaga[], b: Vaga[]) {
  if (a === b) return true
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    const vagaA = a[i]
    const vagaB = b[i]
    if (vagaA.vaga !== vagaB.vaga || vagaA.status !== vagaB.status) {
      return false
    }
  }
  return true
}

export function useVagas() {
  const wsUrl = 'ws://localhost:8000/ws'
  const wsRef = useRef<WebSocket | null>(null)
  const [vagas, setVagas] = useState<Vaga[]>([])
  const [connected, setConnected] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  useEffect(() => {
    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
      }

      ws.onmessage = (ev) => {
        try {
          const payload = JSON.parse(ev.data)
          if (Array.isArray(payload?.status)) {
            let houveAlteracao = false
            const proxVagas = payload.status as Vaga[]
            setVagas((anterior) => {
              if (listasIguais(anterior, proxVagas)) {
                return anterior
              }
              houveAlteracao = true
              return proxVagas
            })

            if (houveAlteracao) {
              setLastUpdated(new Date())
            }
          }
        } catch {
          // Silencia payloads inválidos para manter o hook resiliente
        }
      }

      const handleDisconnect = () => {
        setConnected(false)
      }

      ws.onclose = handleDisconnect
      ws.onerror = handleDisconnect

      return () => {
        ws.close()
        wsRef.current = null
      }
    } catch (e) {
      setConnected(false)
    }
    return undefined
  }, [wsUrl])

  return { vagas, connected, lastUpdated }
}
