import { useMemo } from 'react'
import { LootGroup } from "./components/LootGroup";
import { Loot } from "./components/Loot";
import { StatsBar } from './components/StatsBar'
import type { StatItem } from './components/StatsBar'
import { StatusBadge } from './components/StatusBadge'
import { StatusLegend } from './components/StatusLegend'
import './Park.css';
import { useVagas } from './hooks/useVagas'
import type { Vaga } from './types/park.type'

const MOCKED_VAGAS_LEFT: Vaga[] = [
    { vaga: 1, status: 0 },
    { vaga: 2, status: 1 },
    { vaga: 3, status: 0 },
    { vaga: 4, status: 1 },
    { vaga: 5, status: 0 },
]

const MOCKED_VAGAS_RIGHT: Vaga[] = [
    { vaga: 6, status: 1 },
    { vaga: 7, status: 0 },
    { vaga: 8, status: 1 },
    { vaga: 9, status: 0 },
    { vaga: 10, status: 1 },
]

const MOCKED_VAGAS = [...MOCKED_VAGAS_LEFT, ...MOCKED_VAGAS_RIGHT]

// Divide vagas entre lado esquerdo (1-5) e direito (6-10)
function splitVagasBySide(vagas: Vaga[]): [Vaga[], Vaga[]] {
    const left = vagas.filter(v => v.vaga <= 5)
    const right = vagas.filter(v => v.vaga > 5)
    return [left, right]
}

// Calcula estatísticas de ocupação
function calculateOccupancy(vagas: Vaga[]) {
    const total = vagas.length
    const occupied = vagas.filter(v => v.status === 1).length
    const free = total - occupied
    const occupancy = total > 0 ? Math.round((occupied / total) * 100) : 0
    
    return { total, occupied, free, occupancy }
}

// Cria item de estatística com formatação
function createStatItem(
    id: string,
    label: string,
    value: string,
    helper: string,
    variant?: 'success' | 'danger' | 'warning' | 'info'
): StatItem {
    return { id, label, value, helper, variant }
}

// Formata data no padrão brasileiro
function formatTime(date: Date | null): string {
    if (!date) return 'Aguardando dados'
    
    return new Intl.DateTimeFormat('pt-BR', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    }).format(date)
}

// Gera todas as estatísticas do dashboard
function generateStats(
    total: number,
    occupied: number,
    free: number,
    occupancy: number,
    hasRealtimeData: boolean,
): StatItem[] {
    const sourceLabel = hasRealtimeData ? 'Dados em tempo real' : 'Dados simulados'
    const occupancyVariant = occupancy >= 75 ? 'danger' : occupancy >= 50 ? 'warning' : 'info'
    const freeVariant = free > 0 ? 'success' : 'danger'
    
    return [
        createStatItem('total', 'Total de vagas', String(total), sourceLabel),
        createStatItem('occupied', 'Ocupadas', String(occupied), `${occupancy}% de ocupação`, occupancyVariant),
        createStatItem('free', 'Disponíveis', String(free), free > 0 ? 'Há vagas liberadas' : 'Sem vagas livres agora', freeVariant),
    ]
}

export function Park() {
    const { vagas, connected, lastUpdated } = useVagas()

    const hasRealtimeData = vagas.length > 0
    const dataset = hasRealtimeData ? vagas : MOCKED_VAGAS

    const [leftSide, rightSide] = useMemo(() => 
        splitVagasBySide(hasRealtimeData ? vagas : []), 
        [vagas, hasRealtimeData]
    )

    const displayedLeft = hasRealtimeData ? leftSide : MOCKED_VAGAS_LEFT
    const displayedRight = hasRealtimeData ? rightSide : MOCKED_VAGAS_RIGHT
    
    const { total, occupied, free, occupancy } = calculateOccupancy(dataset)

    const stats = useMemo(() => 
        generateStats(total, occupied, free, occupancy, hasRealtimeData),
        [total, occupied, free, occupancy, hasRealtimeData, connected, lastUpdated]
    )

    return (
        <div className="park-root">
            <header className="park-header">
                <h1>UPx4 - SmartPark</h1>
                <StatusBadge connected={connected} />
            </header>

            <main className="park-main">
                <div className="park-container">
                    <div className="park-dashboard-top">
                        <div className="park-headline">
                            <div className="park-title">SmartPark Dashboard</div>
                        </div>
                        <StatusLegend />
                    </div>

                    <StatsBar items={stats} />

                    <div className="park-layout">
                        <div className="park-left-section">
                            <LootGroup position="left">
                                {displayedLeft.map((v) => (
                                    <Loot key={v.vaga} vaga={v.vaga} status={v.status} />
                                ))}
                            </LootGroup>
                        </div>

                        <div className="park-right-section">
                            <LootGroup position="right">
                                {displayedRight.map((v) => (
                                    <Loot key={v.vaga} vaga={v.vaga} status={v.status} />
                                ))}
                            </LootGroup>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
