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

const MOCKED_VAGAS: Vaga[] = [
    { vaga: 1, status: 0 },
    { vaga: 2, status: 1 },
    { vaga: 3, status: 0 },
    { vaga: 4, status: 1 },
    { vaga: 5, status: 0 },
    { vaga: 6, status: 1 },
    { vaga: 7, status: 0 },
    { vaga: 8, status: 1 },
    { vaga: 9, status: 0 },
    { vaga: 10, status: 1 },
    { vaga: 11, status: 0 },
    { vaga: 12, status: 1 },
    { vaga: 13, status: 0 },
    { vaga: 14, status: 1 },
    { vaga: 15, status: 0 },
    { vaga: 16, status: 1 },
    { vaga: 17, status: 0 },
    { vaga: 18, status: 1 },
    { vaga: 19, status: 0 },
    { vaga: 20, status: 1 },
]

function divideVagasIntoColumns(vagas: Vaga[]): Vaga[][] {
    const columns: Vaga[][] = []
    for (let i = 0; i < vagas.length; i += 5) {
        columns.push(vagas.slice(i, i + 5))
    }
    return columns
}

function calculateOccupancy(vagas: Vaga[]) {
    const total = vagas.length
    const occupied = vagas.filter(v => v.status === 1).length
    const free = total - occupied
    const occupancy = total > 0 ? Math.round((occupied / total) * 100) : 0
    
    return { total, occupied, free, occupancy }
}

function createStatItem(
    id: string,
    label: string,
    value: string,
    helper: string,
    variant?: 'success' | 'danger' | 'warning' | 'info'
): StatItem {
    return { id, label, value, helper, variant }
}

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
    // Obtém vagas do WebSocket ou usa dados de exemplo
    const { vagas, connected, lastUpdated } = useVagas()

    const hasRealtimeData = vagas.length > 0
    const dataset = hasRealtimeData ? vagas : MOCKED_VAGAS

    const vagasColumns = useMemo(() => 
        divideVagasIntoColumns(dataset), 
        [dataset]
    )
    
    const { total, occupied, free, occupancy } = calculateOccupancy(dataset)

    const stats = useMemo(() => 
        generateStats(total, occupied, free, occupancy, hasRealtimeData),
        [total, occupied, free, occupancy, hasRealtimeData, connected, lastUpdated]
    )

    return (
        <div className="park-root">
            <main className="park-main">
                <div className="park-container">
                    <div className="park-dashboard-top">
                        <div className="park-headline">
                            <div className="park-title">SmartPark Dashboard</div>
                        </div>
                        <div className="park-top-right">
                            <StatusLegend />
                            <StatusBadge connected={connected} />
                        </div>
                    </div>

                    <StatsBar items={stats} />

                    <div className="park-layout">
                        {vagasColumns.map((column, index) => (
                            <div key={index} className="park-column-section">
                                <LootGroup position="column">
                                    {column.map((v) => (
                                        <Loot key={v.vaga} vaga={v.vaga} status={v.status} />
                                    ))}
                                </LootGroup>
                            </div>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    );
}
