export type Vaga = { vaga: number; status: 0 | 1 }

export type LootGroupProps = {
    children?: React.ReactNode;
    position?: 'left-top' | 'left-bottom' | 'right-top' | 'right-bottom' | 'left' | 'right';
}