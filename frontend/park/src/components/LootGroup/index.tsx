import styles from './styles.module.css';
import type { LootGroupProps } from '../../types/park.type';


export function LootGroup({ children, position }: LootGroupProps) {
    const getPositionClass = () => {
        if (!position) return '';
        return styles[position.replace('-', '')];
    };

    return (
        <div className={`${styles.lootGroupContainer} ${getPositionClass()}`}>
            {children}
        </div>
    );
}
