import type { Vaga } from '../../types/park.type';
import styles from './styles.module.css';



export function Loot({ vaga, status }: Vaga) {
    const getClassName = () => {
        if (status === 1) return `${styles.loot} ${styles.ocupada}`
        return `${styles.loot} ${styles.livre}`
    }

    return (
        <div className={styles.lootContainer}>
            <div className={getClassName()}>
                <span className={styles.lootNumber}>Vaga - {vaga}</span>
            </div>
        </div>
    );
}
