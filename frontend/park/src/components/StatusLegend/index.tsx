import styles from './styles.module.css'

export function StatusLegend() {
  return (
    <ul className={styles.legend} >
      <li className={styles.item}>
        <span className={`${styles.dot} ${styles.free}`} />
        <span className={styles.text}>
          <strong>Disponível</strong>
          <span className={styles.helper}>Vaga liberada</span>
        </span>
      </li>
      <li className={styles.item}>
        <span className={`${styles.dot} ${styles.occupied}`} />
        <span className={styles.text}>
          <strong>Ocupada</strong>
          <span className={styles.helper}>Vaga em uso</span>
        </span>
      </li>
    </ul>
  )
}
