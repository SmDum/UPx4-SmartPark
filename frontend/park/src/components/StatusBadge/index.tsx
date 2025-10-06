import styles from './styles.module.css'

type StatusBadgeProps = {
  connected: boolean
}

export function StatusBadge({ connected }: StatusBadgeProps) {
  return (
    <div
      className={styles.badge}
      data-connected={connected}
      role="status"
    >
      <span className={styles.dot} />
      <div className={styles.text}>
        <span className={styles.caption}>Conexão</span>
        <span className={styles.state}>{connected ? 'Online' : 'Offline'}</span>
      </div>
    </div>
  )
}
