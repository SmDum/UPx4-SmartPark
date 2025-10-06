import styles from './styles.module.css'

export type StatItem = {
  id: string
  label: string
  value: string
  helper?: string
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
}

type StatsBarProps = {
  items: StatItem[]
}

export function StatsBar({ items }: StatsBarProps) {
  return (
    <section className={styles.bar}>
      {items.map(({ id, label, value, helper, variant = 'default' }) => (
        <article key={id} className={`${styles.card} ${styles[variant]}`}>
          <span className={styles.label}>{label}</span>
          <span className={styles.value}>{value}</span>
          {helper ? <span className={styles.helper}>{helper}</span> : null}
        </article>
      ))}
    </section>
  )
}
