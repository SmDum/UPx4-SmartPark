import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { Park } from './Park'


createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Park />
  </StrictMode>,
)
