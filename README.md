Simulación M/M/1 — Sistema de Transporte Público
¿Alguna vez esperaste el camión y simplemente no cabiste? Eso no es mala suerte. Es matemática.
Este proyecto aplica Teoría de Colas (Investigación de Operaciones) para analizar cuándo y por qué las rutas de transporte público se saturan, usando Python y datos sintéticos con perfiles realistas de demanda.
¿Qué problema resuelve?
Las ciudades invierten en transporte público sin siempre saber en qué rutas y en qué horarios el sistema deja de funcionar bien. 
Este modelo permite identificarlo con matemática, antes de que se convierta en un problema operativo real.

Resultados principales
RutaHora picoEstadoRuta 1 — CentroMañana / Tarde🟢 EstableRuta 2 — UniversidadMañana🔴 Colapso totalRuta 3 — PeriféricoTodo el día🟢 Estable
Hallazgo clave: La Ruta 2 en hora pico matutina recibe más pasajeros de los que puede atender. La cola crece indefinidamente. Con aumentar la frecuencia de 10 a 7 minutos, el problema desaparece casi por completo.

Modelo utilizado: M/M/1
El modelo M/M/1 es el sistema de colas más fundamental:

M — Llegadas aleatorias de pasajeros (proceso de Poisson)
M — Tiempo de servicio aleatorio (distribución exponencial)
1 — Un servidor (el autobús al llegar a la parada)

Métricas calculadas
MétricaSignificado simpleρ = λ/μQué tan ocupado está el sistema (0 = vacío, 1 = colapso)WqCuánto espera un pasajero típico en la filaLqCuántas personas hay formadas en promedioP(W > t)Probabilidad de esperar más de t minutos

