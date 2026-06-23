# Predicción de ascensos y descensos en LaLiga Hypermotion

**Trabajo Fin de Máster** — Pedro Sarmiento
Máster Universitario en Inteligencia Artificial · Universidad Internacional de La Rioja (UNIR) · 2026

---

## Resumen

Comparativa empírica de siete enfoques predictivos para los resultados de LaLiga Hypermotion (Segunda División española), sobre 16 temporadas (2010/11 – 2025/26) y con el log-loss del mercado de apuestas (~1.03) como referencia a batir.

Sobre las predicciones partido a partido se ejecutan después simulaciones de Monte Carlo (100 000 iteraciones por temporada) que estiman, para cada equipo, la probabilidad de ascenso directo, ascenso por playoff, permanencia y descenso.

Modelos comparados:

1. Doble Poisson
2. Regresión logística
3. Random Forest
4. XGBoost
5. LightGBM
6. MLP

## Reproducción

### Requisitos

- Python 3.11 o 3.12 (recomendado vía `pyenv`).
- ~2 GB de espacio en disco con datos y modelos generados.

### Instalación

```bash
git clone https://github.com/Pedro-Sarmiento/Prediccion-de-ascensos-y-descensos-en-LaLiga-Hypermotion.git
cd Prediccion-de-ascensos-y-descensos-en-LaLiga-Hypermotion

python3.12 -m venv .venv
source .venv/bin/activate

make install-dev    # core + ML + JupyterLab
# make install-all  # incluye PyTorch (notebook 08, pesado)
```

### Ejecución

```bash
make notebook       # abre JupyterLab
```

Ejecutar los notebooks en orden numérico:

1. **`00_eda.ipynb`** — exploración inicial y validación del dataset.
2. **`01_feature_engineering.ipynb`** — construye las 5 familias de features (forma reciente, posición pre-jornada, Elo, valor de mercado, cuotas) y guarda `data/features/dataset.parquet`.
3. **`02` – `07`** — un notebook por modelo: hiperparámetros, entrenamiento, métricas y persistencia.
4. **`08_experimentos.ipynb`** — comparativa entre modelos + simulaciones Monte Carlo + análisis económico.

---

## Datos

| Fuente | Cobertura | Estado en el repo |
|---|---|---|
| **football-data.co.uk** | SP2 2010/11 – 2025/26 (16 temp., 7 337 partidos) | Versionado en `data/raw/footballdata/` |
| **Transfermarkt** — clasificaciones tras cada jornada | 16 temp. × 42 jornadas (672 CSV) | No versionado — obtener manualmente |
| **Transfermarkt** — plantillas con valor de mercado | 16 temp. × ~22 equipos × ~25 jugadores | No versionado — obtener manualmente |

Los datos de Transfermarkt no se versionan por su volumen y por respeto a la fuente original. El notebook `01_feature_engineering.ipynb` documenta las URLs y la estructura esperada para quien desee reconstruirlos.

## Licencia

Repositorio publicado con fines académicos en el contexto de la defensa del Trabajo Fin de Máster. Los datos de football-data.co.uk se utilizan bajo sus condiciones de uso público.

---

## Autor

**Pedro Sarmiento**
Trabajo Fin de Máster — Máster Universitario en Inteligencia Artificial
Universidad Internacional de La Rioja (UNIR), 2026
