# digits-mnist-app

Aplicación Streamlit para visualizar experimentos de hiperparámetros sobre el dataset MNIST de dígitos manuscritos.

## Páginas

| Página | Descripción |
|--------|-------------|
| 📊 Exploración del Dataset | Muestras por dígito, distribución de clases, intensidad media |
| 🏆 Comparativa de Modelos | Tabla y gráficas destacando el mejor, peor e intermedio |
| 📈 Curvas de Entrenamiento | Curvas de accuracy y loss por experimento |
| 🔮 Inferencia en Vivo | Predicción sobre imágenes del conjunto de test |

## Experimentos incluidos

10 modelos CNN variando: número de neuronas/capas, épocas, learning rate, batch size y técnicas avanzadas (Dropout, BatchNorm, L2, EarlyStopping).

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

App desplegada en: https://digits-mnist-app.streamlit.app
