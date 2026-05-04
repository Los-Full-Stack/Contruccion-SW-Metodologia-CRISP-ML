# 🎓 Sistema de Alerta Temprana para la Deserción Estudiantil (MLP)

**Proyecto de Construcción de Software - Universidad Continental (Cusco, 2026)**[cite: 1]

Este repositorio contiene la implementación de un modelo de Inteligencia Artificial (Perceptrón Multicapa) diseñado para detectar tempranamente a estudiantes en riesgo de abandono universitario, abordando la problemática del 15% de deserción en el primer ciclo[cite: 1].

---

## 🚀 Enlaces del Proyecto

*   **Aplicación Web (Streamlit):** [SAT Inteligencia Predictiva](https://interfacepy-rtjmmzo9omyuwkznyqw2tk.streamlit.app/)
*   **Bitácora Interactiva y Código (Google Colab):** [Ver Notebook](https://colab.research.google.com/drive/1j1lqYIZCL0mdm0AoLcOu2SZSqthV9XA1?usp=sharing)
*   **Repositorio GitHub:** [Los-Full-Stack/Contruccion-SW-Metodologia-CRISP-ML](https://github.com/Los-Full-Stack/Contruccion-SW-Metodologia-CRISP-ML)

---

## 🧠 Arquitectura y Metodología

El proyecto fue desarrollado bajo la metodología **CRISP-ML**, abarcando desde la comprensión del negocio hasta el despliegue del modelo.

### Stack Tecnológico y Técnicas Utilizadas
*   **Lenguaje:** Python
*   **Preprocesamiento:** `StandardScaler` (Normalización Z-score)
*   **Balanceo de Clases:** `SMOTE` (Synthetic Minority Over-sampling Technique) para pasar de un entorno desbalanceado a una proporción 50/50
*   **Modelo de Machine Learning:** `MLPClassifier` (Perceptrón Multicapa) de Scikit-Learn
*   **Arquitectura de Red:** 64-32-1 (Dos capas ocultas de 64 y 32 neuronas)
*   **Funciones de Activación:** ReLU (capas internas) y Sigmoide (capa de salida)
*   **Optimizador:** Adam

---

## 📊 Resultados del Modelo

Dado que el problema requiere priorizar la detección de todos los posibles desertores, se optimizó el modelo para minimizar los *Falsos Negativos*. 

*   **Recall (Sensibilidad) para la clase "En Riesgo":** 0.98 (Detecta al 98% de desertores reales)
*   **Precisión:** 0.80
*   **F1-Score:** 0.88
*   **Average Precision (Curva PR):** 0.93
*   **Pérdida Final (Loss):** 0.0050 (Convergencia en 244 iteraciones)

La Matriz de Confusión demostró que, de una muestra de 175 estudiantes en riesgo, el modelo identificó correctamente a 171, limitando el error a solo 4 casos (falsos negativos).

---

## 🛠️ Cómo ejecutar el proyecto (Localmente)

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/Los-Full-Stack/Contruccion-SW-Metodologia-CRISP-ML.git](https://github.com/Los-Full-Stack/Contruccion-SW-Metodologia-CRISP-ML.git)
   cd Contruccion-SW-Metodologia-CRISP-ML