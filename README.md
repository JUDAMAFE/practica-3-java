#  Práctica 3: Programación Orientada a Objetos

**Estudiantes:** Juan Diego Parra, Juan Daniel Martinez    
**Fecha de entrega:** Noviembre 10, 2025  



---

##  Descripción del proyecto

El programa implementa una clase llamada **`LinearRegression`**, la cual modela la ecuación de una regresión lineal.


El modelo se entrena usando la ecuación normal, calcula los pesos y el bias, realiza predicciones, evalúa el error mediante el coeficiente de determinación R², y permite escalar los datos.

Se trabajó con dos conjuntos de datos:

| Dataset | Tipo de regresión | Descripción |
|----------|------------------|--------------|
| `Ice_cream_selling_data.csv` | Simple | Relación entre temperatura y ventas de helado |
| `student_exam_scores.csv` | Múltiple | Relación entre horas de estudio, asistencia y nota final |

---

##  Estructura de la clase

```java
class LinearRegression {
    private double[] weights; // Pesos del modelo
    private double bias;      // Término independiente

    public void fit(double[][] X, double[] y);       // Entrena el modelo
    public double[] predict(double[][] X);           // Predice resultados
    public double score(double[] y_true, double[] y_pred); // Calcula R²
    public double[][] scaleData(double[][] X);       // Escala los datos (Min-Max)
}
```

---

##  Entrenamiento y prueba

El modelo divide los datos en 80% para entrenamiento y 20% para prueba:

```java
int[][] idx = splitIndices(X.length, 0.8, 42L);
```

Esto evita el sobreentrenamiento y permite evaluar la capacidad de generalización.

---

##  Resultados obtenidos

###  Ice Cream (Regresión simple)
```
Pesos (ice): [-9.74]
Bias  (ice): 20.98
R² (ice, test): -0.037
```
 R² negativo debido a que la relacion es no lineal. la relacion es curva (parabola) de modo que el modelo de regresion lineal no es adecuado para expresarla.

###  Student Exam Scores (Regresión múltiple)
```
Pesos (stu): [17.14, 4.49, 5.70, 9.70]
Bias  (stu): 15.65
R² (stu, test): 0.8479
```
Buen ajuste, el modelo predice correctamente con un R² ≈ 0.85.

---

##  Conclusiones
1. El modelo lineal solo funciona si existe una relación lineal entre las variables.  
2. El dataset de helados muestra una relación curva (no lineal), por eso el ajuste es bajo.  
3. El modelo de estudiantes obtuvo un R² = 0.84, mostrando un buen desempeño.  

---

## Ejecución en Visual Studio Code
1. Guardar los archivos:
   ```
   LinearRegression.java
   Ice_cream_selling_data.csv
   student_exam_scores.csv
   ```
2. Compilar y ejecutar.
   

---

##  Estructura del repositorio
```
Prac3-OOP/
│
├── LinearRegression.java
├── Ice_cream_selling_data.csv
├── student_exam_scores.csv
└── README.md
