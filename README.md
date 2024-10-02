<h1 align="center">🌟 Práctica 3 - Visión por Computador (Curso 2024/2025)</h1>

<img align="left" width="200" height="180" src="https://i.imgur.com/PPxO7nw.gif"></a>
Se han completado todas las tareas solicitadas de la **Práctica 3** para la asignatura **Visión por Computador**. Detección de formas.

*Trabajo realizado por*:

[![GitHub](https://img.shields.io/badge/GitHub-Heliot%20J.%20Segura%20Gonzalez-orange?style=flat-square&logo=github)](https://github.com/kratoscordoba7)

[![GitHub](https://img.shields.io/badge/GitHub-Alejandro%20David%20Arzola%20Saavedra%20-black?style=flat-square&logo=github)](https://github.com/AlejandroDavidArzolaSaavedra)

## 🛠️ Librerías Utilizadas

[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy)](Link_To_Your_NumPy_Page)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23FD8C00?style=for-the-badge&logo=opencv)](Link_To_Your_OpenCV_Page)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%43FF6400?style=for-the-badge&logo=matplotlib&logoColor=white)](Link_To_Your_Matplotlib_Page)


---
## 🚀 Cómo empezar

Para comenzar con el proyecto, sigue estos pasos:

> [!NOTE]  
> Debes de situarte en un environment configurado como se definió en el cuaderno de la práctica de [otsedom](https://github.com/otsedom/otsedom.github.io/blob/main/VC/P1/README.md#111-comandos-basicos-de-anaconda).

### Paso 1: Abrir VSCode y situarse en el directorio:
   
   `C:\Users\TuNombreDeUsuario\anaconda3\envs\VCP3
   
### Paso 2: Clonar y trabajar en el proyecto localmente (VS Code)
1. **Clona el repositorio**: Ejecuta el siguiente comando en tu terminal para clonar el repositorio:
   ```bash
   git clone https://github.com/kratoscordoba7/VCP3.git
   ```
2. Una vez clonado, todos los archivos han de estar situado en el environment del paso 1

### Paso 3: Abrir Anaconda prompt y activar el envioroment:
   ```bash
   conda activate NombreDeTuEnvironment
   ```
Tras estos pasos debería poder ejecutar el proyecto localmente

---

<h2 align="center">📋 Tareas</h2>

### Tarea 1 Detectar monedas

En esta práctica el objetivo es extraer información geométrica de los objetos presentes en una imagen. Para ello, se presentan ejemplos haciendo uso de detección de contornos y cálculo de la transformada de Hough para la detección de formas circulares.
En una primera tarea se asume que todos los objetos de interés en la imagen son circulares, en concreto monedas de la UE. Tras mostrar diversas aproximaciones para obtener sus contornos, el reto o tarea consiste en determinar la cantidad de dinero presente en la imagen.  


### Tarea 2 Detectar microplasticos

Para la segunda tarea, se proporcionan tres subimágenes de tres clases de microplásticos recogidos en playas canarias. La tarea propuesta consiste en determinar patrones en sus características geométricas que puedan permitir su clasificación en dichas imágenes y otras. Como fuente de partida, se proporciona enlace al trabajo [SMACC: A System for Microplastics Automatic Counting and Classification](https://doi.org/10.1109/ACCESS.2020.2970498) en el que se adoptan algunas propiedades geométricas para dicho fin. De forma resumida, las características geométricas utilizadas en dicho trabajo fueron:

- Área en píxeles
- Perímetro en píxeles
- Compacidad (relación del cuadrado del perímetro con el área)
- Relación del área con la del contenedor
- Relación del ancho y el alto del contenedor
- Relación entre los ejes de la elipse ajustada
- Definido el centroide, relación entre las distancias menor y mayor al contorno





---

> [!IMPORTANT]  
> Los archivos presentados aquí son una modificación de los archivos originales de [otsedom](https://github.com/otsedom/otsedom.github.io/tree/main/VC).



---

## 📚 Bibliografía

1. [Opencv](https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html)
2. [Mondrian](https://www3.gobiernodecanarias.org/medusa/ecoescuela/sa/2017/04/17/descubriendo-a-mondrian/)
3. [Marilyn POP ART](https://temasycomentariosartepaeg.blogspot.com/p/autor-andy-warhol-1928-1987-titulo.html)

---

**Universidad de Las Palmas de Gran Canaria**  

EII - Grado de Ingeniería Informática  
Obra bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional

---
