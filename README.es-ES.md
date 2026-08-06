

# StomataPy es una Colección de Recursos que Habilitan la Segmentación Generalizada de Estomas mediante un Enfoque de Intervención Humana Impulsado por la Comunidad

[![Documentation](https://img.shields.io/badge/📖%20Documentation-blue?style=flat-square&logo=readthedocs&logoColor=white)](https://stomatapy.readthedocs.io/en/latest/)

**Concretamente, nuestros recursos permiten a la comunidad de estudios de estomas contribuir con imágenes conjuntamente, para mejorar de forma iterativa los modelos de código abierto**
- **StomataPy400K**: El conjunto de datos de estomas anotados más grande hasta la fecha (próximamente en código abierto)
- **ISAT-SAM**: Herramienta Interactiva de Anotación de Estomas con Segment Anything Model (ya disponible en código abierto)
- **Modelos StomataPy400K**: una serie de modelos para segmentaciones relacionadas con estomas (próximamente en código abierto)
---

# StomataPy400K: El conjunto de datos de estomas anotados más grande hasta la fecha (próximamente en código abierto)
<img src="asserts/datasets_preview.gif" width="800" height=auto /> </div>
<br>
- Imágenes totales: 7,838
- Especies vegetales totales: 425
- Imágenes totales con máscaras: 393,671 (Etiquetadas automáticamente: 290,898, 73.9 %)

```
    ├── Superclases
        ├── 'pavement cell': 113,561
        ├── 'stomatal complex': 168,084
    ├── Subclases de  'stomatal complex'
        ├── 'stoma': 97,691
        ├── 'outer ledge': 11,928
        ├── 'pore': 2,407
```

- Modalidades totales: 7

```
    ├── ClearStain_Brightfield
    ├── Imprints_Brightfield
    ├── Imprints_DIC
    ├── Leaf_Brightfield
    ├── Leaf_Topometry
    ├── Peels_Brightfield
    ├── Peels_SEM
```
El conjunto de datos se compartirá en HuggingFace: [https://huggingface.co/datasets/aliasz/StomataPy400K](https://huggingface.co/datasets/aliasz/StomataPy400K)

---
# ISAT-SAM: Herramienta Interactiva de Anotación de Estomas con Segment Anything Model
<img src="asserts/isat_demo.gif" width="800" height=auto /> </div>
<br>
Ya disponible en GitHub: [https://github.com/yatengLG/ISAT_with_segment_anything](https://github.com/yatengLG/ISAT_with_segment_anything)

---
# Modelos: Actualmente en fase de prueba beta

Los modelos se compartirán en HuggingFace: [https://huggingface.co/aliasz/StomataPy400K-Models](https://huggingface.co/aliasz/StomataPy400K-Models)  

**Nota**: necesitas la **clave secreta** para acceder a los modelos. Si te interesa probar los modelos, contáctame en hongyuan.zhang@usys.ethz.ch  

Prueba el modelo aquí: <a target="_blank" href="https://colab.research.google.com/github/Alias-z/StomataPy/blob/master/StomataPy_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---
# Salón de la Fama
Agradecemos profundamente a los siguientes **participantes de la prueba beta**:

**Sara Paola Nastasi** y **Alex Costa** de la Universidad de Milán, Italia  
**Robert Caine**, **Nitkamon Iamprasertkun**, **Yixiang Shan** y **Safia El Amiri**, de la Universidad de Sheffield, Reino Unido  
**Ron Eric Stein** y **Tabea Lara Zwaller**, de la Universidad de Heidelberg, Alemania  
**Laboratorio Didier Le Thiec** de INRAE, Francia  
**Emilio Petrone Mendoza** de la Universidad de Nápoles Federico II, Italia  
**Hana Horak** de la Universidad de Tartu, Estonia  
**Mengjie Fan** y **Tracy Lawson**, de la Universidad de Essex, Reino Unido  
**Pawandeep Singh Kohli** y **Micheal Rasissig** de la Universidad de Berna, Suiza  
**Nattiwong Pankasem** de la Universidad de California San Diego, EE. UU.  
**Xiaojuan Wang** del Museo de Ciencia y Tecnología de Shanghái, China  
**Xiaoqian Sha** y **Tian Zhang** de la Universidad de Henan, China
