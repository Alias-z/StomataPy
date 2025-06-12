# StomataPy is a collection of resources
- **StomataPy400K**: The largest annotated stomata dataset ever (Opensource soon)
- **ISAT-SAM**: Interactive Stomata Annotation Tool with Segment Anything Model (Opensourced already)
- **StomataPy400K models**: a series of models for stomata related segmentations (Opensource soon)
---

# StomataPy400K: The largest annotated stomata dataset ever (Opensource soon)
<img src="asserts/datasets_preview.gif" width="500" height=auto /> </div>
<br>
- Total images: 7,838
- Total plant species: 425
- Total images with masks: 393,671 (Autolabeled: 290,898, 73.9 %)

```
    ├── Superclasses
        ├── 'pavement cell': 113,561
        ├── 'stomatal complex': 168,084
    ├── Subclasses of  'stomatal complex'
        ├── 'stoma': 97,691
        ├── 'outer ledge': 11,928
        ├── 'pore': 2,407
```

- Total_modalities: 7

```
    ├── ClearStain_Brightfield
    ├── Imprints_Brightfield
    ├── Imprints_DIC
    ├── Leaf_Brightfield
    ├── Leaf_Topometry
    ├── Peels_Brightfield
    ├── Peels_SEM
```
The dataset will be shared on HuggingFace: [https://huggingface.co/datasets/aliasz/StomataPy400K](https://huggingface.co/datasets/aliasz/StomataPy400K)

---
# ISAT-SAM: Interactive Stomata Annotation Tool with Segment Anything Model
<img src="asserts/isat_demo.gif" width="500" height=auto /> </div>
<br>
Already available on GitHub: [https://github.com/yatengLG/ISAT_with_segment_anything](https://github.com/yatengLG/ISAT_with_segment_anything)

---
# Models: Now in beta-test

The models will be shared on HuggingFace: [https://huggingface.co/aliasz/StomataPy400K-Models](https://huggingface.co/aliasz/StomataPy400K-Models)  

**Note**: you need the **secret key** to access the models. If you are interested in testing the models, please contact me at hongyuan.zhang@usys.ethz.ch  

Try the model here: <a target="_blank" href="https://colab.research.google.com/github/Alias-z/StomataPy/blob/master/StomataPy_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---
# Hall of Fame
We greatly appreciate the following **beta-test participants**:

**Sara Paola Nastasi** and **Alex Costa** from University of Milan, Italy  
**Robert Caine**, **Nitkamon Iamprasertkun**, **Yixiang Shan**, and **Safia El Amiri**, from University of Sheffield, UK  
**Ron Eric Stein** and **Tabea Lara Zwaller**, from Universität Heidelberg, Germany  
**Didier Le Thiec Lab** from INRAE, France  
**Emilio Petrone Mendoza** from University of Naples Federico II, Italy  
**Hana Horak** from University of Tartu, Estonia  
**Mengjie Fan** and **Tracy Lawson**, from University of Essex, UK  
**Pawandeep Singh Kohli** and **Micheal Rasissig** from Uiversity of Bern, Switzerland  
**Nattiwong Pankasem** from University of California San Diego, USA  
**Xiaojuan Wang** from Shanghai Science & Technology Museum, China  
**Xiaoqian Sha** and **Tian Zhang** from Henan University, China  
