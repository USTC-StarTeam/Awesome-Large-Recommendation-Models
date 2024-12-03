# Large Recommendation Models

Welcome to the GitHub repository dedicated to exploring and advancing large recommendation models. This repository will be continuously updated with the latest works and insights in this rapidly evolving field.

---

## 🔥🔥🔥 Scaling New Frontiers: Insights into Large Recommendation Models

**[Project Page [This Page]](https://github.com/USTC-StarTeam/Large-Recommendation-Models)** | **[Paper](https://arxiv.org/abs/2412.00714.pdf)**

- **Scalability Analysis**: This pioneering paper delves into the scalability of large recommendation model architectures, leveraging popular Transformers such as HSTU, Llama, GPT, and SASRec. :star2:
- **Comprehensive Study**: We conduct an extensive ablation study and parameter analysis on HSTU, uncovering the origins of scaling laws. Our work also enhances the scalability of the traditional Transformer-based sequential recommendation model, SASRec, by integrating effective modules from scalable large recommendation models. :star2:
- **Complex User Behavior**: This is the first study to assess the performance of large recommendation models on complex user behavior sequence data, pinpointing areas for improvement in modeling intricate user behaviors, including auxiliary information, multi-behaviors, and cross-domain joint modeling. :star2:
- **Ranking Tasks Evaluation**: To our knowledge, this is the first comprehensive evaluation of large recommendation models on ranking tasks, demonstrating their scalability. Our findings offer valuable insights into designing efficient large ranking recommendation models, with a focus on datasets and hyperparameters. :star2:

---

## 🔥🔥🔥 Predictive Models in Sequential Recommendations: Bridging Performance Laws with Data Quality Insights

**[Paper](https://arxiv.org/abs/2412.00430)**

- **Scalability Analysis**: This paper introduces a Performance Law to address the scalability of Sequential Recommendation (SR) models by analyzing model performance rather than loss, aiming to optimize computational resource management. :star2:
- **Data Quality Extension**: The study emphasizes understanding users' interest patterns through their historical interactions and introduces Approximate Entropy (ApEn) as a significant measure of data quality, improving the interaction data analysis critical for scaling law. :star2:
- **Comprehensive Study**: We propose a novel correlation between model size and performance by fitting metrics such as hit rate (HR) and normalized discounted cumulative gain (NDCG), validated theoretically and experimentally across different models and datasets. :star2:
- **Optimizing Model Parameters**: This approach facilitates the determination of optimal parameters for embedding dimensions and model layers, as calculated using the Performance Law, and observes potential performance gains when scaling the model across different frameworks. :star2:

---

## 🔥🔥🔥 A Survey on Large Language Models for Recommendation

**[Project Page](https://github.com/WLiK/LLM4Rec-Awesome-Papers)** | **[Paper](https://arxiv.org/abs/2305.19860)**

- **Comprehensive Review**: We present the first systematic review and analysis of the application of generative large models in recommendation systems, offering a foundational understanding of this innovative field. :star2:
- **Categorical Framework**: Our research classifies current studies on large language models in recommendation systems into three distinct paradigms. This categorization provides a clear and structured overview, facilitating a deeper understanding of the diverse approaches within this emerging discipline. :star2:
- **Analysis of Strengths and Challenges**: We evaluate the strengths and weaknesses of existing methods, identify key challenges faced by LLM-based recommendation systems, and offer insights to inspire future research in this promising area. :star2:

---

## Related Works of Large Recommendation Models
- [Large Recommendation Models](#large-recommendation-models)
  - [🔥🔥🔥 Scaling New Frontiers: Insights into Large Recommendation Models](#-scaling-new-frontiers-insights-into-large-recommendation-models)
  - [🔥🔥🔥 Predictive Models in Sequential Recommendations: Bridging Performance Laws with Data Quality Insights](#-predictive-models-in-sequential-recommendations-bridging-performance-laws-with-data-quality-insights)
  - [🔥🔥🔥 A Survey on Large Language Models for Recommendation](#-a-survey-on-large-language-models-for-recommendation)
  - [Related Works of Large Recommendation Models](#related-works-of-large-recommendation-models)
  - [Paper List](#paper-list)
    - [Long Sequence Modeling](#long-sequence-modeling)
    - [Sequence Modeling with Side Information](#sequence-modeling-with-side-information)
    - [Multiple Behavior Modeling](#multiple-behavior-modeling)
    - [Multiple Domain Modeling](#multiple-domain-modeling)
    - [Data Engineering](#data-engineering)
    - [Tokenizer Application](#tokenizer-application)

---

## Paper List

### Long Sequence Modeling
|  Title  |   Link  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| **IFA: Interaction Fidelity Attention for Entire Lifelong Behaviour Sequence Modeling** | [PDF](https://arxiv.org/pdf/2406.09742) | 2024-6-14 | - |
| **Learning to retrieve user behaviors for click-through rate estimation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3579354) | 2023-4-8 | [GitHub](https://github.com/qinjr/UBR4CTR) |
| **PinnerFormer: Sequence Modeling for User Representation at Pinterest** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3534678.3539156) | 2022-8-14 | - |
| **Linear-time self attention with codeword histogram for efficient recommendation** | [PDF](https://arxiv.org/pdf/2105.14068) | 2021-6-3 | [GitHub](https://github.com/libertyeagle/LISA) |
| **Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction** | [PDF](https://arxiv.org/pdf/2006.05639) | 2020-10-19 | - |
| **User behavior retrieval for click-through rate prediction** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3397271.3401440) | 2020-7-25 | [GitHub](https://github.com/qinjr/UBR4CTR) |
| **Practice on long sequential user behavior modeling for click-through rate prediction** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3292500.3330666) | 2019-7-25 | [GitHub](https://github.com/UIC-Paper/MIMN) |
| **Lifelong sequential modeling with personalized memorization for user response prediction** | [PDF](https://arxiv.org/pdf/1905.00758) | 2019-7-18 | [GitHub](https://github.com/alimamarankgroup/HPMN) |
| **Sequential recommendation with user memory networks**      | [PDF](https://dl.acm.org/doi/pdf/10.1145/3159652.3159668) | 2018-1-2 | - |

### Sequence Modeling with Side Information
|  Title  |   Link  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| **MEANTIME: Mixture of Attention Mechanisms with Multi-temporal Embeddings for Sequential Recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3383313.3412216) | 2020-9-22 | [GitHub](https://github.com/SungMinCho/MEANTIME) |
| **Time Matters: Sequential Recommendation with Complex Temporal Information** | [PDF](https://static.aminer.cn/upload/pdf/1975/1327/2004/5f0277e911dc830562231df0_0.pdf) | 2020-7-25 | - |
| **Time Interval Aware Self-Attention for Sequential Recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3336191.3371786) | 2020-1-22 | - |
| **BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer** | [PDF](http://ofey.me/papers/BERT4Rec.pdf) | 2019-11-3 | [GitHub](https://github.com/FeiSun/BERT4Rec) |
| **Self-Attentive Sequential Recommendation** | [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8594844&casa_token=ctgKMepDQVQAAAAA:qIMsayAqvgz99fM0Mn1mGKYK2L4uli4-dapeOK25U3DzpJt0ymUoshtUP40YpZYr06gpSSjtgDs&tag=1) | 2018-12-30 | [GitHub](https://github.com/kang205/SASRec) |

### Multiple Behavior Modeling
|  Title  |   Link  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| **Multi-Behavior Generative Recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3627673.3679730) | 2024-10-21 | [GitHub](https://github.com/anananan116/MBGen) |
| **Denoising Pre-Training and Customized Prompt Learning for Efficient Multi-Behavior Sequential Recommendation** | [PDF](https://arxiv.org/pdf/2408.11372) | 2024-8-21 | - |
| **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations** | [PDF](https://arxiv.org/pdf/2402.17152) | 2024-5-6 | [GitHub](https://github.com/facebookresearch/generative-recommenders) |
| **Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation** | [PDF](https://www.atailab.cn/seminar2024Spring/pdf/2024_WWW_EfficientNoise-DecouplingforMulti-BehaviorSequentialRecommendation.pdf) | 2024-3-13 | - |
| **Personalized Behavior-Aware Transformer for Multi-Behavior Sequential Recommendation** | [PDF](https://arxiv.org/pdf/2402.14473) | 2023-10-27 | [GitHub](https://github.com/TiliaceaeSU/PBAT) |
| **Coarse-to-fine knowledge-enhanced multi-interest learning framework for multi-behavior recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3606369) | 2023-8-18 | - |
| **Hierarchical projection enhanced multi-behavior recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3580305.3599838) | 2023-8-4 | [GitHub](https://github.com/MC-CV/HPMR) |
| **Compressed interaction graph based framework for multi-behavior recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3543507.3583312) | 2023-4-30 | - |
| **Multi-behavior sequential transformer recommender** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3477495.3532023) | 2022-7-7 | [GitHub](https://github.com/huawei-noah/benchmark/tree/main/FuxiCTR/model_zoo) |
| **Multi-view multi-behavior contrastive learning in recommendation** | [PDF](https://arxiv.org/pdf/2203.10576) | 2022-4-8 | [GitHub](https://github.com/wyqing20/MMCLR) |
| **Deep multifaceted transformers for multi-objective ranking in large-scale e-commerce recommender systems** | [PDF](https://www.researchgate.net/profile/Yulong-Gu-5/publication/344752297_Deep_Multifaceted_Transformers_for_Multi-objective_Rank-ing_in_Large-Scale_E-commerce_Recommender_Systems/links/5f8dbf1f299bf1b53e32af1c/Deep-Multifaceted-Transformers-for-Multi-objective-Rank-ing-in-Large-Scale-E-commerce-Recommender-Systems.pdf) | 2020-10-19 | [GitHub](https://github.com/guyulongcs/CIKM2020_DMT) |
| **Multiplex behavioral relation learning for recommendation via memory augmented transformer network** | [PDF](https://arxiv.org/pdf/2110.04002) | 2020-7-25 | [GitHub](https://github.com/akaxlh/MATN) |
| **Buying or browsing?: Predicting real-time purchasing intent using attention-based deep network with multiple behavior** | [PDF](https://www.researchgate.net/profile/Long-Guo-14/publication/334719244_Buying_or_Browsing_Predicting_Real-time_Purchasing_Intent_using_Attention-based_Deep_Network_with_Multiple_Behavior/links/5ece6c88299bf1c67d206b2e/Buying-or-Browsing-Predicting-Real-time-Purchasing-Intent-using-Attention-based-Deep-Network-with-Multiple-Behavior.pdf) | 2019-7-25 | - |
| **Neural Multi-task Recommendation from Multi-behavior Data** | [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8731537&casa_token=R4GpeevirHcAAAAA:CVIql6V0IhPuxaKzmujEfwEoMCc9rVe5I-7BOgBI14Smwc4o0xmkd6_SCSH4PaIjKjV_qlk5fgY&tag=1) | 2019-6-6 | - |
### Multiple Domain Modeling
|  Title  |   Link  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| **MF-GSLAE: A Multi-Factor User Representation Pre-training Framework for Dual-Target Cross-Domain Recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3690382) | 2024-10-24 | [GitHub](https://github.com/USTC-StarTeam/MF-GSLAE) |
| **MDAP: A Multi-view Disentangled and Adaptive Preference Learning Framework for Cross-Domain Recommendation** | [PDF](https://arxiv.org/pdf/2410.05877) | 2024-10-8 | [GitHub](https://github.com/The-garden-of-sinner/MDAP) |
| **Learning Partially Aligned Item Representation for Cross-Domain Sequential Recommendation** | [PDF](https://arxiv.org/pdf/2405.12473) | 2024-8-21 | - |
| **Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation** | [PDF](https://arxiv.org/pdf/2406.03085) | 2024-6-5 | [GitHub](https://github.com/TingJShen/URLLM) |
| **A Unified Framework for Adaptive Representation Enhancement and Inversed Learning in Cross-Domain Recommendation** | [PDF](https://arxiv.org/pdf/2404.00268) | 2024-5-30 | - |
| **Triple Sequence Learning for Cross-Domain Recommendation** | [PDF](https://arxiv.org/pdf/2304.05027) | 2024-2-9 | - |
| **Learning vector-quantized item representation for transferable sequential recommenders** | [PDF](https://arxiv.org/pdf/2210.12316) | 2023-4-30 | [GitHub](https://github.com/RUCAIBox/VQ-Rec) |
| **Contrastive Cross-Domain Sequential Recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3511808.3557262) | 2022-10-17 | [GitHub](https://github.com/cjx96/C2DSR) |
| **Towards universal sequence representation learning for recommender systems** | [PDF](https://arxiv.org/pdf/2206.05941) | 2022-8-14 | [GitHub](https://github.com/RUCAIBox/UniSRec) |
| **RecGURU: Adversarial Learning of Generalized User Representations for Cross-domain Recommendation** | [PDF](https://arxiv.org/pdf/2111.10093) | 2022-2-15 | [GitHub](https://github.com/Chain123/RecGURU) |

### Data Engineering
|  Title  |   Link  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| **Dataset Regeneration for Sequential Recommendation** | [PDF](https://arxiv.org/pdf/2405.17795) | 2024-8-24 | [GitHub](https://github.com/USTC-StarTeam/DR4SR) |
| **Entropy Law: The Story Behind Data Compression and LLM Performance** | [PDF](https://arxiv.org/pdf/2407.06645) | 2024-7-11 | [GitHub](https://github.com/USTC-StarTeam/ZIP) |
| **A Survey on Data-Centric Recommender Systems** | [PDF](https://arxiv.org/pdf/2401.17878) | 2024-3-28 | - |
| **Data Management For Training Large Language Models: A Survey** | [PDF](https://arxiv.org/pdf/2312.01700) | 2023-12-1 | - |
| **Robust preference-guided denoising for graph based social recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3543507.3583374) | 2023-4-30 | [GitHub](https://github.com/tsinghua-fib-lab/Graph-Denoising-SocialRec) |
| **Autodenoise: Automatic data instance denoising for recommendations** | [PDF](https://arxiv.org/pdf/2303.06611) | 2023-4-30 | - |
| **An empirical analysis of compute-optimal large language model training** | [PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf) | 2022-11-1 | - |
| **Hierarchical item inconsistency signal learning for sequence denoising in sequential recommendation** | [PDF](https://www.atailab.cn/seminar2022Fall/pdf/2022_CIKM_Hierarchical%20Item%20Inconsistency%20Signal%20Learning%20for%20Sequence%20Denoising%20in%20Sequential%20Recommendation.pdf) | 2022-10-17 | [GitHub](https://github.com/zc-97/HSD) |
| **The Datasets Dilemma: How Much Do We Really Know About Recommendation Datasets?** | [PDF](https://yileccc.github.io/paper/wsdm22-dataset.pdf) | 2022-2-15 | [GitHub](https://github.com/almightyGOSU/TheDatasetsDilemma) |
| **Mixgcf: An improved training method for graph neural network-based recommender systems** | [PDF](https://keg.cs.tsinghua.edu.cn/yuxiao/papers/KDD21-Huang-et-al-MixGCF.pdf) | 2021-8-14 | [GitHub](https://github.com/huangtinglin/MixGCF) |
| **Joint item recommendation and attribute inference: An adaptive graph convolutional network approach** | [PDF](https://arxiv.org/pdf/2005.12021) | 2020-7-25 | - |
| **Scaling laws for neural language models** | [PDF](https://arxiv.org/pdf/2001.08361) | 2020-1-23 | - |
| **Enhancing collaborative filtering with generative augmentation** | [PDF](https://drive.google.com/file/d/1Oy1C2eSxqG5mtI9CZYtoKYrqcKomN381/view?pli=1) | 2019-7-25 | - |

### Tokenizer Application
|  Title  |   Link  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| **Breaking Determinism: Fuzzy Modeling of Sequential Recommendation Using Discrete State Space Diffusion Model** | [PDF](https://arxiv.org/pdf/2410.23994) | 2024-11-1 | - |
| **Toward a Theory of Tokenization in LLMs** | [PDF](https://arxiv.org/pdf/2404.08335) | 2024-4-12 | - |
| **Text is all you need: Learning language representations for sequential recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3580305.3599519) | 2023-8-4 | - |
| **Recommender systems with generative retrieval** | [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf) | 2023-5-18 | - |
| **Sinkhorn Collaborative Filtering** | [PDF](https://yileccc.github.io/paper/www21-sinkhorncf.pdf) | 2021-6-3 | [GitHub](https://github.com/boathit/sinkhorncf) |
| **Automated hate speech detection on Twitter** | [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9128428&casa_token=PqGZTMJxutwAAAAA:krgIt_idrcc3M_MrdOk6f6mMfB1E5sDPbQviHmbLd2r-ksYKiBCmLR4WDe3QdVlJ9usL5FHCAzM-&tag=1) | 2019-9-21 | - |
| **ANR: Aspect-based Neural Recommender** | [PDF](https://raihanjoty.github.io/papers/chin-et-al-cikm-18.pdf) | 2018-10-17 | - |
| **Neural attentional rating regression with review-level explanations** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3178876.3186070) | 2018-4-10 | - |
| **Transnets: Learning to transform for recommendation** | [PDF](https://arxiv.org/pdf/1704.02298) | 2017-8-27 | - |
| **Neural Collaborative Filtering** | [PDF](https://arxiv.org/pdf/1708.05031) | 2017-4-3 | [GitHub](https://github.com/zc-97/HSD) |
| **Joint deep modeling of users and items using reviews for recommendation** | [PDF](https://dl.acm.org/doi/pdf/10.1145/3018661.3018665) | 2017-2-2 | - |
| **Variational graph auto-encoders** | [PDF](https://qiniu.pattern.swarma.org/pdf/arxiv/1611.07308.pdf) | 2016-11-21 | - |
| **Convolutional matrix factorization for document context-aware recommendation** | [PDF](https://dparra.sitios.ing.uc.cl/classes/recsys-2016-2/students/CNN-GSepulveda.pdf) | 2016-9-7 | - |
| **node2vec: Scalable feature learning for networks** | [PDF](https://dl.acm.org/doi/pdf/10.1145/2939672.2939754) | 2016-8-13 | - |

