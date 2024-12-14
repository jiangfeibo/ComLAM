
# A Comprehensive Survey of Large AI Models for Communications: Foundations, Applications, and Future Directions

## Abstract



## Contents

* [A Comprehensive Survey of Large AI Models for Communications: Foundations, Applications, and Future Directions](#A-Comprehensive-Survey-of-Large-AI-Models-for-Communications-Foundations-Applications-and-Future-Directions)
  * [Abstract](#Abstract)
  * [Contents](#Contents)
  * [I.INTRODUCTION](#IINTRODUCTION)
    * [A. Background](#A-Background)
      * [1) Traditional machine learning-assisted communication](#1-Traditional-machine-learning-assisted-communication)
      * [2) Deep supervised learning-assisted communication](#2-Deep-supervised-learning-assisted-communication)
      * [3) Deep reinforcement learning-assisted communication](#3-Deep-reinforcement-learning-assisted-communication)
      * [4) Generative AI-assisted communication](4-Generative-AI-assisted-communication)
    * [B. Motivation](#B-Motivation)
      * [1) Outstanding global perspective and decision-making capability](#1-Outstanding-global-perspective-and-decision-making-capability)
      * [2) Significant robustness and generalization capability](#2-Significant-robustness-and-generalization-capability)
      * [3) Remarkable comprehension and emergent capabilities](#3-Remarkable-comprehension-and-emergent-capabilities)
    * [C. Related survey work](#C-Related-survey-work)
    * [D. Contributions](#D-Contributions)
      * [1) Learning foundations of LAMs for communications](#1-Learning-foundations-of-LAMs-for-communications)
      * [2) Design foundations of LAMs for communications](#2-Design-foundations-of-LAMs-for-communications)
      * [3) Applications of LAMs for communications](#3-Applications-of-LAMs-for-communications)
      * [4) Research challenges and future directions of LAMs for communications](#4-Research-challenges-and-future-directions-of-LAMs-for-communications)
  * [II. LEARNING FOUNDATIONS OF LAMS FOR COMMUNICATIONS](#II-LEARNING-FOUNDATIONS-OF-LAMS-FOR-COMMUNICATIONS)
    * [A. Development history of LAMs](#A-Development-history-of-LAMs)
      * [1) Emergence stage](#1-Emergence-stage)
      * [2) Initial stage](#2-Initial-stage)
      * [3) Mature stage](#3-Mature-stage)
      * [4) Multimodal stage](#4-Multimodal-stage)
      * [5) World model stage](#5-World-model-stage)
    * [B. Communication datasets for LAMs](#B-Communication-datasets-for-LAMs)
      * [1) General datasets](#1-General-datasets)
      * [2) Communication content filtering](#2-Communication-content-filtering)
      * [3) Communication pre-training datasets](#3-Communication-pre-training-datasets)
      * [4) Communication fine-tuning datasets](#4-Communication-fine-tuning-datasets)
      * [5) Communication alignment datasets](#5-Communication-alignment-datasets)
    * [C. Pre-training of LAMs for communications](#C-Pre-training-of-LAMs-for-communications)
      * [1) Pre-training on general datasets](#1-Pre-training-on-general-datasets)
      * [2) Continual pre-training on communication datasets](#2-Continual-pre-training-on-communication-datasets)
      * [3) Optimization strategies for pre-training](#3-Optimization-strategies-for-pre-training)
    * [D. Fine-tuning of LAMs for communications](#D-Fine-tuning-of-LAMs-for-communications)
      * [1) Diverse instruction tasks](#1-Diverse-instruction-tasks)
      * [2) Fine-tuning steps](#2-Fine-tuning-steps)
      * [3) Fine-tuning techniques](#3-Fine-tuning-techniques)
    * [E. Alignment of LAMs for communications](#E-Alignment-of-LAMs-for-communications)
      * [1) RLHF](#1-RLHF)
      * [2) Other alignment technologies](#2-Other-alignment-technologies)
    * [F. Evaluation of LAMs for communications](#F-Evaluation-of-LAMs-for-communications)
      * [1) Communication Q&A and classification evaluation](#1-Communication-Q-A-and-classification-evaluation)
      * [2) Communication modeling and code generation evaluation](#2-Communication-modeling-and-code-generation-evaluation)
      * [3) Communication reasoning evaluation](#3-Communication-reasoning-evaluation)
      * [4) Communication tool learning evaluation](#4-Communication-tool-learning-evaluation)
      * [5) Communication security evaluation](#5-Communication-security-evaluation)
  * [III. DESIGN FOUNDATIONS OF LAMS FOR COMMUNICATIONS](#III-DESIGN-FOUNDATIONS-OF-LAMS-FOR-COMMUNICATIONS)
    * [A. Key architecture of LAMs](#A-Key-architecture-of-LAMs)
      * [1) Transformer](#1-Transformer)
      * [2) Variational autoencoder](#2-Variational-autoencoder)
      * [3) Diffusion models](#3-Diffusion-models)
      * [4) Mamba](#4-Mamba)
    * [B. Classification of LAMs](#B-Classification-of-LAMs)
      * [1) Large language model](#1-Large-language-model)
      * [2) Large vision model](#2-Large-vision-model)
      * [3) Vision-language model](#3-Vision-language-model)
      * [4) Large multimodal model](#4-Large-multimodal-model)
      * [5) World model](#5-World-model)
      * [6) Lightweight large AI model](#6-Lightweight-large-AI-model)
    * [C. Optimization of LAMs](#C-Optimization-of-LAMs)
      * [1) In-context learning](#1-In-context-learning)
      * [2) X of thought](#2-X-of-thought)
      * [3) Retrieval-augmented generation](#3-Retrieval-augmented-generation)
      * [4) Multi-agent system](#4-Multi-agent-system)
      * [5) Mixture of experts](#5-Mixture-of-experts)
  * [IV. LAMS FOR PHYSICAL LAYER DESIGN](#IV-LAMS-FOR-PHYSICAL-LAYER-DESIGN)
    * [A. LLM-assisted physical layer design](#A-LLM-assisted-physical-layer-design)
    * [B. GAI model-assisted physical layer design](#B-GAI-model-assisted-physical-layer-design)
  * [V. LAMS FOR RESOURCE ALLOCATION AND OPTIMIZATION](#V-LAMS-FOR-RESOURCE-ALLOCATION-AND-OPTIMIZATION)
    * [A. Computing resource allocation](#A-Computing-resource-allocation)
    * [B. Spectrum resource allocation](#B-pectrum-resource-allocation)
    * [C. Energy resource optimization](#C-Energy-resource-optimization)
  * [VI. LAMS FOR NETWORK DESIGN AND MANAGEMENT](#VI-LAMS-FOR-NETWORK-DESIGN-AND-MANAGEMENT)
    * [A. Network design](#A-Network-design)
    * [B. Network management](#B-Network-management)
  * [VII. LAMS FOR EDGE INTELLIGENCE](#VII-LAMS-FOR-EDGE-INTELLIGENCE)
    * [A. Edge training and application of LAMs](#A-Edge-training-and-application-of-LAMs)
    * [B. Edge resource scheduling meets LAMs](#B-Edge-resource-scheduling-meets-LAMs)
    * [C. Cross-domain collaboration of LAMs](#C-Cross-domain-collaboration-of-LAMs)
  * [VIII. LAMS FOR SEMANTIC COMMUNICATION](#VIII-LAMS-FOR-SEMANTIC-COMMUNICATION)
    * [A. LLM-based semantic communication systems](#A-LLM-based-semantic-communication-systems)
    * [B. Other LAM-based semantic communication systems](#B-Other-LAM-based-semantic-communication-systems)
  * [IX. LAMS FOR FEDERATED LEARNING](#IX-LAMS-FOR-FEDERATED-LEARNING)
    * [A. Federated fine-tuning for LAMs](#A-Federated-fine-tuning-for-LAMs)
    * [B. Personlized federated learning for LAMs](#B-Personlized-federated-learning-for-LAMs)
  * [X. LAM-BASED AGENT SYSTEMS](#X-LAM-BASED-AGENT-SYSTEMS)
    * [A. Agent systems based on LLMs](#A-Agent-systems-based-on-LLMs)
    * [B. Agent systems based on other GAI models](#B-Agent-systems-based-on-other-GAI-models)
  * [XI. LAMS FOR SECURITY AND PRIVACY](#XI-LAMS-FOR-SECURITY-AND-PRIVACY)
    * [A. Network threat detection and defense](#A-Network-threat-detection-and-defense)
    * [B. Trusted AI in communication networks](#B-Trusted-AI-in-communication-networks)
  * [XII. LAMS FOR DIGITAL TWIN AND METAVERSE](#XII-LAMS-FOR-DIGITAL-TWIN-AND-METAVERSE)
    * [A. LAMs for digital twin](#A-LAMs-for-digital-twin)
    * [B. GAI models for metaverse](#B-GAI-models-for-metaverse)
  * [XIII. LAMS FOR EMERGING APPLICATIONS](#XIII-LAMS-FOR-EMERGING-APPLICATIONS)
    * [A. Smart healthcare](#A-Smart-healthcare)
    * [B. Carbon emissions](#B-Carbon-emissions)
    * [C. Automation systems](#C-Automation-systems)
    * [D. Artificial intelligence of things](#D-Artificial-intelligence-of-things)
    * [E. Integrated satellite, aerial, and terrestrial networks](#E-Integrated-satellite-aerial-and-terrestrial-networks)
    * [F. Integration of UAVs and LLMs](#F-Integration-of-UAVs-and-LLMs)
  * [XIV. RESEARCH CHALLENGES](#XIV-RESEARCH-CHALLENGES)
    * [A. The lack of communication data and knowledge](#A-The-lack-of-communication-data-and-knowledge)
      * [1) The lack of communication data](#1-The-lack-of-communication-data)
      * [2) The lack of communication knowledge](#2-The-lack-of-communication-knowledge)
    * [B. Insufficient logical reasoning capabilities](#B-Insufficient-logical-reasoning-capabilities)
      * [1) Limitations in logical understanding](#1-Limitations-in-logical-understanding)
      * [2) Limitations in logical reasoning](#2-Limitations-in-logical-reasoning)
    * [C. Inadequate explanation and evaluation mechanisms](#C-Inadequate-explanation-and-evaluation-mechanisms)
      * [1) Deficient explanation mechanism](#1-Deficient-explanation-mechanism)
      * [2) Insufficient evaluation mechanism](#2-Insufficient-evaluation-mechanism)
    * [D. Difficulties in the deployment of LAMs](#D-Difficulties-in-the-deployment-of-LAMs)
      * [1) Hardware resource limitation](#1-Hardware-resource-limitation)
      * [2) Communication resources limitation](#2-Communication-resources-limitation)
  * [XV. FUTURE RESEARCH DIRECTIONS](#XV-FUTURE-RESEARCH-DIRECTIONS)
    * [A. Continual learning for LAMs](#A-Continual-learning-for-LAMs)
      * [1) Continual learning algorithms](#1-Continual-learning-algorithms)
      * [2) Continual learning evaluation](#2-Continual-learning-evaluation)
    * [B. Agentic AI](#B-Agentic-AI)
      * [1) Single-agent optimization](#1-Single-agent-optimization)
      * [2) Multi-agent optimization](#2-Multi-agent-optimization)
    * [C. Explainable LAMs](#C-Explainable-LAMs)
      * [1) Interpretable model](#1-nterpretable-model)
      * [2) Interpretable evaluations](#2-Interpretable-evaluations)
    * [D. Efficient deployment of LAMs](#D-Efficient-deployment-of-LAMs)
      * [1) Pruning](#1-Pruning)
      * [2) Quantization](#2-Quantization)
      * [3) Knowledge distillation](#3-Knowledge-distillation)
  * [The Team](#The-Team)
  * [Acknowledgments](#Acknowledgments)
  * [Update log](#Update-log)

## I.INTRODUCTION
### A. Background
#### 1) Traditional machine learning-assisted communication

#### 2) Deep supervised learning-assisted communication

#### 3) Deep reinforcement learning-assisted communication

#### 4) Generative AI-assisted communication

### B. Motivation
#### 1) Outstanding global perspective and decision-making capability

#### 2) Significant robustness and generalization capability

#### 3) Remarkable comprehension and emergent capabilities

### C. Related survey work

### D. Contributions
#### 1) Learning foundations of LAMs for communications

#### 2) Design foundations of LAMs for communications

#### 3) Applications of LAMs for communications

#### 4) Research challenges and future directions of LAMs for communications

## II. LEARNING FOUNDATIONS OF LAMS FOR COMMUNICATIONS
### A. Development history of LAMs
#### 1) Emergence stage

#### 2) Initial stage

#### 3) Mature stage

#### 4) Multimodal stage

#### 5) World model stage

### B. Communication datasets for LAMs
#### 1) General datasets

#### 2) Communication content filtering

#### 3) Communication pre-training datasets

#### 4) Communication fine-tuning datasets

#### 5) Communication alignment datasets

### C. Pre-training of LAMs for communications
#### 1) Pre-training on general datasets

#### 2) Continual pre-training on communication datasets

#### 3) Optimization strategies for pre-training

### D. Fine-tuning of LAMs for communications
#### 1) Diverse instruction tasks

#### 2) Fine-tuning steps

#### 3) Fine-tuning techniques

### E. Alignment of LAMs for communications
#### 1) RLHF

#### 2) Other alignment technologies

### F. Evaluation of LAMs for communications
#### 1) Communication Q&A and classification evaluation

#### 2) Communication modeling and code generation evaluation

#### 3) Communication reasoning evaluation

#### 4) Communication tool learning evaluation

#### 5) Communication security evaluation

## III. DESIGN FOUNDATIONS OF LAMS FOR COMMUNICATIONS
### A. Key architecture of LAMs
#### 1) Transformer
Y. Wang, Z. Gao, D. Zheng, S. Chen, D. Gündüz, and H. V.Poor, “Transformer-empowered 6g intelligent networks: From massive mimo processing to semantic communication,” IEEE Wireless Communications, vol. 30, no. 6, pp. 127–135, 2022.[<a href="https://ieeexplore.ieee.org/abstract/document/9961131/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Yoo, T. Jung, L. Dai, S. Kim, and C.-B. Chae, “Real-time semantic communications with a vision transformer,” in 2022 IEEE International Conference on Communications Workshops (ICC Workshops). IEEE, 2022, pp. 1–2.[<a href="https://ieeexplore.ieee.org/abstract/document/9914635/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Wu, Y. Shao, E. Ozfatura, K. Mikolajczyk, and D. Gündüz,“Transformer-aided wireless image transmission with channel feedback,” IEEE Transactions on Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10500305/" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2) Variational autoencoder
M. A. Alawad, M. Q. Hamdan, and K. A. Hamdi, “Innovative variational autoencoder for an end-to-end communication system,”IEEE Access, 2022.[<a href="https://ieeexplore.ieee.org/abstract/document/9964187/" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Bo, Y. Duan, S. Shao, and M. Tao, “Joint coding-modulation for digital semantic communications via variational autoencoder,”IEEE Transactions on Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10495330/" target="_blank" rel="noopener noreferrer">Paper</a>]

Q. Hu, G. Zhang, Z. Qin, Y. Cai, G. Yu, and G. Y. Li, “Robust semantic communications with masked vq-vae enabled codebook,”IEEE Transactions on Wireless Communications, vol. 22, no. 12,pp. 8707–8722, 2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10101778/" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3) Diffusion models
F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,“Large generative model assisted 3d semantic communication,” arXiv preprint arXiv:2403.05783, 2024.[<a href="https://arxiv.org/abs/2403.05783" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Du, R. Zhang, Y. Liu, J. Wang, Y. Lin, Z. Li, D. Niyato, J. Kang,Z. Xiong, S. Cui et al., “Beyond deep reinforcement learning: A tutorial on generative diffusion models in network optimization,”arXiv preprint arXiv:2308.05384, 2023.[<a href="https://arxiv.org/abs/2308.05384" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/hongyangdu/gdmopt" target="_blank" rel="noopener noreferrer">code</a>]

T. Wu, Z. Chen, D. He, L. Qian, Y. Xu, M. Tao, and W. Zhang,“Cddm: Channel denoising diffusion models for wireless communications,” in GLOBECOM 2023-2023 IEEE Global Communications Conference. IEEE, 2023, pp. 7429–7434.[<a href="https://ieeexplore.ieee.org/abstract/document/10436728/" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Duan, T. Wu, Z. Chen, and M. Tao, “Dm-mimo: Diffusion models for robust semantic communications over mimo channels,”arXiv preprint arXiv:2407.05289,2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10681856/" target="_blank" rel="noopener noreferrer">Paper</a>]

G. Chi, Z. Yang, C. Wu, J. Xu, Y. Gao, Y. Liu, and T. X. Han, “Rfdiffusion: Radio signal generation via time-frequency diffusion,” in Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024, pp. 77–92.[<a href="https://dl.acm.org/doi/abs/10.1145/3636534.3649348" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 4) Mamba
T. Wu, Z. Chen, M. Tao, Y. Sun, X. Xu, W. Zhang, and P. Zhang,“Mambajscc: Adaptive deep joint source-channel coding with generalized state space model,” arXiv preprint arXiv:2409.16592, 2024.[<a href="https://arxiv.org/abs/2409.16592" target="_blank" rel="noopener noreferrer">Paper</a>]

D. Yuan, J. Xue, J. Su, W. Xu, and H. Zhou, “St-mamba: Spatial-temporal mamba for traffic flow estimation recovery using limited data,” in 2024 IEEE/CIC International Conference on Communications in China (ICCC). IEEE, 2024, pp. 1928–1933.[<a href="https://ieeexplore.ieee.org/abstract/document/10681692" target="_blank" rel="noopener noreferrer">Paper</a>]

L. Yu, H. Zhang, J. Liu, C. Liu, J. Yuan, Z. Li, and Z. Wang, “Vimsc:Robust underwater acoustic image semantic communication based on vision mamba model,” in Proceedings of the 2024 12th International Conference on Communications and Broadband Networking, 2024, pp.46–52.[<a href="https://dl.acm.org/doi/abs/10.1145/3688636.3688668" target="_blank" rel="noopener noreferrer">Paper</a>]

### B. Classification of LAMs
#### 1) Large language model
F. Jiang, L. Dong, Y. Peng, K. Wang, K. Yang, C. Pan, and X. You,“Large ai model empowered multimodal semantic communications,”IEEE Communications Magazine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10670195" target="_blank" rel="noopener noreferrer">Paper</a>]

P. Jiang, C.-K. Wen, X. Yi, X. Li, S. Jin, and J. Zhang, “Semantic communications using foundation models: Design approaches and open issues,” IEEE Wireless Communications, vol. 31, no. 3, pp.76–84, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10558822" target="_blank" rel="noopener noreferrer">Paper</a>]

F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, D. Niyato, and O. A. Dobre, “Large language model enhanced multi-agent systems
for 6g communications,” IEEE Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10638533" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Wang, Z. Sun, J. Fan, and H. Ma, “On the uses of large language models to design end-to-end learning semantic communication,” in 2024 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2024, pp. 1–6.[<a href="https://ieeexplore.ieee.org/abstract/document/10570717" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, D. Niyato, J. Kang, Z. Xiong, S. Mao, Z. Han, D. I.Kim, and K. B. Letaief, “When large language model agents meet 6g networks: Perception, grounding, and alignment,” IEEE Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10648594" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2) Large vision model
F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,“Large ai model-based semantic communications,” IEEE Wireless Communications, vol. 31, no. 3, pp. 68–75, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10558819" target="_blank" rel="noopener noreferrer">Paper</a>]

S. Tariq, B. E. Arfeto, C. Zhang, and H. Shin, “Segment anything meets semantic communication,” arXiv preprint arXiv:2306.02094, 2023.[<a href="https://arxiv.org/abs/2306.02094" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3) Vision-language model
F. Jiang, C. Tang, L. Dong, K. Wang, K. Yang, and C. Pan, “Visual language model based cross-modal semantic communication systems,” arXiv preprint arXiv:2407.00020, 2024.[<a href="https://arxiv.org/abs/2407.00020" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 4) Large multimodal model
F. Jiang, L. Dong, Y. Peng, K. Wang, K. Yang, C. Pan, and X. You,“Large ai model empowered multimodal semantic communications,”IEEE Communications Magazine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10670195" target="_blank" rel="noopener noreferrer">Paper</a>]

L. Qiao, M. B. Mashhadi, Z. Gao, C. H. Foh, P. Xiao, and M. Bennis, “Latency-aware generative semantic communications with pretrained diffusion models,” arXiv preprint arXiv:2403.17256, 2024.[<a href="https://arxiv.org/abs/2403.17256" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 5) World model
W. Saad, O. Hashash, C. K. Thomas, C. Chaccour, M. Debbah,N. Mandayam, and Z. Han, “Artificial general intelligence (agi)native wireless systems: A journey beyond 6g,” arXiv preprint arXiv:2405.02336, 2024.[<a href="https://arxiv.org/abs/2405.02336" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 6) Lightweight large AI model
T. S. Do, T. P. Truong, T. Do, H. P. Van, and S. Cho, “Lightweight multiuser multimodal semantic communication system for multimodal large language model communication,” Authorea Preprints, 2024.[<a href="https://www.authorea.com/doi/full/10.22541/au.172479430.09168922" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Rong, Y. Mao, X. He, and M. Chen, “Large-scale traffic flow forecast with lightweight llm in edge intelligence,” IEEE Internet of Things Magazine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10742575" target="_blank" rel="noopener noreferrer">Paper</a>]

### C. Optimization of LAMs
#### 1) In-context learning
M. Zecchin, K. Yu, and O. Simeone, "In-context learning for MIMO equalization using transformer-based sequence models," in *2024 IEEE International Conference on Communications Workshops (ICC Workshops)*, IEEE, 2024, pp. 1573-1578. [<a href="https://arxiv.org/abs/2311.06101" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Abbas, K. Kar, and T. Chen, “Leveraging large language models for wireless symbol detection via in-context learning,” arXiv preprint arXiv:2409.00124, 2024.[<a href="https://arxiv.org/abs/2409.00124" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2) X of thought
Y. Du, H. Deng, S. C. Liew, K. Chen, Y. Shao, and H. Chen,“The power of large language models for wireless communication system development: A case study on fpga platforms,” arXiv preprint arXiv:2307.07319, 2023.[<a href="https://arxiv.org/abs/2307.07319" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Zou, Q. Zhao, L. Bariah, Y. Tian, M. Bennis, S. Lasaulce,M. Debbah, and F. Bader, “Genainet: Enabling wireless collective intelligence via knowledge transfer and reasoning,” arXiv preprint arXiv:2402.16631, 2024.[<a href="https://arxiv.org/abs/2402.16631" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Shao, J. Tong, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang,“Wirelessllm: Empowering large language models towards wireless intelligence,” arXiv preprint arXiv:2405.17053, 2024.[<a href="https://arxiv.org/abs/2405.17053" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3) Retrieval-augmented generation
A.-L. Bornea, F. Ayed, A. De Domenico, N. Piovesan, and A. Maatouk, “Telco-rag: Navigating the challenges of retrieval-augmented language models for telecommunications,” arXiv preprint arXiv:2404.15939, 2024.[<a href="https://arxiv.org/abs/2404.15939" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Tang and W. Guo, “Automatic retrieval-augmented generation of 6g network specifications for use cases,” arXiv preprint arXiv:2405.03122, 2024.[<a href="https://arxiv.org/abs/2405.03122" target="_blank" rel="noopener noreferrer">Paper</a>]

X. Huang, Y. Tang, J. Li, N. Zhang, and X. S. Shen, “Toward effective retrieval augmented generative services in 6g networks,” IEEE
Network, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10620276" target="_blank" rel="noopener noreferrer">Paper</a>]

S. Xu, C. K. Thomas, O. Hashash, N. Muralidhar, W. Saad, and N. Ramakrishnan, “Large multi-modal models (lmms) as universal foundation models for ai-native wireless systems,” arXiv preprint arXiv:2402.01748, 2024.[<a href="https://arxiv.org/abs/2402.01748" target="_blank" rel="noopener noreferrer">Paper</a>]

G. Y. GMY, J. A. Ayala-Romero, A. Garcia-Saavedra, and X. Costa-Perez, “Telecomrag: Taming telecom standards with retrieval augmented generation and llms,” Authorea Preprints, 2024.[<a href="https://arxiv.org/abs/2406.07053" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 4) Multi-agent system
J. Tong, J. Shao, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang,“Wirelessagent: Large language model agents for intelligent wireless networks,” arXiv preprint arXiv:2409.07964, 2024.[<a href="https://arxiv.org/abs/2409.07964" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/weiiguo/wireless-agent" target="_blank" rel="noopener noreferrer">code</a>]

F. Jiang, L. Dong, Y. Peng, K. Wang, K. Yang,C. Pan, D. T. Niyato, and O. A. Dobre, “Large language model enhanced multi-agent systems for 6g communications,” ArXiv, vol. abs/2312.07850, 2023. [<a href="https://ieeexplore.ieee.org/abstract/document/10638533/" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 5) Mixture of experts
R. Zhang, H. Du, Y. Liu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, and D. I. Kim, “Interactive generative ai agents for satellite networks through a mixture of experts transmission,”arXiv preprint arXiv:2404.09134, 2024.[<a href="https://arxiv.org/abs/2404.09134" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Wang, H. Du, G. Sun, J. Kang, H. Zhou, D. Niyato, and J. Chen,“Optimizing 6g integrated sensing and communications (isac) via expert networks,” arXiv preprint arXiv:2406.00408, 2024.[<a href="https://arxiv.org/abs/2406.00408" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, Y. Fang,D. I. Kim et al., “Integration of mixture of experts and multimodal generative ai in internet of vehicles: A survey,” arXiv preprint arXiv:2404.16356, 2024.[<a href="https://arxiv.org/abs/2404.16356" target="_blank" rel="noopener noreferrer">Paper</a>]

## IV. LAMS FOR PHYSICAL LAYER DESIGN
### A. LLM-assisted physical layer design

### B. GAI model-assisted physical layer design

## V. LAMS FOR RESOURCE ALLOCATION AND OPTIMIZATION
### A. Computing resource allocation

### B. Spectrum resource allocation

### C. Energy resource optimization

## VI. LAMS FOR NETWORK DESIGN AND MANAGEMENT
### A. Network design

### B. Network management

## VII. LAMS FOR EDGE INTELLIGENCE
### A. Edge training and application of LAMs

### B. Edge resource scheduling meets LAMs

### C. Cross-domain collaboration of LAMs

## VIII. LAMS FOR SEMANTIC COMMUNICATION
### A. LLM-based semantic communication systems

### B. Other LAM-based semantic communication systems

## IX. LAMS FOR FEDERATED LEARNING
### A. Federated fine-tuning for LAMs

### B. Personlized federated learning for LAMs

## X. LAM-BASED AGENT SYSTEMS
### A. Agent systems based on LLMs

### B. Agent systems based on other GAI models

## XI. LAMS FOR SECURITY AND PRIVACY
### A. Network threat detection and defense

### B. Trusted AI in communication networks

## XII. LAMS FOR DIGITAL TWIN AND METAVERSE
### A. LAMs for digital twin
Y. Xia, M. Shenoy, N. Jazdi, and M. Weyrich, “Towards autonomous system: flexible modular production system enhanced with large language model agents,” in 2023 IEEE 28th International Conference on Emerging Technologies and Factory Automation (ETFA). IEEE, 2023, pp. 1–8.[<a href="https://ieeexplore.ieee.org/abstract/document/10275362/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/YuchenXia/GPT4IndustrialAutomation" target="_blank" rel="noopener noreferrer">Code</a>]
Y. Hong, J. Wu, and R. Morello, “Llm-twin: mini-giant model-driven beyond 5g digital twin networking framework with semantic secure communication and computation,” Scientific Reports, vol. 14, no. 1, p.19065, 2024.[<a href="https://www.nature.com/articles/s41598-024-69474-5" target="_blank" rel="noopener noreferrer">Paper</a>][<a href=" https://github.com/CURRYSGITHUB/LLM-twin/tree/main." target="_blank" rel="noopener noreferrer">Code</a>]
W. Wu, X. Huang, and T. H. Luan, “Ai-native network digital twin for intelligent network management in 6g,” arXiv preprint arXiv:2410.01584, 2024.[<a href="https://arxiv.org/abs/2410.01584" target="_blank" rel="noopener noreferrer">Paper</a>]
K. Duran, L. V. Cakir, M. Ozdem, K. Gursu, and B. Canberk, “Generative ai-enabled digital twins for 6g-enhanced smart cities,” arXiv preprint arXiv:2411.14222, 2024.[<a href="https://arxiv.org/abs/2411.14222" target="_blank" rel="noopener noreferrer">Paper</a>]
M. Corici, H. Buhr, and T. Magedanz, “Generative twin for 6g and beyond 5g networks: Vision, challenges and architecture,” in 2023 2nd International Conference on 6G Networking (6GNet). IEEE, 2023, pp. 1–6.[<a href="https://ieeexplore.ieee.org/abstract/document/10317780/" target="_blank" rel="noopener noreferrer">Paper</a>]
### B. GAI models for metaverse

## XIII. LAMS FOR EMERGING APPLICATIONS
### A. Smart healthcare
M. Abbasian, I. Azimi, A. M. Rahmani, and R. Jain, “Conversational health agents: A personalized llm-powered agent framework,” arXiv preprint arXiv:2310.02374, 2023.[<a href="https://arxiv.org/abs/2310.02374" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/Institute4FutureHealth/CHA" target="_blank" rel="noopener noreferrer">Code</a>]
### B. Carbon emissions
J. Wen, R. Zhang, D. Niyato, J. Kang, H. Du, Y. Zhang, and Z. Han, “Generative ai for low-carbon artificial intelligence of things,” arXiv preprint arXiv:2404.18077, 2024.[<a href="https://arxiv.org/abs/2404.18077" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="7https://github.com/mlco2/codecarbon" target="_blank" rel="noopener noreferrer">Code</a>]
### C. Automation systems
H. Wen, Y. Li, G. Liu, S. Zhao, T. Yu, T. J.-J. Li, S. Jiang, Y. Liu, Y. Zhang, and Y. Liu, “Autodroid: Llm-powered task automation in android,” in Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024, pp. 543–557.[<a href="https://dl.acm.org/doi/abs/10.1145/3636534.3649379" target="_blank" rel="noopener noreferrer">Paper</a>]
### D. Artificial intelligence of things
H. Cui, Y. Du, Q. Yang, Y. Shao, and S. C. Liew, “Llmind: Orchestrating ai and iot with llm for complex task execution,” IEEE Communications Magazine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10697418/" target="_blank" rel="noopener noreferrer">Paper</a>]
### E. Integrated satellite, aerial, and terrestrial networks
S. Javaid, R. A. Khalil, N. Saeed, B. He, and M.-S. Alouini, “Leveraging large language models for integrated satellite-aerial-terrestrial networks: recent advances and future directions,” arXiv preprint arXiv:2407.04581, 2024.[<a href="https://arxiv.org/abs/2407.04581" target="_blank" rel="noopener noreferrer">Paper</a>]
### F. Integration of UAVs and LLMs
S. Javaid, H. Fahim, B. He, and N. Saeed, “Large language models for uavs: Current state and pathways to the future,” IEEE Open Journal of Vehicular Technology, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10643253/" target="_blank" rel="noopener noreferrer">Paper</a>]
## XIV. RESEARCH CHALLENGES
### A. The lack of communication data and knowledge
#### 1) The lack of communication data

#### 2) The lack of communication knowledge

### B. Insufficient logical reasoning capabilities
#### 1) Limitations in logical understanding

#### 2) Limitations in logical reasoning

### C. Inadequate explanation and evaluation mechanisms
#### 1) Deficient explanation mechanism

#### 2) Insufficient evaluation mechanism

### D. Difficulties in the deployment of LAMs
#### 1) Hardware resource limitation

#### 2) Communication resources limitation

## XV. FUTURE RESEARCH DIRECTIONS
### A. Continual learning for LAMs

#### 1) Continual learning algorithms

#### 2) Continual learning evaluation

### B. Agentic AI
#### 1) Single-agent optimization

#### 2) Multi-agent optimization

### C. Explainable LAMs
#### 1) Interpretable model

#### 2) Interpretable evaluations

### D. Efficient deployment of LAMs
#### 1) Pruning

#### 2) Quantization

#### 3) Knowledge distillation



## ComLAM的关键架构、分类和优化方法
### 大模型分类
#### 1、大语言模型（Large Language Model）

F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,“Large ai model-based semantic communications,” IEEE Wireless Communications, vol. 31, no. 3, pp. 68–75, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10558819" target="_blank" rel="noopener noreferrer">Paper</a>]

P. Jiang, C.-K. Wen, X. Yi, X. Li, S. Jin, and J. Zhang, “Semantic communications using foundation models: Design approaches and open issues,” IEEE Wireless Communications, vol. 31, no. 3, pp.76–84, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10558822" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2、大视觉模型（Large Vision Model）

F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,“Large generative model assisted 3d semantic communication,”arXiv preprint arXiv:2403.05783, 2024.[<a href="https://arxiv.org/abs/2403.05783" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3、视觉语言模型（Vision-Language Model）

F. Jiang, C. Tang, L. Dong, K. Wang, K. Yang, and C. Pan, “Visual language model based cross-modal semantic communication systems,” arXiv preprint arXiv:2407.00020, 2024.[<a href="https://arxiv.org/abs/2407.00020" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 4、多模态大模型（Multimodal Large Model）

L. Qiao, M. B. Mashhadi, Z. Gao, C. H. Foh, P. Xiao, and M. Bennis, “Latency-aware generative semantic communications with pretrained diffusion models,” arXiv preprint arXiv:2403.17256, 2024.[<a href="https://arxiv.org/abs/2403.17256" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 5、世界模型（World Model）

W. Saad, O. Hashash, C. K. Thomas, C. Chaccour, M. Debbah,N. Mandayam, and Z. Han, “Artificial general intelligence (agi)native wireless systems: A journey beyond 6g,” arXiv preprint arXiv:2405.02336, 2024.[<a href="https://arxiv.org/abs/2405.02336" target="_blank" rel="noopener noreferrer">Paper</a>]

### 大模型的关键架构
1、Transformer

Y. Liu, “Roberta: A robustly optimized bert pretraining approach,”arXiv preprint arXiv:1907.11692, 2019.[<a href="https://arxiv.org/abs/1907.11692" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/IndicoDataSolutions/finetune/tree/b1b072cc2b0dd16545f96fc949d5d04af52d02d3" target="_blank" rel="noopener noreferrer">code</a>]

Y. Wang, Z. Gao, D. Zheng, S. Chen, D. Gündüz, and H. V.Poor, “Transformer-empowered 6g intelligent networks: From massive mimo processing to semantic communication,” IEEE Wireless Communications, vol. 30, no. 6, pp. 127–135, 2022.[<a href="https://ieeexplore.ieee.org/abstract/document/9961131/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Yoo, T. Jung, L. Dai, S. Kim, and C.-B. Chae, “Real-time semantic communications with a vision transformer,” in 2022 IEEE International Conference on Communications Workshops (ICC Workshops). IEEE, 2022, pp. 1–2.[<a href="https://ieeexplore.ieee.org/abstract/document/9914635/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Wu, Y. Shao, E. Ozfatura, K. Mikolajczyk, and D. Gündüz,“Transformer-aided wireless image transmission with channel feedback,” IEEE Transactions on Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10500305/" target="_blank" rel="noopener noreferrer">Paper</a>]

2、变分自编码器（VAE）

M. A. Alawad, M. Q. Hamdan, and K. A. Hamdi, “Innovative variational autoencoder for an end-to-end communication system,”IEEE Access, 2022.[<a href="https://ieeexplore.ieee.org/abstract/document/9964187/" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Bo, Y. Duan, S. Shao, and M. Tao, “Joint coding-modulation for digital semantic communications via variational autoencoder,”IEEE Transactions on Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10495330/" target="_blank" rel="noopener noreferrer">Paper</a>]

Q. Hu, G. Zhang, Z. Qin, Y. Cai, G. Yu, and G. Y. Li, “Robust semantic communications with masked vq-vae enabled codebook,”IEEE Transactions on Wireless Communications, vol. 22, no. 12,pp. 8707–8722, 2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10101778/" target="_blank" rel="noopener noreferrer">Paper</a>]

3、扩散模型

H. Du, R. Zhang, Y. Liu, J. Wang, Y. Lin, Z. Li, D. Niyato, J. Kang,Z. Xiong, S. Cui et al., “Beyond deep reinforcement learning: A tutorial on generative diffusion models in network optimization,”arXiv preprint arXiv:2308.05384, 2023.[<a href="https://arxiv.org/abs/2308.05384" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/hongyangdu/gdmopt" target="_blank" rel="noopener noreferrer">code</a>]

T. Wu, Z. Chen, D. He, L. Qian, Y. Xu, M. Tao, and W. Zhang,“Cddm: Channel denoising diffusion models for wireless communications,” in GLOBECOM 2023-2023 IEEE Global Communications Conference. IEEE, 2023, pp. 7429–7434.[<a href="https://ieeexplore.ieee.org/abstract/document/10436728/" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Duan, T. Wu, Z. Chen, and M. Tao, “Dm-mimo: Diffusion models for robust semantic communications over mimo channels,”arXiv preprint arXiv:2407.05289,2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10681856/" target="_blank" rel="noopener noreferrer">Paper</a>]

G. Chi, Z. Yang, C. Wu, J. Xu, Y. Gao, Y. Liu, and T. X. Han, “Rfdiffusion: Radio signal generation via time-frequency diffusion,” in Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024, pp. 77–92.[<a href="https://dl.acm.org/doi/abs/10.1145/3636534.3649348" target="_blank" rel="noopener noreferrer">Paper</a>]

4、Mamba

O. Lieber, B. Lenz, H. Bata, G. Cohen, J. Osin, I. Dalmedigos, E. Safahi, S. Meirom, Y. Belinkov, S. Shalev-Shwartz et al.,“Jamba: A hybrid transformer-mamba language model,” arXiv preprint arXiv:2403.19887, 2024.[<a href="https://arxiv.org/abs/2403.19887" target="_blank" rel="noopener noreferrer">Paper</a>]

J. T. Halloran, M. Gulati, and P. F. Roysdon, “Mamba state-space models can be strong downstream learners,” arXiv preprint arXiv:2406.00209, 2024.[<a href="https://arxiv.org/abs/2406.00209" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Li, Z. Zhang, H. Chen, and Z. Ma, “Mamba: Bringing multidimensional abr to webrtc,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 9262–9270.[<a href="https://dl.acm.org/doi/abs/10.1145/3581783.3611915" target="_blank" rel="noopener noreferrer">Paper</a>]

B. N. Patro and V. S. Agneeswaran, “Simba: Simplified mamba-based architecture for vision and multivariate time series,” arXiv preprint arXiv:2403.15360, 2024.[<a href="https://arxiv.org/abs/2403.15360" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/badripatro/simba" target="_blank" rel="noopener noreferrer">code</a>]


### 大模型分类及其在通信中的应用
#### 1、大语言模型LLM

<table><thead>
  <tr>
    <th>Category</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td rowspan=3>GPT系列</td>
    <td>GPT-3</td>
    <td>2020</td>
    <td><a href="https://splab.sdu.edu.cn/GPT3.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/openai/gpt-3/blob/master/overlap_frequency.md" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>GPT-4</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2312.00752" target="_blank" rel="noopener noreferrer">Paper</a></td>
   <td><a href="https://github.com/state-spaces/mamba" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>OpenAI o1</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Gemma 系列</td>
    <td>Gemma 1</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2403.08295" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>Gemma 2</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2408.00118" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/google-deepmind/dangerous-capability-evaluations" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>LLaMA 系列</td>
    <td>LLaMA-2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/llama" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LLaMA-3</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
</tbody></table>




#### 2、大视觉模型LVM
<table><thead>
  <tr>
    <th>Category</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td rowspan=2>SAM 系列</td>
    <td>SAM-1</td>
    <td>2023</td>
    <td><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>SAM-2</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2408.08315" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/983632847/sam-for-videos" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>DINO 系列</td>
    <td>DINO V1</td>
    <td>2021</td>
    <td><a href="https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/dino" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>DINO V2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2304.07193" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/dinov2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=3>Stable Diffusion 系列</td>
    <td> Stable Diffusion V1</td>
    <td>2022</td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/CompVis/latent-diffusion" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Stable Diffusion V2</td>
    <td>2022</td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/CompVis/latent-diffusion" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
   <td>Stable Diffusion V3</td>
   <td>2024</td>
   <td><a href="https://openreview.net/forum?id=FPnUhsQJ5B" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/faiss/wiki/The-index-factory" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
</tbody></table>

#### 3、视觉语言模型VLM
<table><thead>
  <tr>
    <th>Category</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>LLaVA</td>
    <td>LLaVA</td>
    <td>2024</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/haotian-liu/LLaVA" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Qwen-VL 系列</td>
    <td> Qwen-VL</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2308.12966" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/QwenLM/Qwen-VL" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Qwen-VL-Chat</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2308.12966" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/QwenLM/Qwen-VL" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Mini-GPT4</td>
    <td>Mini-GPT4</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2304.10592" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Vision-CAIR/MiniGPT-4" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
</tbody></table>

#### 4、多模态大模型


<table><thead>
  <tr>
    <th>Category</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td rowspan=2>CoDi 系列</td>
    <td>CoDi-1</td>
    <td>2024</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/33edf072fe44f19079d66713a1831550-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/microsoft/i-Code/tree/main/i-Code-V3" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>CoDi-2</td>
    <td>2024</td>
    <td><a href="http://openaccess.thecvf.com/content/CVPR2024/html/Tang_CoDi-2_In-Context_Interleaved_and_Interactive_Any-to-Any_Generation_CVPR_2024_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/microsoft/i-Code/tree/main/CoDi-2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Meta-Transformer</td>
    <td>Meta-Transformer</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2307.10802" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/invictus717/MetaTransformer" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
     <td>ImageBind</td>
     <td>ImageBind</td>
     <td>2023</td>
     <td><a href="http://openaccess.thecvf.com/content/CVPR2023/html/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td></td>
   </tr>
</tbody></table>

#### 5、世界模型
<table><thead>
  <tr>
    <th>Category</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>Sora</td>
    <td>Sora</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2402.17177" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/lichao-sun/SoraReview" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
     <td>JEPA </td>
     <td>JEPA </td>
     <td>2022</td>
     <td><a href="https://openreview.net/pdf?id=BZ5a1r-kVsf" target="_blank" rel="noopener noreferrer">Paper</a></td>
        <td></td>
   </tr>
   <tr align="center">
      <td>Vista</td>
      <td>Vista</td>
      <td>2024</td>
      <td><a href="https://arxiv.org/abs/2405.17398" target="_blank" rel="noopener noreferrer">Paper</a></td>
      <td><a href="https://github.com/OpenDriveLab/Vista" target="_blank" rel="noopener noreferrer">Code</a></td>
    </tr>
</tbody></table>

#### 6、轻量级大模型
<table><thead>
  <tr>
    <th>Category</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>TinyLlama</td>
    <td>TinyLlama</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2401.02385" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/jzhang38/TinyLlama" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
     <td>MobileVLM</td>
     <td>MobileVLM</td>
     <td>2024</td>
     <td><a href="https://arxiv.org/abs/2402.03766" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="https://github.com/Meituan-AutoML/MobileVLM" target="_blank" rel="noopener noreferrer">Code</a></td>
   </tr>
   <tr align="center">
      <td>Mini-Gemini</td>
      <td>Mini-Gemini</td>
      <td>2024</td>
      <td><a href="https://arxiv.org/abs/2403.18814" target="_blank" rel="noopener noreferrer">Paper</a></td>
      <td><a href="https://github.com/dvlab-research/MGM" target="_blank" rel="noopener noreferrer">Code</a></td>
    </tr>
</tbody></table>

### 大模型的优化方法
#### 1、In-context learning(ICL)

M. Zecchin, K. Yu, and O. Simeone, "In-context learning for MIMO equalization using transformer-based sequence models," in *2024 IEEE International Conference on Communications Workshops (ICC Workshops)*, IEEE, 2024, pp. 1573-1578. [<a href="https://arxiv.org/abs/2311.06101" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Abbas, K. Kar, and T. Chen, “Leveraging large language models for wireless symbol detection via in-context learning,” arXiv preprint arXiv:2409.00124, 2024.[<a href="https://arxiv.org/abs/2409.00124" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2、XoT

T. B. Brown, “Language models are few-shot learners,” arXiv preprint arXiv:2005.14165, 2020.[<a href="https://splab.sdu.edu.cn/GPT3.pdf" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L.Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774,2023.[<a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/openai/evals" target="_blank" rel="noopener noreferrer">code</a>]

A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra,A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann et al., “Palm: Scaling language modeling with pathways,” Journal of Machine Learning Research, vol. 24, no. 240, pp. 1–113, 2023.[<a href="https://www.jmlr.org/papers/v24/22-1144.html" target="_blank" rel="noopener noreferrer">Paper</a>]

S. Rasal, “Llm harmony: Multi-agent communication for problem solving,” arXiv preprint arXiv:2401.01312, 2024.[<a href="https://arxiv.org/abs/2401.01312" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/sumedhrasal/simulation" target="_blank" rel="noopener noreferrer">code</a>]

#### 3、检索生成增强(RAG)

A.-L. Bornea, F. Ayed, A. De Domenico, N. Piovesan, and A. Maatouk, “Telco-rag: Navigating the challenges of retrieval-augmented language models for telecommunications,” arXiv preprint arXiv:2404.15939, 2024.[<a href="https://arxiv.org/abs/2404.15939" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Rackauckas, “Rag-fusion: a new take on retrieval-augmented generation,” arXiv preprint arXiv:2402.03367, 2024.[<a href="https://arxiv.org/abs/2402.03367" target="_blank" rel="noopener noreferrer">Paper</a>]

W. Jiang, S. Zhang, B. Han, J. Wang, B. Wang, and T. Kraska,“Piperag: Fast retrieval-augmented generation via algorithm-system co-design,” arXiv preprint arXiv:2403.05676, 2024.[<a href="https://arxiv.org/abs/2403.05676" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Tang and W. Guo, “Automatic retrieval-augmented generation of 6g network specifications for use cases,” arXiv preprint arXiv:2405.03122, 2024.[<a href="https://arxiv.org/abs/2405.03122" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 4、多智能体系统(MAS)

J. Tong, J. Shao, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang,“Wirelessagent: Large language model agents for intelligent wireless networks,” arXiv preprint arXiv:2409.07964, 2024.[<a href="https://arxiv.org/abs/2409.07964" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/weiiguo/wireless-agent" target="_blank" rel="noopener noreferrer">code</a>]

F. Jiang, L. Dong, Y. Peng, K. Wang, K. Yang,C. Pan, D. T. Niyato, and O. A. Dobre, “Large language model enhanced multi-agent systems for 6g communications,” ArXiv, vol. abs/2312.07850, 2023. [<a href="https://ieeexplore.ieee.org/abstract/document/10638533/" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 5、混合专家模型(MoE)

R. Zhang, H. Du, Y. Liu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, and D. I. Kim, “Interactive generative ai agents for satellite networks through a mixture of experts transmission,”arXiv preprint arXiv:2404.09134, 2024.[<a href="https://arxiv.org/abs/2404.09134" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Wang, H. Du, G. Sun, J. Kang, H. Zhou, D. Niyato, and J. Chen,“Optimizing 6g integrated sensing and communications (isac) via expert networks,” arXiv preprint arXiv:2406.00408, 2024.[<a href="https://arxiv.org/abs/2406.00408" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, Y. Fang,D. I. Kim et al., “Integration of mixture of experts and multimodal generative ai in internet of vehicles: A survey,” arXiv preprint arXiv:2404.16356, 2024.[<a href="https://arxiv.org/abs/2404.16356" target="_blank" rel="noopener noreferrer">Paper</a>]

## 大模型在通信领域中的应用
### 大模型在 PHY 和 MAC 层中的设计
#### 1、大模型在 PHY 层的设计
  
N. Van Huynh, J. Wang, H. Du, D. T. Hoang, D. Niyato, D. N.Nguyen, D. I. Kim, and K. B. Letaief, “Generative ai for physical layer communications: A survey,” IEEE Transactions on Cognitive Communications and Networking, 2024.[<a href="https://arxiv.org/abs/2312.05594" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Arvinte and J. I. Tamir, “Mimo channel estimation using score-based generative models,” IEEE Transactions on Wireless Communications, 2022.[<a href="https://ieeexplore.ieee.org/abstract/document/9957135/" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Wang, J. Zhang, H. Du, R. Zhang, D. Niyato, B. Ai, and K. B. Letaief, “Generative ai agent for next-generation mimo design: Fundamentals, challenges, and vision,” arXiv preprint arXiv:2404.08878, 2024.[<a href="https://arxiv.org/abs/2404.08878" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Akrout, A. Mezghani, E. Hossain, F. Bellili, and R. W. Heath,“From multilayer perceptron to gpt: A reflection on deep learning research for wireless physical layer,” IEEE Communications Magazine, vol. 62, no. 7, pp. 34–41, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10582856/" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Fontaine, A. Shahid, and E. De Poorter, “Towards a wireless physical-layer foundation model: Challenges and strategies,” arXiv preprint arXiv:2403.12065, 2024.[<a href="https://arxiv.org/abs/2403.12065" target="_blank" rel="noopener noreferrer">Paper</a>]

### 大模型在资源分配和优化的应用
#### 1、大模型的计算资源分配

H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, D. I. Kim et al.,“Enabling ai-generated content (aigc) services in wireless edge networks,” arXiv preprintarXiv:2301.03220, 2023.[<a href="https://arxiv.org/abs/2301.03220" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, H. Huang, and S. Mao, “Diffusion-based reinforcement learning for edge-enabled ai-generated content services,” IEEE Transactions on Mobile Computing, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10409284/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/Lizonghang/AGOD/blob/c8b6a1df58a1de3c1da4903450c7ae2ec8154ebf/main.py" target="_blank" rel="noopener noreferrer">code</a>]

H. Du, G. Liu, Y. Lin, D. Niyato, J. Kang, Z. Xiong, and D. I. Kim,“Mixture of experts for network optimization: A large language model-enabled approach,” arXiv preprint arXiv:2402.09756, 2024.[<a href="https://arxiv.org/abs/2402.09756" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2、大模型的频谱资源分配

R. Zhang, H. Du, Y. Liu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, and D. I. Kim, “Interactive generative ai agents for satellite networks through a mixture of experts transmission,”arXiv preprint arXiv:2404.09134, 2024.[<a href="https://arxiv.org/abs/2404.09134" target="_blank" rel="noopener noreferrer">Paper</a>]

D. Chen, Q. Qi, Q. Fu, J. Wang, J. Liao, and Z. Han, “Transformer-based reinforcement learning for scalable multi-uav area coverage,”IEEE Transactions on Intelligent Transportation Systems, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10423879/" target="_blank" rel="noopener noreferrer">Paper</a>]

X. Du and X. Fang, “An integrated communication and computing scheme for wi-fi networks based on generative ai and reinforcement learning,” arXiv preprint arXiv:2404.13598, 2024.[<a href="https://arxiv.org/abs/2404.13598" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3、大模型的能量资源优化

M. Xu, D. Niyato, J. Kang, Z. Xiong, S. Guo, Y. Fang, and D. I. Kim, “Generative ai-enabled mobile tactical multimedia networks: Distribution, generation, and perception,” arXiv preprint arXiv:2401.06386, 2024.[<a href="https://arxiv.org/abs/2401.06386" target="_blank" rel="noopener noreferrer">Paper</a>]

### 大模型在网络的设计与管理的应用
#### 1、网络的设计

Y. Huang, H. Du, X. Zhang, D. Niyato, J. Kang, Z. Xiong,S. Wang, and T. Huang, “Large language models for networking:Applications, enabling techniques, and challenges,” arXiv preprint arXiv:2311.17474, 2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10614634/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Zou, Q. Zhao, L. Bariah, M. Bennis, and M. Debbah, “Wireless multi-agent generative ai: From connected intelligence to collective intelligence,” arXiv preprint arXiv:2307.02757, 2023.[<a href="https://arxiv.org/abs/2307.02757" target="_blank" rel="noopener noreferrer">Paper</a>]

L. He, G. Sun, D. Niyato, H. Du, F. Mei, J. Kang, M. Debbah et al., “Generative ai for game theory-based mobile networking,”arXiv preprint arXiv:2404.09699, 2024.[<a href="https://arxiv.org/abs/2404.09699" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2、网络的管理

Y. Du, S. C. Liew, K. Chen, and Y. Shao, “The power of large language models for wireless communication system development:A case study on fpga platforms,” arXiv preprint arXiv:2307.07319,2023.[<a href="https://arxiv.org/abs/2307.07319" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Wang, L. Zhang, Y. Yang, Z. Zhuang, Q. Qi, H. Sun, L. Lu,J. Feng, and J. Liao, “Network meets chatgpt: Intent autonomous management, control and operation,” Journal of Communications and Information Networks, vol. 8, no. 3, pp. 239–255, 2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10272352/" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Hong, J. Wu, and R. Morello, “Llm-twin: Mini-giant model-driven beyond 5g digital twin networking framework with semantic secure communication and computation,” arXiv preprint arXiv:2312.10631, 2023.[<a href="https://www.nature.com/articles/s41598-024-69474-5" target="_blank" rel="noopener noreferrer">Paper</a>]

A. Dandoush, V. Kumarskandpriya, M. Uddin, and U. Khalil,“Large language models meet network slicing management and orchestration,” arXiv preprint arXiv:2403.13721, 2024.[<a href="https://arxiv.org/abs/2403.13721" target="_blank" rel="noopener noreferrer">Paper</a>]

L. Yue and T. Chen, “Ai large model and 6g network,” in 2023 IEEE Globecom Workshops (GC Wkshps). IEEE, 2023, pp. 2049–2054.[<a href="https://ieeexplore.ieee.org/abstract/document/10465211/" target="_blank" rel="noopener noreferrer">Paper</a>]


### 大模型在边缘智能的应用
#### 1、边缘端 AIGC 的学习与应用

Y. Liu, H. Du, D. Niyato, J. Kang, S. Cui, X. Shen, and P. Zhang,“Optimizing mobile-edge ai-generated everything (aigx) services by prompt engineering: Fundamental, framework, and case study,”IEEE Network, 2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10330096/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Du, R. Zhang, D. Niyato, J. Kang, Z. Xiong, D. I. Kim,X. S. Shen, and H. V. Poor, “Exploring collaborative distributed diffusion-based ai-generated content (aigc) in wireless networks,”IEEE Network, 2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10172151/" target="_blank" rel="noopener noreferrer">Paper</a>]

G. Sun, W. Xie, D. Niyato, H. Du, J. Kang, J. Wu, S. Sun, and P. Zhang, “Generative ai for advanced uav networking,” arXiv preprint arXiv:2404.10556, 2024.[<a href="https://arxiv.org/abs/2404.10556" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, Y. Wu, D. Cai, X. Li, and S. Wang, “Federated fine-tuning of billion-sized language models across mobile devices,”arXiv preprint arXiv:2308.13894, 2023.[<a href="https://www.caidongqi.com/pdf/arXiv-FwdLLM.pdf" target="_blank" rel="noopener noreferrer">Paper</a>]

R. Zhang, K. Xiong, H. Du, D. Niyato, J. Kang, X. Shen, and H. V.Poor, “Generative ai-enabled vehicular networks: Fundamentals,framework, and case study,” IEEE Network, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10506539/" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Chen, H. H. Yang, Y. Tay, K. F. E. Chong, and T. Q. Quek,“The role of federated learning in a wireless world with foundation models,” IEEE Wireless Communications, vol. 31, no. 3, pp. 42–49,2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10558823/" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Zhang, Z. Wei, B. Liu, X. Wang, Y. Yu, and R. Zhang, “Cloud-edge-terminal collaborative aigc for autonomous driving,” IEEE Wireless Communications, vol. 31, no. 4, pp. 40–47, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10628024/" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2、边缘端大模型资源管理与调度
 
J. Wang, H. Du, D. Niyato, J. Kang, Z. Xiong, D. I. Kim, and K. B.Letaief, “Toward scalable generative ai via mixture of experts in mobile edge networks,” arXiv preprint arXiv:2402.06942, 2024.[<a href="https://arxiv.org/abs/2402.06942" target="_blank" rel="noopener noreferrer">Paper</a>]

O. Friha, M. A. Ferrag, B. Kantarci, B. Cakmak, A. Ozgun, and N. Ghoualmi-Zine, “Llm-based edge intelligence: A comprehensive survey on architectures, applications, security and trustworthiness,” IEEE Open Journal of the Communications Society, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10669603/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Du, R. Zhang, D. Niyato, J. Kang, Z. Xiong, S. Cui,X. Shen, and D. I. Kim, “User-centric interactive ai for distributed diffusion model-based ai-generated content,” arXiv preprint arXiv:2311.11094, 2023.[<a href="https://arxiv.org/abs/2311.11094" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Shen, J. Shao, X. Zhang, Z. Lin, H. Pan, D. Li, J. Zhang, and K. B. Letaief, “Large language models empowered autonomous edge ai for connected intelligence,” IEEE Communications Mag-
azine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10384606" target="_blank" rel="noopener noreferrer">Paper</a>]

L. Dong, F. Jiang, Y. Peng, K. Wang, K. Yang, C. Pan, and R. Schober, “Lambo: Large language model empowered edge intelligence,” arXiv preprint arXiv:2308.15078, 2023.[<a href="https://arxiv.org/abs/2308.15078" target="_blank" rel="noopener noreferrer">Paper</a>]

T. Zhou, J. Yu, J. Zhang, and D. H. Tsang, “Federated promptbased decision transformer for customized vr services in mobile edge computing system,” arXiv preprint arXiv:2402.09729, 2024.[<a href="https://arxiv.org/abs/2402.09729" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/xuyanyu-shh/VR-EyeTracking" target="_blank" rel="noopener noreferrer">Code</a>]

S. Zhang, Q. Liu, K. Chen, B. Di, H. Zhang, W. Yang, D. Niyato,Z. Han, and H. V. Poor, “Large models for aerial edges: An edge-cloud model evolution and communication paradigm,” IEEE Journal on Selected Areas in Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10681129" target="_blank" rel="noopener noreferrer">Paper</a>]

B. Lai, J. Wen, J. Kang, H. Du, J. Nie, C. Yi, D. I. Kim, and S. Xie, “Resource-efficient generative mobile edge networks in 6g era: Fundamentals, framework and case study,” IEEE Wireless Communications, vol. 31, no. 4, pp. 66–74, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10628023" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3、边缘端大模型跨域协同与融合
（1）与无线感知技术的融合

J. Wang, H. Du, D. Niyato, J. Kang, Z. Xiong, D. Rajan, S. Mao,and X. Shen, “A unified framework for guiding generative ai with wireless perception in resource constrained mobile edge networks,”IEEE Transactions on Mobile Computing, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10472660" target="_blank" rel="noopener noreferrer">Paper</a>]

（2）与强化学习技术的融合

H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, H. Huang, and S. Mao, “Diffusion-based reinforcement learning for edge-enabled ai-generated content services,” IEEE Transactions on Mobile Computing, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10409284" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/Lizonghang/AGOD" target="_blank" rel="noopener noreferrer">Code</a>]

M. Xu, D. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, “Joint foundation model caching and inference of generative ai services for edge intelligence,” in GLOBECOM 2023-2023 IEEE Global Communications Conference. IEEE, 2023, pp. 35483553.[<a href="https://ieeexplore.ieee.org/abstract/document/10436771" target="_blank" rel="noopener noreferrer">Paper</a>]

（3）与缓存和推理技术的融合

——, “Cached model-as-a-resource: Provisioning large language model agents for edge intelligence in space-air-ground integrated networks,” 2024.[<a href="https://arxiv.org/abs/2403.05826" target="_blank" rel="noopener noreferrer">Paper</a>]

### 大模型在语义通信的应用
#### 1、AIGC 增强的语义通信系统
（1）基于扩散模型的语义通信优化

F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,“Large ai model-based semantic communications,” IEEE Wireless Communications, vol. 31, no. 3, pp. 68–75, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10558819" target="_blank" rel="noopener noreferrer">Paper</a>]

（2）基于 Transformer 的语义增强和推理

F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,“Large generative model assisted 3d semantic communication,”arXiv preprint arXiv:2403.05783, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10436771" target="_blank" rel="noopener noreferrer">Paper</a>]

（3）基于 LLM 的语义通信优化

T. Wu, Z. Chen, D. He, L. Qian, Y. Xu, M. Tao, and W. Zhang,“Cddm: Channel denoising diffusion models for wireless semantic communications,” IEEE Transactions on Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10480348" target="_blank" rel="noopener noreferrer">Paper</a>]

F. Ni, B. Wang, R. Li, Z. Zhao, and H. Zhang, “Interplay of semantic communication and knowledge learning,” arXiv preprint arXiv:2402.03339, 2024.[<a href="https://onlinelibrary.wiley.com/doi/abs/10.1002/9781394223336.ch5" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Wang, L. Zou, S. Wei, F. Liao, J. Zhuo, H. Mi, and R. Lai, “Large language model enabled semantic communication systems,” arXiv preprint arXiv:2407.14112, 2024.[<a href="https://arxiv.org/abs/2407.14112" target="_blank" rel="noopener noreferrer">Paper</a>]

（4）基于大视觉模型的语义通信

Y. Zhao, Y. Yue, S. Hou, B. Cheng, and Y. Huang, “Lamosc: Large language model-driven semantic communication system for visual transmission,” IEEE Transactions on Cognitive Communications and Networking, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10531769" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Kim, S. Seo, J. Park, M. Bennis, S.-L. Kim, and J. Choi,“Knowledge distillation from language-oriented to emergent communication for multi-agent remote control,” arXiv preprintarXiv:2401.12624, 2024.[<a href="https://arxiv.org/abs/2401.12624" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2、智能体驱动的语义通信系统

W. Yang, Z. Xiong, Y. Yuan, W. Jiang, T. Q. Quek, and M. Debbah, “Agent-driven generative semantic communication for remote surveillance,” arXiv preprint arXiv:2404.06997, 2024.[<a href="https://www.techrxiv.org/doi/full/10.36227/techrxiv.172165653.34559702" target="_blank" rel="noopener noreferrer">Paper</a>]

F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, D. Niyato,and O. A. Dobre, “Large language model enhanced multi-agent systems for 6g communications,” IEEE Wireless Communications,2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10638533" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3、语义通信与无线感知

J. Wang, H. Du, D. Niyato, Z. Xiong, J. Kang, S. Mao, and X. S. Shen, “Guiding ai-generated digital content with wireless perception,” IEEE Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10515205" target="_blank" rel="noopener noreferrer">Paper</a>]

### 大模型在安全隐私的应用
#### 1、网络安全威胁检测与防御
（1）后门攻击防御 

H. Yang, K. Xiang, M. Ge, H. Li, R. Lu, and S. Yu, “A comprehen-
sive overview of backdoor attacks in large language models within
communication networks,” IEEE Network, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10440367" target="_blank" rel="noopener noreferrer">Paper</a>]

（2）网络威胁检测

M. A. Ferrag, M. Ndhlovu, N. Tihanyi, L. C. Cordeiro, M. Debbah,T. Lestable, and N. S. Thandi, “Revolutionizing cyber threat detection with large language models: A privacy-preserving bert based lightweight model for iot/iiot devices,” IEEE Access, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10423646" target="_blank" rel="noopener noreferrer">Paper</a>]

Ferrag M A, Ndhlovu M, Tihanyi N, et al. Revolutionizing cyber threat detection with large language models[J]. arXiv preprint arXiv:2306.14263, 2023.[<a href="https://arxiv.org/abs/2306.14263" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Wang, Y. Li, Q. Qi, Y. Lu, and B. Wu, “Multilayered fault detection and localization with transformer for microservice systems,”IEEE Transactions on Reliability, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10423414" target="_blank" rel="noopener noreferrer">Paper</a>]

（3）软件漏洞检测

M. A. Ferrag, A. Battah, N. Tihanyi, M. Debbah, T. Lestable, and L. C. Cordeiro, “Securefalcon: The next cyber reasoning system for cyber security,” arXiv preprint arXiv:2307.06616, 2023.[<a href="https://arxiv.org/abs/2307.06616" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/cdpierse/transformers-interpret" target="_blank" rel="noopener noreferrer">Code</a>]


#### 2、通信网络中的可信 AI

H. Luo, J. Luo, and A. V. Vasilakos, “Bc4llm: Trusted artificial intelligence when blockchain meets large language models,” arXiv preprint arXiv:2310.06278, 2023.[<a href="https://arxiv.org/abs/2310.06278" target="_blank" rel="noopener noreferrer">Paper</a>]


H. Du, D. Niyato, J. Kang, Z. Xiong, K.-Y. Lam, Y. Fang,and Y. Li, “Spear or shield: Leveraging generative ai to tackle security threats of intelligent network services,” arXiv preprint arXiv:2306.02384, 2023.[<a href="https://arxiv.org/abs/2306.02384" target="_blank" rel="noopener noreferrer">Paper</a>]

C. T. Nguyen, Y. Liu, H. Du, D. T. Hoang, D. Niyato, D. N. Nguyen, and S. Mao, “Generative ai-enabled blockchain networks:Fundamentals, applications, and case study,” IEEE Network, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10552807" target="_blank" rel="noopener noreferrer">Paper</a>]


### 大模型在新兴应用的应用
#### 1、数字孪生
N. Sehad, L. Bariah, W. Hamidouche, H. Hellaoui, R. Jäntti,and M. Debbah, “Generative ai for immersive communication:The next frontier in internet-of-senses through 6g,” arXiv preprintarXiv:2404.01713, 2024.[<a href="https://arxiv.org/abs/2404.01713" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://arxiv.org/format/2404.01713" target="_blank" rel="noopener noreferrer">Code</a>]

Y. Xia, M. Shenoy, N. Jazdi, and M. Weyrich, “Towards au-tonomous system: flexible modular production system enhanced with large language model agents,” in 2023 IEEE 28th Interna-tional Conference on Emerging Technologies and Factory Automa-tion (ETFA). IEEE, 2023, pp. 1–8.[<a href="https://ieeexplore.ieee.org/abstract/document/10275362" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/YuchenXia/GPT4IndustrialAutomation" target="_blank" rel="noopener noreferrer">Code</a>]

N. Das, A. Kotal, D. Roseberry, and A. Joshi, “Change manage-ment using generative modeling on digital twins,” in 2023 IEEE International Conference on Intelligence and Security Informatics(ISI). IEEE, 2023, pp. 1–6.[<a href="https://ieeexplore.ieee.org/abstract/document/10297181" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Tao, W. Xu, Y. Huang, X. Wang, and X. You, “Wireless network digital twin for 6g: Generative ai as a key enabler,” IEEE Wireless Communications, vol. 31, no. 4, pp. 24–31, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10628026" target="_blank" rel="noopener noreferrer">Paper</a>]

C. Zhao, H. Du, D. Niyato, J. Kang, Z. Xiong, D. I. Kim, K. B.Letaief et al., “Generative ai for secure physical layer communica-tions: A survey,” arXiv preprint arXiv:2402.13553, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10623395" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2、智慧医疗
J. Chen, C. Yi, H. Du, D. Niyato, J. Kang, J. Cai, and X. Shen,“A revolution of personalized healthcare: Enabling human digital twin with mobile aigc,” IEEE Network, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10438453" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3、元宇宙
M. Xu, D. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao,and Z. Han, “Sparks of gpts in edge intelligence for metaverse:caching and inference for mobile aigcservices,” arXiv preprint arXiv:2304.08782, 2023.[<a href="https://arxiv.org/abs/2304.08782" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Lin, Z. Gao, H. Du, D. Niyato, J. Kang, A. Jamalipour, andX. S. Shen, “A unified framework for integrating semantic communication and aigenerated content in metaverse,” IEEE Network,2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10273254" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, H. Huang, and S. Mao,“Generative ai-aided optimization for ai-generated content (aigc) services in edge networks,” arXiv preprint arXiv:2303.13052, 2023.[<a href="https://arxiv.org/abs/2303.13052v1" target="_blank" rel="noopener noreferrer">Paper</a>]

G. Liu, H. Du, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour,S. Mao, and D. I. Kim, “Fusion of mixture of experts and generative artificial intelligence in mobile edge metaverse,” arXiv preprint arXiv:2404.03321, 2024.[<a href="https://arxiv.org/abs/2404.03321" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, D. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, “Generative ai-empowered effective physical-virtual synchronization in the vehicular metaverse,” in 2023 IEEE Inter-national Conference on Metaverse Computing, Networking and Applications (MetaCom). IEEE, 2023, pp. 607–611.[<a href="https://ieeexplore.ieee.org/abstract/document/10271797" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Wen, R. Zhang, D. Niyato, J. Kang, H. Du, Y. Zhang, and Z. Han, “Generative ai for low-carbon artificial intelligence of things,” arXiv preprint arXiv:2404.18077, 2024.[<a href="https://arxiv.org/abs/2404.18077" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/mlco2/codecarbon" target="_blank" rel="noopener noreferrer">Code</a>] 
#### 4、其他领域

M. Abbasian, I. Azimi, A. M. Rahmani, and R. Jain, “Conversational health agents: A personalized llm-powered agent framework,” arXiv preprint arXiv:2310.02374, 2023.[<a href="https://arxiv.org/abs/2310.02374" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/Institute4FutureHealth/CHA" target="_blank" rel="noopener noreferrer">Code</a>] 

H. Wen, Y. Li, G. Liu, S. Zhao, T. Yu, T. J.-J. Li, S. Jiang, Y. Liu,Y. Zhang, and Y. Liu, “Autodroid: Llm-powered task automation in android,” in Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024, pp. 543–557.[<a href="https://dl.acm.org/doi/abs/10.1145/3636534.3649379" target="_blank" rel="noopener noreferrer">Paper</a>]

S. Javaid, R. A. Khalil, N. Saeed, B. He, and M.-S. Alouini,“Leveraging large language models for integrated satellite-aerialterrestrial networks: recent advances and future directions,” arXiv preprint arXiv:2407.04581, 2024.[<a href="https://arxiv.org/abs/2407.04581" target="_blank" rel="noopener noreferrer">Paper</a>]

S. Javaid, H. Fahim, B. He, and N. Saeed, “Large language models for uavs: Current state and pathways to the future,” IEEE Open Journal of Vehicular Technology, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10643253" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Cui, Y. Du, Q. Yang, Y. Shao, and S. C. Liew, “Llmind:Orchestrating ai and iot with llm for complex task execution,”IEEE Communications Magazine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10697418" target="_blank" rel="noopener noreferrer">Paper</a>]

L. Bariah, H. Zou, Q. Zhao, B. Mouhouche, F. Bader, and M. Debbah, “Understanding telecom language through large language models,” in GLOBECOM 2023-2023 IEEE Global Communications Conference. IEEE, 2023, pp. 6542–6547.[<a href="https://ieeexplore.ieee.org/abstract/document/10437725" target="_blank" rel="noopener noreferrer">Paper</a>]

## The Team
| Section 	| Student Contributors 	|
|:-------:	|:--------------------:	|
|         	|                      	|
|         	|                      	|
|         	|                      	|


## Acknowledgments



## Update Log
| Version 	| Time 	| Update Content 	|
|:-------:	|:----:	|:--------------:	|
|         	|      	|                	|
|         	|      	|                	|
|         	|      	|                	|












