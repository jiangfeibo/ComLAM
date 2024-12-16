
# A Comprehensive Survey of Large AI Models for Communications: Foundations, Applications, and Future Directions

## Abstract
The 6G wireless communications aim to establish an intelligent world of ubiquitous connectivity, providing unprecedented communication experience. Large artificial intelligence models (LAMs), with their outstanding cognitive and generalization capabilities, can efficiently provide artificial intelligence (AI) services for diverse communication applications, making them crucial tools for addressing complex challenges in future wireless communication systems. This study comprehensively reviews the foundations, applications, and future directions of LAMs for communications. First, we introduce the current development state of AI-based communication systems, emphasizing the motivation behind empowering communications with LAMs and summarizing the contributions. Then, we examine the learning foundations of LAMs for communications, encompassing dataset construction, model training, and evaluation. Next, we present the design foundations of LAMs for communications, including key architectures, model classifications, and optimization methods. Following this, we detail the research advancements of LAMs in various communication scenarios. Finally, we analyze the main challenges and summarize potential future directions.

![5b2e3bba7b265e977f6731643077467](https://github.com/user-attachments/assets/b61fcde3-f804-4a9d-ad25-9b4435587238)
<p align="center">Fig. 1: Large AI model empowered Future Wireless Communication Systems.</p>

![4fd0356155dccb0ed625ed033b047d7](https://github.com/user-attachments/assets/c00dd862-2d41-4c55-ae55-0ca3a629b554)
<p align="center">Fig. 2: Overall organization of the survey.</p>

![6726b9d26a50858d841035ca3bc22ac](https://github.com/user-attachments/assets/47175a23-3cc8-43d8-90e4-0505f84aeb92)
<p align="center">Fig. 3: Communication applications of LAMs.</p>

## Contents

* [A Comprehensive Survey of Large AI Models for Communications: Foundations, Applications, and Future Directions](#A-Comprehensive-Survey-of-Large-AI-Models-for-Communications-Foundations-Applications-and-Future-Directions)
  * [Abstract](#Abstract)
  * [Contents](#Contents)
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
  * [Communication datasets for LAMs](#Communication-datasets-for-LAMs)
  * [Classification of LAMs](#Classification-of-LAMs)
  * [共享代码的论文表](#共享代码的论文表)
  * [The Team](#The-Team)
  * [Acknowledgments](#Acknowledgments)
  * [Update log](#Update-log)
  
## III. DESIGN FOUNDATIONS OF LAMS FOR COMMUNICATIONS
### A. Key architecture of LAMs
#### 1) Transformer
Y. Wang, Z. Gao, D. Zheng, S. Chen, D. Gündüz, and H. V.Poor, “Transformer-empowered 6g intelligent networks: From massive mimo processing to semantic communication,” IEEE Wireless Communications, vol. 30, no. 6, pp. 127–135, 2022.[<a href="https://ieeexplore.ieee.org/abstract/document/9961131/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Yoo, T. Jung, L. Dai, S. Kim, and C.-B. Chae, “Real-time semantic communications with a vision transformer,” in 2022 IEEE International Conference on Communications Workshops (ICC Workshops). IEEE, 2022, pp. 1–2.[<a href="https://ieeexplore.ieee.org/abstract/document/9914635/" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Wu, Y. Shao, E. Ozfatura, K. Mikolajczyk, and D. Gündüz,“Transformer-aided wireless image transmission with channel feedback,” IEEE Transactions on Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10500305/" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2) Variational autoencoder
M. A. Alawad, M. Q. Hamdan, and K. A. Hamdi, “Innovative variational autoencoder for an end-to-end communication system,”IEEE Access, 2022.[<a href="https://ieeexplore.ieee.org/abstract/document/9964187/" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Bo, Y. Duan, S. Shao, and M. Tao, “Joint coding-modulation for digital semantic communications via variational autoencoder,”IEEE Transactions on Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10495330/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/SJTU-mxtao/Joint-Coding-Modulation-for-Digital-Semantic-Communication" target="_blank" rel="noopener noreferrer">code</a>]

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
for 6g communications,” IEEE Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10638533" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/jiangfeibo/CommLLM" target="_blank" rel="noopener noreferrer">code</a>]

Y. Wang, Z. Sun, J. Fan, and H. Ma, “On the uses of large language models to design end-to-end learning semantic communication,” in 2024 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2024, pp. 1–6.[<a href="https://ieeexplore.ieee.org/abstract/document/10570717" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, D. Niyato, J. Kang, Z. Xiong, S. Mao, Z. Han, D. I.Kim, and K. B. Letaief, “When large language model agents meet 6g networks: Perception, grounding, and alignment,” IEEE Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10648594" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2) Large vision model
F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,“Large ai model-based semantic communications,” IEEE Wireless Communications, vol. 31, no. 3, pp. 68–75, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10558819" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/jiangfeibo/LAMSC" target="_blank" rel="noopener noreferrer">code</a>]

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
M. Zecchin, K. Yu, and O. Simeone, "In-context learning for MIMO equalization using transformer-based sequence models," in *2024 IEEE International Conference on Communications Workshops (ICC Workshops)*, IEEE, 2024, pp. 1573-1578. [<a href="https://ieeexplore.ieee.pubapi.xyz/document/10615360" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/kclip/ICL-Equalization" target="_blank" rel="noopener noreferrer">code</a>]

M. Abbas, K. Kar, and T. Chen, “Leveraging large language models for wireless symbol detection via in-context learning,” arXiv preprint arXiv:2409.00124, 2024.[<a href="https://arxiv.org/abs/2409.00124" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 2) X of thought
Y. Du, H. Deng, S. C. Liew, K. Chen, Y. Shao, and H. Chen,“The power of large language models for wireless communication system development: A case study on fpga platforms,” arXiv preprint arXiv:2307.07319, 2023.[<a href="https://arxiv.org/abs/2307.07319" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Zou, Q. Zhao, L. Bariah, Y. Tian, M. Bennis, S. Lasaulce,M. Debbah, and F. Bader, “Genainet: Enabling wireless collective intelligence via knowledge transfer and reasoning,” arXiv preprint arXiv:2402.16631, 2024.[<a href="https://arxiv.org/abs/2402.16631" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Shao, J. Tong, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang,“Wirelessllm: Empowering large language models towards wireless intelligence,” arXiv preprint arXiv:2405.17053, 2024.[<a href="https://arxiv.org/abs/2405.17053" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 3) Retrieval-augmented generation
A.-L. Bornea, F. Ayed, A. De Domenico, N. Piovesan, and A. Maatouk, “Telco-rag: Navigating the challenges of retrieval-augmented language models for telecommunications,” arXiv preprint arXiv:2404.15939, 2024.[<a href="https://arxiv.org/abs/2404.15939" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/netop-team/telco-rag" target="_blank" rel="noopener noreferrer">code</a>]

Y. Tang and W. Guo, “Automatic retrieval-augmented generation of 6g network specifications for use cases,” arXiv preprint arXiv:2405.03122, 2024.[<a href="https://arxiv.org/abs/2405.03122" target="_blank" rel="noopener noreferrer">Paper</a>]

X. Huang, Y. Tang, J. Li, N. Zhang, and X. S. Shen, “Toward effective retrieval augmented generative services in 6g networks,” IEEE
Network, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10620276" target="_blank" rel="noopener noreferrer">Paper</a>]

S. Xu, C. K. Thomas, O. Hashash, N. Muralidhar, W. Saad, and N. Ramakrishnan, “Large multi-modal models (lmms) as universal foundation models for ai-native wireless systems,” arXiv preprint arXiv:2402.01748, 2024.[<a href="https://arxiv.org/abs/2402.01748" target="_blank" rel="noopener noreferrer">Paper</a>]

G. Y. GMY, J. A. Ayala-Romero, A. Garcia-Saavedra, and X. Costa-Perez, “Telecomrag: Taming telecom standards with retrieval augmented generation and llms,” Authorea Preprints, 2024.[<a href="https://arxiv.org/abs/2406.07053" target="_blank" rel="noopener noreferrer">Paper</a>]

#### 4) Multi-agent system
J. Tong, J. Shao, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang,“Wirelessagent: Large language model agents for intelligent wireless networks,” arXiv preprint arXiv:2409.07964, 2024.[<a href="https://arxiv.org/abs/2409.07964" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/weiiguo/wireless-agent" target="_blank" rel="noopener noreferrer">code</a>]

F. Jiang, L. Dong, Y. Peng, K. Wang, K. Yang,C. Pan, D. T. Niyato, and O. A. Dobre, “Large language model enhanced multi-agent systems for 6g communications,” ArXiv, vol. abs/2312.07850, 2023. [<a href="https://ieeexplore.ieee.org/abstract/document/10638533/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/jiangfeibo/CommLLM" target="_blank" rel="noopener noreferrer">code</a>]

#### 5) Mixture of experts
R. Zhang, H. Du, Y. Liu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, and D. I. Kim, “Interactive generative ai agents for satellite networks through a mixture of experts transmission,”arXiv preprint arXiv:2404.09134, 2024.[<a href="https://arxiv.org/abs/2404.09134" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/RickyZang/GAI-agent-satellite" target="_blank" rel="noopener noreferrer">code</a>]

J. Wang, H. Du, G. Sun, J. Kang, H. Zhou, D. Niyato, and J. Chen,“Optimizing 6g integrated sensing and communications (isac) via expert networks,” arXiv preprint arXiv:2406.00408, 2024.[<a href="https://arxiv.org/abs/2406.00408" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, Y. Fang,D. I. Kim et al., “Integration of mixture of experts and multimodal generative ai in internet of vehicles: A survey,” arXiv preprint arXiv:2404.16356, 2024.[<a href="https://arxiv.org/abs/2404.16356" target="_blank" rel="noopener noreferrer">Paper</a>]

## IV. LAMS FOR PHYSICAL LAYER DESIGN
### A. LLM-assisted physical layer design
Z. Xiao, C. Ye, Y. Hu, H. Yuan, Y. Huang, Y. Feng, L. Cai, an J. Chang, “Llm agents as 6g orchestrator: A paradigm for task-oriented physical-layer automation,” arXiv preprint arXiv:2410.03688, 2024.[<a href="https://arxiv.org/abs/2410.03688" target="_blank" rel="noopener noreferrer">Paper</a>]

S. Fan, Z. Liu, X. Gu, and H. Li, “Csi-llm: A novel downlink channel prediction method aligned with llm pre-training,” arXiv preprint
arXiv:2409.00005, 2024.[<a href="https://arxiv.org/abs/2409.00005" target="_blank" rel="noopener noreferrer">Paper</a>]

B. Liu, X. Liu, S. Gao, X. Cheng, and L. Yang, “Llm4cp: Adapting large language models for channel prediction,” arXiv preprint arXiv:2406.14440, 2024.[<a href="https://arxiv.org/abs/2406.14440" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/liuboxun/LLM4CP" target="_blank" rel="noopener noreferrer">code</a>]

Y. Sheng, K. Huang, L. Liang, P. Liu, S. Jin, and G. Y. Li,“Beam prediction based on large language models,” arXiv preprint arXiv:2408.08707, 2024.[<a href="https://arxiv.org/abs/2408.08707" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Akrout, A. Mezghani, E. Hossain, F. Bellili, and R. W. Heath,“From multilayer perceptron to gpt: A reflection on deep learning research for wireless physical layer,” IEEE Communications Magazine,vol. 62, no. 7, pp. 34–41, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10582856" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Wang, J. Zhang, H. Du, R. Zhang, D. Niyato, B. Ai, and K. B.Letaief, “Generative ai agent for next-generation mimo design: Fundamentals, challenges, and vision,” arXiv preprint arXiv:2404.08878,2024.[<a href="https://arxiv.org/abs/2404.08878" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://zhewang77.github.io/GAIMIMO/" target="_blank" rel="noopener noreferrer">code</a>]

### B. GAI model-assisted physical layer design
N. Van Huynh, J. Wang, H. Du, D. T. Hoang, D. Niyato, D. N.Nguyen, D. I. Kim, and K. B. Letaief, “Generative ai for physical layer communications: A survey,” IEEE Transactions on Cognitive Communications and Networking, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10490142" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Arvinte and J. I. Tamir, “Mimo channel estimation using score-based generative models,” IEEE Transactions on Wireless Communications, 2022.[<a href="https://ieeexplore.ieee.org/abstract/document/9957135" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/utcsilab/score-based-channels" target="_blank" rel="noopener noreferrer">code</a>]

## V. LAMS FOR RESOURCE ALLOCATION AND OPTIMIZATION
### A. Computing resource allocation
H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, H. Huang, and S. Mao,“Diffusion-based reinforcement learning for edge-enabled ai-generated content services,” IEEE Transactions on Mobile Computing, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10409284" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/Lizonghang/AGOD" target="_blank" rel="noopener noreferrer">code</a>]

H. Du, G. Liu, Y. Lin, D. Niyato, J. Kang, Z. Xiong, and D. I. Kim,“Mixture of experts for network optimization: A large language model-enabled approach,” arXiv preprint arXiv:2402.09756, 2024.[<a href="https://arxiv.org/abs/2402.09756" target="_blank" rel="noopener noreferrer">Paper</a>]

### B. Spectrum resource allocation
R. Zhang, H. Du, Y. Liu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, and D. I. Kim, “Interactive generative ai agents for satellite networks through a mixture of experts transmission,” arXiv preprintarXiv:2404.09134, 2024.[<a href="https://ui.adsabs.harvard.edu/abs/2024arXiv240409134Z/abstract" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/RickyZang/GAI-agent-satellite" target="_blank" rel="noopener noreferrer">code</a>]

X. Du and X. Fang, “An integrated communication and computing scheme for wi-fi networks based on generative ai and reinforcement learning,” arXiv preprint arXiv:2404.13598, 2024.[<a href="https://arxiv.org/abs/2404.13598" target="_blank" rel="noopener noreferrer">Paper</a>]

### C. Energy resource optimization
M. Xu, D. Niyato, J. Kang, Z. Xiong, S. Guo, Y. Fang, and D. I. Kim,“Generative ai-enabled mobile tactical multimedia networks: Distribution, generation, and perception,” arXiv preprint arXiv:2401.06386,2024.[<a href="https://arxiv.org/abs/2401.06386" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, D. I. Kim et al., “Enabling ai-generated content (aigc) services in wireless edge networks,” arXiv preprint arXiv:2301.03220, 2023.[<a href="https://arxiv.org/abs/2301.03220" target="_blank" rel="noopener noreferrer">Paper</a>]

## VI. LAMS FOR NETWORK DESIGN AND MANAGEMENT
### A. Network design
Y. Huang, H. Du, X. Zhang, D. Niyato, J. Kang, Z. Xiong, S. Wang,and T. Huang, “Large language models for networking: Applications,enabling techniques, and challenges,” arXiv preprint arXiv:2311.17474,2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10614634" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Zou, Q. Zhao, L. Bariah, M. Bennis, and M. Debbah, “Wireless multi-agent generative ai: From connected intelligence to collective intelligence,” arXiv preprint arXiv:2307.02757, 2023.[<a href="https://arxiv.org/abs/2307.02757" target="_blank" rel="noopener noreferrer">Paper</a>]

L. He, G. Sun, D. Niyato, H. Du, F. Mei, J. Kang, M. Debbah et al., “Generative ai for game theory-based mobile networking,” arXiv preprint arXiv:2404.09699, 2024.[<a href="https://arxiv.org/abs/2404.09699" target="_blank" rel="noopener noreferrer">Paper</a>]

### B. Network management
Y. Du, H. Deng, S. C. Liew, K. Chen, Y. Shao, and H. Chen,“The power of large language models for wireless communication system development: A case study on fpga platforms,” arXiv preprint arXiv:2307.07319, 2023.[<a href="https://arxiv.org/abs/2307.07319" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Wang, L. Zhang, Y. Yang, Z. Zhuang, Q. Qi, H. Sun, L. Lu, J. Feng,and J. Liao, “Network meets chatgpt: Intent autonomous management,control and operation,” Journal of Communications and Information Networks, vol. 8, no. 3, pp. 239–255, 2023.[<a href="https://ieeexplore.ieee.org/abstract/document/10272352" target="_blank" rel="noopener noreferrer">Paper</a>]

A. Dandoush, V. Kumarskandpriya, M. Uddin, and U. Khalil, “Large language models meet network slicing management and orchestration,”
arXiv preprint arXiv:2403.13721, 2024.[<a href="https://arxiv.org/abs/2403.13721" target="_blank" rel="noopener noreferrer">Paper</a>]

L. Yue and T. Chen, “Ai large model and 6g network,” in 2023 IEEE Globecom Workshops (GC Wkshps). IEEE, 2023, pp. 2049–2054.[<a href="https://ieeexplore.ieee.org/abstract/document/10465211" target="_blank" rel="noopener noreferrer">Paper</a>]

## VII. LAMS FOR EDGE INTELLIGENCE
### A. Edge training and application of LAMs
Z. Yu, Z. Wang, Y. Li, H. You, R. Gao, X. Zhou, S. R. Bommu,Y. K. Zhao, and Y. C. Lin, “Edge-llm: Enabling efficient large language model adaptation on edge devices via layerwise unified compression and adaptive layer tuning and voting,” arXiv preprint arXiv:2406.15758, 2024.[<a href="https://arxiv.org/abs/2406.15758" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/GATECH-EIC/Edge-LLM" target="_blank" rel="noopener noreferrer">code</a>]

M. Zhang, J. Cao, X. Shen, and Z. Cui, “Edgeshard: Efficient llm inference via collaborative edge computing,” arXiv preprint arXiv:2405.14371, 2024.[<a href="https://arxiv.org/abs/2405.14371" target="_blank" rel="noopener noreferrer">Paper</a>]

G. Qu, Q. Chen, W. Wei, Z. Lin, X. Chen, and K. Huang, “Mobile edge intelligence for large language models: A contemporary survey,”arXiv preprint arXiv:2407.18921, 2024.[<a href="https://arxiv.org/abs/2407.18921" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, Y. Wu, D. Cai, X. Li, and S. Wang, “Federated fine-tuning of billion-sized language models across mobile devices,” arXiv preprint arXiv:2308.13894, 2023.[<a href="https://arxiv.org/abs/2308.13894" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/UbiquitousLearning/FwdLLM" target="_blank" rel="noopener noreferrer">code</a>]

W. Zhao, W. Jing, Z. Lu, and X. Wen, “Edge and terminal cooperation enabled llm deployment optimization in wireless network,” in 2024 IEEE/CIC International Conference on Communications in China(ICCC Workshops). IEEE, 2024, pp. 220–225.[<a href="https://ieeexplore.ieee.org/abstract/document/10693742" target="_blank" rel="noopener noreferrer">Paper</a>]

A. Khoshsirat, G. Perin, and M. Rossi, “Decentralized llm inference over edge networks with energy harvesting,” arXiv preprint arXiv:2408.15907, 2024.[<a href="https://arxiv.org/abs/2408.15907" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Lin, G. Qu, Q. Chen, X. Chen, Z. Chen, and K. Huang, “Pushing large language models to the 6g edge: Vision, challenges, and opportunities,” arXiv preprint arXiv:2309.16739, 2023.[<a href="https://arxiv.org/abs/2309.16739" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Rong, Y. Mao, X. He, and M. Chen, “Large-scale traffic flow forecast with lightweight llm in edge intelligence,” IEEE Internet of Things Magazine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10742575" target="_blank" rel="noopener noreferrer">Paper</a>]

### B. Edge resource scheduling meets LAMs
O. Friha, M. A. Ferrag, B. Kantarci, B. Cakmak, A. Ozgun, and N. Ghoualmi-Zine, “Llm-based edge intelligence: A comprehensive survey on architectures, applications, security and trustworthiness,”IEEE Open Journal of the Communications Society, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10669603" target="_blank" rel="noopener noreferrer">Paper</a>]

L. Dong, F. Jiang, Y. Peng, K. Wang, K. Yang, C. Pan, and R. Schober,“Lambo: Large language model empowered edge intelligence,” arXiv preprint arXiv:2308.15078, 2023.[<a href="https://arxiv.org/abs/2308.15078" target="_blank" rel="noopener noreferrer">Paper</a>]

B. Lai, J. Wen, J. Kang, H. Du, J. Nie, C. Yi, D. I. Kim, and S. Xie,“Resource-efficient generative mobile edge networks in 6g era: Fundamentals, framework and case study,” IEEE Wireless Communications,vol. 31, no. 4, pp. 66–74, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10628023" target="_blank" rel="noopener noreferrer">Paper</a>]

### C. Cross-domain collaboration of LAMs
J. Wang, H. Du, D. Niyato, J. Kang, Z. Xiong, D. Rajan, S. Mao,and X. Shen, “A unified framework for guiding generative ai with wireless perception in resource constrained mobile edge networks,”IEEE Transactions on Mobile Computing, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10472660" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Chen, R. Li, X. Yu, Z. Zhao, and H. Zhang, “Adaptive layer splitting for wireless llm inference in edge computing: A model-based reinforcement learning approach,” arXiv preprint arXiv:2406.02616,2024.[<a href="https://arxiv.org/abs/2406.02616" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Xu, D. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, “Joint foundation model caching and inference of generative ai services for edge intelligence,” in GLOBECOM 2023-2023 IEEE Global Communications Conference. IEEE, 2023, pp. 3548–3553.[<a href="https://ieeexplore.ieee.org/abstract/document/10436771" target="_blank" rel="noopener noreferrer">Paper</a>]

## VIII. LAMS FOR SEMANTIC COMMUNICATION
### A. LLM-based semantic communication systems
Z. Wang, L. Zou, S. Wei, F. Liao, J. Zhuo, H. Mi, and R. Lai,“Large language model enabled semantic communication systems,”arXiv preprint arXiv:2407.14112, 2024.[<a href="https://arxiv.org/abs/2407.14112" target="_blank" rel="noopener noreferrer">Paper</a>]

F. Jiang, S. Tu, L. Dong, C. Pan, J. Wang, and X. You, “Large generative model-assisted talking-face semantic communication system,”arXiv preprint arXiv:2411.03876, 2024.[<a href="https://arxiv.org/abs/2411.03876" target="_blank" rel="noopener noreferrer">Paper</a>]

W. Chen, W. Xu, H. Chen, X. Zhang, Z. Qin, Y. Zhang, and Z. Han,“Semantic communication based on large language model for underwater image transmission,” arXiv preprint arXiv:2408.12616, 2024.[<a href="https://arxiv.org/abs/2408.12616" target="_blank" rel="noopener noreferrer">Paper</a>]

A. Kalita, “Large language models (llms) for semantic communication in edge-based iot networks,” arXiv preprint arXiv:2407.20970, 2024.[<a href="https://arxiv.org/abs/2407.20970" target="_blank" rel="noopener noreferrer">Paper</a>]

P. Jiang, C.-K. Wen, X. Yi, X. Li, S. Jin, and J. Zhang, “Semantic communications using foundation models: Design approaches and open issues,” IEEE Wireless Communications, vol. 31, no. 3, pp. 76–84,2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10558822" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Wang, Z. Sun, J. Fan, and H. Ma, “On the uses of large language models to design end-to-end learning semantic communication,” in 2024 IEEE Wireless Communications and Networking Conference(WCNC). IEEE, 2024, pp. 1–6.[<a href="https://ieeexplore.ieee.org/abstract/document/10570717" target="_blank" rel="noopener noreferrer">Paper</a>]

### B. Other LAM-based semantic communication systems
F. Zhang, Y. Du, K. Chen, Y. Shao, and S. C. Liew, “Addressing out-of-distribution challenges in image semantic communication systems with multi-modal large language models,” arXiv preprint arXiv:2407.15335,2024.[<a href="https://arxiv.org/abs/2407.15335" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Xie, Z. Qin, X. Tao, and Z. Han, “Towards intelligent communications: Large model empowered semantic communications,” arXiv preprint arXiv:2402.13073, 2024.[<a href="https://arxiv.org/abs/2402.13073" target="_blank" rel="noopener noreferrer">Paper</a>]

W. Yang, Z. Xiong, S. Mao, T. Q. Quek, P. Zhang, M. Debbah, and R. Tafazolli, “Rethinking generative semantic communication for multiuser systems with multi-modal llm,” arXiv preprint arXiv:2408.08765,2024.[<a href="https://arxiv.org/abs/2408.08765" target="_blank" rel="noopener noreferrer">Paper</a>]

F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,“Large generative model assisted 3d semantic communication,” arXiv
preprint arXiv:2403.05783, 2024.[<a href="https://arxiv.org/abs/2403.05783" target="_blank" rel="noopener noreferrer">Paper</a>]

F. Jiang, C. Tang, L. Dong, K. Wang, K. Yang, and C. Pan, “Visual language model based cross-modal semantic communication systems,”
arXiv preprint arXiv:2407.00020, 2024.[<a href="https://arxiv.org/abs/2407.00020" target="_blank" rel="noopener noreferrer">Paper</a>]

T. S. Do, T. P. Truong, T. Do, H. P. Van, and S. Cho, “Lightweight multiuser multimodal semantic communication system for multimodal large language model communication,” Authorea Preprints, 2024.[<a href="https://www.authorea.com/doi/full/10.22541/au.172479430.09168922" target="_blank" rel="noopener noreferrer">Paper</a>]

F. Jiang, L. Dong, Y. Peng, K. Wang, K. Yang, C. Pan, and X. You,“Large ai model empowered multimodal semantic communications,”IEEE Communications Magazine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10670195" target="_blank" rel="noopener noreferrer">Paper</a>]

## IX. LAMS FOR FEDERATED LEARNING
### A. Federated fine-tuning for LAMs
M. Xu, D. Cai, Y. Wu, X. Li, and S. Wang, “Fwdllm: Efficient fedllm using forward gradient,” arXiv preprint arXiv:2308.13894, 2023.[<a href="https://arxiv. org/abs/2308.13894" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/UbiquitousLearning/FwdLLM.git" target="_blank" rel="noopener noreferrer">code</a>]

H. Woisetschlager, A. Erben, S. Wang, R. Mayer, and H.-A. Jacobsen, “Federated fine-tuning of llms on the very edge: The good, the bad, the ugly,” in Proceedings of the Eighth Workshop on Data Management for End-to-End Machine Learning, 2024, pp. 39–50.[<a href="https://dl.acm.org/doi abs/10.1145/3650203.3663331" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Wang, Y. Zhou, Y. Shi, K. Letaief et al., “Federated fine-tuning for pre-trained foundation models over wireless networks,” arXiv preprint arXiv:2407.02924, 2024.[<a href="https://arxiv.org/abs/2407.02924" target="_blank" rel="noopener noreferrer">Paper</a>]

### B. Personlized federated learning for LAMs
F. Jiang, L. Dong, S. Tu, Y. Peng, K. Wang, K. Yang, C. Pan, and D. Niyato, “Personalized wireless federated learning for large language models,” arXiv preprint arXiv:2404.13238, 2024.[<a href="https://arxiv.org/abs/2404.13238" target="_blank" rel="noopener noreferrer">Paper</a>]

Y. Peng, F. Jiang, L. Dong, K. Wang, and K. Yang, “Gai-enabled explainable personalized federated semi-supervised learning,” arXiv preprint arXiv:2410.08634, 2024.[<a href="https://arxiv.org/abs/2410.08634" target="_blank" rel="noopener noreferrer">Paper</a>]

——, “Personalized federated learning for generative ai-assisted semantic communications,” arXiv preprint arXiv:2410.02450, 2024.[<a href="https://arxiv.org/abs/2410.02450" target="_blank" rel="noopener noreferrer">Paper</a>]

## X. LAM-BASED AGENT SYSTEMS
### A. Agent systems based on LLMs
M. Xu, D. Niyato, J. Kang, Z. Xiong, S. Mao, Z. Han, D. I. Kim, and K. B. Letaief, “When large language model agents meet 6g networks: Perception, grounding, and alignment,” IEEE Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10648594/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/Cogito2012/CarCrashDatase" target="_blank" rel="noopener noreferrer">code</a>]

F. Jiang, Y. Peng, L. Dong, K. Wang, K. Yang, C. Pan, D. Niyato, and O. A. Dobre, “Large language model enhanced multi-agent systems for 6g communications,” IEEE Wireless Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10638533/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/jiangfeibo/CommLLM.git" target="_blank" rel="noopener noreferrer">code</a>]

J. Tong, J. Shao, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang, “Wirelessagent: Large language model agents for intelligent wireless networks,” arXiv preprint arXiv:2409.07964, 2024.[<a href="https://arxiv.org/abs/2409.07964" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/weiiguo/Wireless-Agent" target="_blank" rel="noopener noreferrer">code</a>]

H. Zou, Q. Zhao, L. Bariah, M. Bennis, and M. Debbah, “Wireless multi-agent generative ai: From connected intelligence to collective intelligence,” arXiv preprint arXiv:2307.02757, 2023.[<a href="https://arxiv.org/abs/2307.02757" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Wang, J. Zhang, H. Du, R. Zhang, D. Niyato, B. Ai, and K. B. Letaief, “Generative ai agent for next-generation mimo design: Fundamentals, challenges, and vision,” arXiv preprint arXiv:2404.08878,2024.[<a href="https://arxiv.org/abs/2404.08878" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://zhewang77.github.io/GAIMIMO/" target="_blank" rel="noopener noreferrer">code</a>]

R. Zhang, H. Du, Y. Liu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, and D. I. Kim, “Generative ai agents with large language model for satellite networks via a mixture of experts transmission,” IEEE Journal on Selected Areas in Communications, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10679152/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/RickyZang/GAI-agent-satellite" target="_blank" rel="noopener noreferrer">code</a>]

Y. Wang, M. M. Afzal, Z. Li, J. Zhou, C. Feng, S. Guo, and T. Q. Quek, “Large language models for base station siting: Intelligent deployment based on prompt or agent,” arXiv preprint arXiv:2408.03631, 2024.[<a href="https://arxiv.org/abs/2408.03631" target="_blank" rel="noopener noreferrer">Paper</a>]

### B. Agent systems based on other GAI models
W. Yang, Z. Xiong, Y. Yuan, W. Jiang, T. Q. Quek, and M. Debbah, “Agent-driven generative semantic communication for remote surveillance,” arXiv preprint arXiv:2404.06997, 2024.[<a href="https://arxiv.org/abs/2404.06997" target="_blank" rel="noopener noreferrer">Paper</a>]

Z. Chen, Q. Sun, N. Li, X. Li, Y. Wang, and I. Chih-Lin, “Enabling mobile ai agent in 6g era: Architecture and key technologies,” IEEE Network, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10599391/" target="_blank" rel="noopener noreferrer">Paper</a>]

## XI. LAMS FOR SECURITY AND PRIVACY
### A. Network threat detection and defense
H. Yang, K. Xiang, M. Ge, H. Li, R. Lu, and S. Yu, “A comprehensive overview of backdoor attacks in large language models within communication networks,” IEEE Network, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10440367/" target="_blank" rel="noopener noreferrer">Paper</a>]

M. A. Ferrag, M. Ndhlovu, N. Tihanyi, L. C. Cordeiro, M. Debbah, T. Lestable, and N. S. Thandi, “Revolutionizing cyber threat detection with large language models: A privacy-preserving bert-based lightweight model for iot/iiot devices,” IEEE Access, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10423646/" target="_blank" rel="noopener noreferrer">Paper</a>]

J. Wang, Y. Li, Q. Qi, Y. Lu, and B. Wu, “Multilayered fault detection and localization with transformer for microservice systems,” IEEE Transactions on Reliability, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10423414/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/OpenNetAI/Sock-Shop-Dataset" target="_blank" rel="noopener noreferrer">code</a>]

M. A. Ferrag, A. Battah, N. Tihanyi, M. Debbah, T. Lestable, and L. C. Cordeiro, “Securefalcon: The next cyber reasoning system for cyber security,” arXiv preprint arXiv:2307.06616, 2023.[<a href="https://arxiv.org/abs/2307.06616" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/features/copilot/" target="_blank" rel="noopener noreferrer">code</a>]

### B. Trusted AI in communication networks
H. Luo, J. Luo, and A. V. Vasilakos, “Bc4llm: Trusted artificial intelligence when blockchain meets large language models,” arXiv preprint arXiv:2310.06278, 2023.[<a href="https://arxiv.org/abs/2310.06278" target="_blank" rel="noopener noreferrer">Paper</a>]

H. Du, D. Niyato, J. Kang, Z. Xiong, K.-Y. Lam, Y. Fang, and Y. Li, “Spear or shield: Leveraging generative ai to tackle security threats of intelligent network services,” arXiv preprint arXiv:2306.02384, 2023.[<a href="https://arxiv.org/abs/2306.02384" target="_blank" rel="noopener noreferrer">Paper</a>]

S. A. Khowaja, P. Khuwaja, K. Dev, H. A. Hamadi, and E. Zeydan, “Pathway to secure and trustworthy 6g for llms: Attacks, defense, and opportunities,” arXiv preprint arXiv:2408.00722, 2024.[<a href="https://arxiv.org/abs/2408.00722" target="_blank" rel="noopener noreferrer">Paper</a>]

## XII. LAMS FOR DIGITAL TWIN AND METAVERSE
### A. LAMs for digital twin
Y. Xia, M. Shenoy, N. Jazdi, and M. Weyrich, “Towards autonomous system: flexible modular production system enhanced with large language model agents,” in 2023 IEEE 28th International Conference on Emerging Technologies and Factory Automation (ETFA). IEEE, 2023, pp. 1–8.[<a href="https://ieeexplore.ieee.org/abstract/document/10275362/" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/YuchenXia/GPT4IndustrialAutomation" target="_blank" rel="noopener noreferrer">code</a>]

Y. Hong, J. Wu, and R. Morello, “Llm-twin: mini-giant model-driven beyond 5g digital twin networking framework with semantic secure communication and computation,” Scientific Reports, vol. 14, no. 1, p.19065, 2024.[<a href="https://www.nature.com/articles/s41598-024-69474-5" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/CURRYSGITHUB/LLM-twin/tree/main" target="_blank" rel="noopener noreferrer">code</a>]

W. Wu, X. Huang, and T. H. Luan, “Ai-native network digital twin for intelligent network management in 6g,” arXiv preprint arXiv:2410.01584, 2024.[<a href="https://arxiv.org/abs/2410.01584" target="_blank" rel="noopener noreferrer">Paper</a>]

K. Duran, L. V. Cakir, M. Ozdem, K. Gursu, and B. Canberk, “Generative ai-enabled digital twins for 6g-enhanced smart cities,” arXiv preprint arXiv:2411.14222, 2024.[<a href="https://arxiv.org/abs/2411.14222" target="_blank" rel="noopener noreferrer">Paper</a>]

M. Corici, H. Buhr, and T. Magedanz, “Generative twin for 6g and beyond 5g networks: Vision, challenges and architecture,” in 2023 2nd International Conference on 6G Networking (6GNet). IEEE, 2023, pp. 1–6.[<a href="https://ieeexplore.ieee.org/abstract/document/10317780/" target="_blank" rel="noopener noreferrer">Paper</a>]

### B. GAI models for metaverse
G. Liu, H. Du, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, S. Mao, and D. I. Kim, “Fusion of mixture of experts and generative artificial intelligence in mobile edge metaverse,” arXiv preprint arXiv:2404.03321,
2024.[<a href="https://arxiv.org/abs/2404.03321" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/alvinliu97/MOEGAI-Metaverse" target="_blank" rel="noopener noreferrer">code</a>]

M. Xu, D. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, “Generative ai-empowered effective physical-virtual synchronization in the vehicular metaverse,” in 2023 IEEE International Conference on Metaverse Computing, Networking and Applications (MetaCom). IEEE, 2023, pp. 607–611.[<a href="https://ieeexplore.ieee.org/abstract/document/10271797/" target="_blank" rel="noopener noreferrer">Paper</a>]

N. Sehad, L. Bariah, W. Hamidouche, H. Hellaoui, R. Jantti, and M. Debbah, “Generative ai for immersive communication:The next frontier in internet-of-senses through 6g,” arXiv preprint arXiv:2404.01713, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10731639/" target="_blank" rel="noopener noreferrer">Paper</a>]

I. F. Akyildiz, H. Guo, R. Dai, and W. Gerstacker, “Mulsemedia communication research challenges for metaverse in 6g wireless systems,” arXiv preprint arXiv:2306.16359, 2023.[<a href="https://arxiv.org/abs/2306.16359" target="_blank" rel="noopener noreferrer">Paper</a>]

## XIII. LAMS FOR EMERGING APPLICATIONS
### A. Smart healthcare
M. Abbasian, I. Azimi, A. M. Rahmani, and R. Jain, “Conversational health agents: A personalized llm-powered agent framework,” arXiv preprint arXiv:2310.02374, 2023.[<a href="https://arxiv.org/abs/2310.02374" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/Institute4FutureHealth/CHA" target="_blank" rel="noopener noreferrer">code</a>]

### B. Carbon emissions
J. Wen, R. Zhang, D. Niyato, J. Kang, H. Du, Y. Zhang, and Z. Han, “Generative ai for low-carbon artificial intelligence of things,” arXiv preprint arXiv:2404.18077, 2024.[<a href="https://arxiv.org/abs/2404.18077" target="_blank" rel="noopener noreferrer">Paper</a>][<a href="https://github.com/mlco2/codecarbon" target="_blank" rel="noopener noreferrer">code</a>]

### C. Automation systems
H. Wen, Y. Li, G. Liu, S. Zhao, T. Yu, T. J.-J. Li, S. Jiang, Y. Liu, Y. Zhang, and Y. Liu, “Autodroid: Llm-powered task automation in android,” in Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024, pp. 543–557.[<a href="https://dl.acm.org/doi/abs/10.1145/3636534.3649379" target="_blank" rel="noopener noreferrer">Paper</a>]

### D. Artificial intelligence of things
H. Cui, Y. Du, Q. Yang, Y. Shao, and S. C. Liew, “Llmind: Orchestrating ai and iot with llm for complex task execution,” IEEE Communications Magazine, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10697418/" target="_blank" rel="noopener noreferrer">Paper</a>]

### E. Integrated satellite, aerial, and terrestrial networks
S. Javaid, R. A. Khalil, N. Saeed, B. He, and M.-S. Alouini, “Leveraging large language models for integrated satellite-aerial-terrestrial networks: recent advances and future directions,” arXiv preprint arXiv:2407.04581, 2024.[<a href="https://arxiv.org/abs/2407.04581" target="_blank" rel="noopener noreferrer">Paper</a>]

### F. Integration of UAVs and LLMs
S. Javaid, H. Fahim, B. He, and N. Saeed, “Large language models for uavs: Current state and pathways to the future,” IEEE Open Journal of Vehicular Technology, 2024.[<a href="https://ieeexplore.ieee.org/abstract/document/10643253/" target="_blank" rel="noopener noreferrer">Paper</a>]


## Communication datasets for LAMs

<table><thead>
  <tr>
    <th>Category</th>
    <th>datasets</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td rowspan=4>General datasets</td>
    <td>Common Crawl</td>
    <td>2020</td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="http://commoncrawl.org/the-data/get-started/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Pile</td>
    <td>2023</td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/EleutherAI/the-pile" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Dolma</td>
    <td>2024</td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://huggingface.co/datasets/allenai/dolma" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
   <tr align="center">
    <td>RedPajama-data</td>
    <td>2024</td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/togethercomputer/RedPajama-Data" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Communication content filtering</td>
    <td>Common Crawl</td>
    <td>2024</td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="http://commoncrawl.org/the-data/get-started/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>RedPajama</td>
    <td>2024</td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/togethercomputer/RedPajama-Data" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=3>Communication pre-training datasets</td>
    <td>TSpec-LLM</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2406.01768" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>OpenTelecom dataset</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.09424" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Code</a></td>
  <tr align="center">
    <td>TeleQnA dataset</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2310.15051" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://huggingface.co/datasets/netop/TeleQnA" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Communication fine-tuning datasets</td>
    <td>TelecomInstruct dataset</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.09424" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>CSI dataset compliant with 3GPP standards</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2406.14440" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=1>Communication alignment datasets</td>
    <td>TelecomAlign dataset</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.09424" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
</tbody></table>


## Classification of LAMs

<table><thead>
  <tr>
    <th>LAM Category</th>
    <th>Specific Models</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td rowspan=10>Large Language Model</td>
    <td rowspan=5>GPT series</td>
    <td>GPT-1</td>
    <td>2020</td>
    <td><a href="https://hayate-lab.com/wp-content/uploads/2023/05/43372bfa750340059ad87ac8e538c53b.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>GPT-2</td>
    <td>2023</td>
    <td><a href="https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/openai/gpt-2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>GPT-3</td>
    <td>2023</td>
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
    <td><a href="https://github.com/openai/evals" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Gemma series</td>
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
    <td rowspan=3>LLaMA series</td>
    <td>LLaMA-1</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2302.13971" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/llama" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LLaMA-2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/llama" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LLaMA-3</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/openai/tiktoken/tree/main" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>

 
  <tr align="center">
    <td rowspan=7>Large Vision Model</td>
    <td rowspan=2>SAM series</td>
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
    <td rowspan=2>DINO series</td>
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
    <td rowspan=3>Stable Diffusion series</td>
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


  
  <tr align="center">
    <td rowspan=4>Vision Language Model</td>
    <td>LLaVA</td>
    <td>LLaVA</td>
    <td>2024</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/haotian-liu/LLaVA" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Qwen-VL</td>
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


  
  <tr align="center">
    <td rowspan=4>Large Multimodal Model</td>
    <td rowspan=2>CoDi series</td>
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

   
  <tr align="center">
    <td rowspan=3>World Model</td>
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


  <tr align="center">
    <td rowspan=3>Lightweight Large AI Model</td>
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



## 共享代码的论文表
<table><thead>
  <tr>
    <th>Category</th>
    <th>Title</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>Variational autoencoder</td>
    <td>Joint coding-modulation for digital semantic communications via variational autoencoder</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10495330/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/SJTU-mxtao/Joint-Coding-Modulation-for-Digital-Semantic-Communication" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Diffusion models</td>
    <td>Beyond deep reinforcement learning: A tutorial on generative diffusion models in network optimization</td>
    <td><a href="https://arxiv.org/abs/2308.05384" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/hongyangdu/gdmopt" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Large language model</td>
    <td>Large language model enhanced multi-agent systems for 6g communications</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10638533" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/jiangfeibo/CommLLM" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Large vision model</td>
    <td>Large ai model-based semantic communications</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10558819" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/jiangfeibo/LAMSC" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>In-context learning</td>
    <td>In-context learning for MIMO equalization using transformer-based sequence models</td>
    <td><a href="https://ieeexplore.ieee.pubapi.xyz/document/10615360" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/kclip/ICL-Equalization" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Retrieval-augmented generation</td>
    <td>Telco-rag: Navigating the challenges of retrieval-augmented language models for telecommunications</td>
    <td><a href="https://arxiv.org/abs/2404.15939" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/netop-team/telco-rag" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Multi-agent system</td>
    <td>Wirelessagent: Large language model agents for intelligent wireless networks</td>
    <td><a href="https://arxiv.org/abs/2409.07964" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/weiiguo/wireless-agent" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Multi-agent system</td>
    <td>Large language model enhanced multi-agent systems for 6g communications</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10638533/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/jiangfeibo/CommLLM" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Mixture of experts</td>
    <td>Interactive generative ai agents for satellite networks through a mixture of experts transmission</td>
    <td><a href="https://arxiv.org/abs/2404.09134" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/RickyZang/GAI-agent-satellite" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LLM-assisted physical layer design</td>
    <td>Llm4cp: Adapting large language models for channel prediction</td>
    <td><a href="https://arxiv.org/abs/2406.14440" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/liuboxun/LLM4CP" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LLM-assisted physical layer design</td>
    <td>Generative ai agent for next-generation mimo design: Fundamentals, challenges, and vision</td>
    <td><a href="https://arxiv.org/abs/2404.08878" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://zhewang77.github.io/GAIMIMO/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>GAI model-assisted physical layer design</td>
    <td>Mimo channel estimation using score-based generative models</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9957135" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/utcsilab/score-based-channels" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Computing resource allocation</td>
    <td>Diffusion-based reinforcement learning for edge-enabled ai-generated content services</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10409284" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Lizonghang/AGOD" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Spectrum resource allocation</td>
    <td>Interactive generative ai agents for satellite networks through a mixture of experts transmission</td>
    <td><a href="https://ui.adsabs.harvard.edu/abs/2024arXiv240409134Z/abstract" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/RickyZang/GAI-agent-satellite" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Edge training and application of LAMs</td>
    <td>Edge-llm: Enabling efficient large language model adaptation on edge devices via layerwise unified compression and adaptive layer tuning and voting</td>
    <td><a href="https://arxiv.org/abs/2406.15758" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/GATECH-EIC/Edge-LLM" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Edge training and application of LAMs</td>
    <td>Federated fine-tuning of billion-sized language models across mobile devices</td>
    <td><a href="https://arxiv.org/abs/2308.13894" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/UbiquitousLearning/FwdLLM" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Federated fine-tuning for LAMs</td>
    <td>Fwdllm: Efficient fedllm using forward gradient</td>
    <td><a href="https://arxiv. org/abs/2308.13894" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/UbiquitousLearning/FwdLLM.git" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Agent systems based on LLMs</td>
    <td>When large language model agents meet 6g networks: Perception, grounding, and alignment</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10648594/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Cogito2012/CarCrashDatase" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Agent systems based on LLMs</td>
    <td>Large language model enhanced multi-agent systems for 6g communications</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10638533/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/jiangfeibo/CommLLM.git" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Agent systems based on LLMs</td>
    <td>Wirelessagent: Large language model agents for intelligent wireless networks</td>
    <td><a href="https://arxiv.org/abs/2409.07964" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/weiiguo/Wireless-Agent" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Agent systems based on LLMs</td>
    <td>Generative ai agent for next-generation mimo design: Fundamentals, challenges, and vision</td>
    <td><a href="https://arxiv.org/abs/2404.08878" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://zhewang77.github.io/GAIMIMO/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Agent systems based on LLMs</td>
    <td>Generative ai agents with large language model for satellite networks via a mixture of experts transmission</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10679152/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/RickyZang/GAI-agent-satellite" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Network threat detection and defense</td>
    <td>Multilayered fault detection and localization with transformer for microservice systems</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10423414/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/OpenNetAI/Sock-Shop-Dataset" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Network threat detection and defense</td>
    <td>Securefalcon: The next cyber reasoning system for cyber security</td>
    <td><a href="https://arxiv.org/abs/2307.06616" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/features/copilot/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LAMs for digital twin</td>
    <td>Towards autonomous system: flexible modular production system enhanced with large language model agents</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10275362/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/YuchenXia/GPT4IndustrialAutomation" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LAMs for digital twin</td>
    <td>Llm-twin: mini-giant model-driven beyond 5g digital twin networking framework with semantic secure communication and computation</td>
    <td><a href="https://www.nature.com/articles/s41598-024-69474-5" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/CURRYSGITHUB/LLM-twin/tree/main" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>GAI models for metaverse</td>
    <td>Fusion of mixture of experts and generative artificial intelligence in mobile edge metaverse</td>
    <td><a href="https://arxiv.org/abs/2404.03321" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/alvinliu97/MOEGAI-Metaverse" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Smart healthcare</td>
    <td>Conversational health agents: A personalized llm-powered agent framework</td>
    <td><a href="https://arxiv.org/abs/2310.02374" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Institute4FutureHealth/CHA" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Carbon emissions</td>
    <td>Generative ai for low-carbon artificial intelligence of things</td>
    <td><a href="https://arxiv.org/abs/2404.18077" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/mlco2/codecarbon" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
</tbody></table>




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












