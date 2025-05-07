
# A Comprehensive Survey of Large AI Models for Future Communications: Foundations, Applications and Challenges
## Authors
### Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Merouane Debbah, Dusit Niyato, Zhu Han
## Paper
### https://arxiv.org/abs/2505.03556
## Code
### https://github.com/jiangfeibo/ComLAM
## Abstract
The 6G wireless communications aim to establish an intelligent world of ubiquitous connectivity, providing an unprecedented communication experience. Large artificial intelligence models (LAMs) are characterized by significantly larger scales (e.g.,billions or trillions of parameters) compared to typical artificial intelligence (AI) models. LAMs exhibit outstanding cognitive abilities, including strong generalization capabilities for fine-tuning to downstream tasks, and emergent capabilities to handle tasks unseen during training. Therefore, LAMs efficiently provide AI services for diverse communication applications, making them crucial tools for addressing complex challenges in future wireless communication systems. This study provides a comprehensive review of the foundations, applications, and challenges of LAMs in communication. First, we introduce the current state of AI-based communication systems, emphasizing the motivation behind integrating LAMs into communications and summarizing the key contributions. We then present an overview of the essential concepts of LAMs in communication. This includes an introduction to the main architectures of LAMs, such as transformer, diffusion models, and mamba. We also explore the classification of LAMs,including large language models (LLMs), large vision models (LVMs), large multimodal models (LMMs), and world models,and examine their potential applications in communication.Additionally, we cover the training methods and evaluation techniques for LAMs in communication systems. Lastly, we introduce optimization strategies such as chain of thought (CoT), retrieval augmented generation (RAG), and agentic systems. Following this,we discuss the research advancements of LAMs across various communication scenarios, including physical layer design, resource allocation and optimization, network design and management,edge intelligence, semantic communication, agentic systems, and emerging applications. Finally, we analyze the challenges in the current research and provide insights into potential future research
directions.


## Contents

* [A Comprehensive Survey of Large AI Models for Communications: Foundations, Applications, and Challenges](#A-Comprehensive-Survey-of-Large-AI-Models-for-Communications-Foundations-Applications-and-Challenges)
  * [Abstract](#Abstract)
  * [Contents](#Contents)
  * [II. FOUNDATIONS OF LAMS FOR COMMUNICATIONS](#III-FOUNDATIONS-OF-LAMS-FOR-COMMUNICATIONS)
    * [A. Key architecture of LAMs](#A-Key-architecture-of-LAMs)
      * [1) Transformer model](#1-Transformer-model)
      * [2) Diffusion model](#2-Diffusion-model)
      * [3) Mamba model](#3-Mamba-models)
    * [B. Classification of LAMs](#B-Classification-of-LAMs)
      * [1) Large language model](#1-Large-language-model)
      * [2) Large vision model](#2-Large-vision-model)
      * [3) Large multimodal model](#3-Large-multimodal-model)
      * [4) World model](#4-World-model)
    * [C. Training of LAMs for communications](#C-Training-of-LAMs-for-communications)
      * [1) Pre-training of LAMs for communications](#1-Pre-training-of-LAMs-for-communications)
      * [2) Fine-tuning of LAMs for communications](#2-Fine-tuning-of-LAMs-for-communications)
      * [3) Alignment of LAMs for communications](#3-Alignment-of-LAMs-for-communications)
    * [D. Evaluation of LAMs for communications](#D-Evaluation-of-LAMs-for-communications)
      * [1) Communication Q&A](#1-Communication-Q&A)
      * [2) Communication tool learning](#2-Communication-tool-learning)
      * [3) Communication modeling](#3-Communication-modeling)
      * [4) Communication code design](#4-Communication-code-design)
    * [E. Optimization of LAMs](#E-Optimization-of-LAMs)
      * [1) Chain of thought](#1-Chain-of-thought)
      * [2) Retrieval-augmented generation](#2-Retrieval-augmented-generation)
      * [3) Agentic system](#4-Agentic-system)
    * [F. Summary and lessons learned](#F-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [III. LAMS FOR PHYSICAL LAYER DESIGN](#IV-LAMS-FOR-PHYSICAL-LAYER-DESIGN)
    * [A. Channel and beam prediction based on LAMs](#A-Channel-and-beam-prediction-based-on-LAMs)
    * [B. Automated physical layer design based on LAM](#B-Automated-physical-layer-design-based-on-LAM)
    * [C. Summary and lessons learned](#C-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [IV. LAMS FOR RESOURCE ALLOCATION AND OPTIMIZATION](#IV-LAMS-FOR-RESOURCE-ALLOCATION-AND-OPTIMIZATION)
    * [A. Computing resource allocation](#A-Computing-resource-allocation)
    * [B. Spectrum resource allocation](#B-pectrum-resource-allocation)
    * [C. Energy resource optimization](#C-Energy-resource-optimization)
    * [D. Summary and lessons learned](#C-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [V. LAMS FOR NETWORK DESIGN AND MANAGEMENT](#V-LAMS-FOR-NETWORK-DESIGN-AND-MANAGEMENT)
    * [A. Network design](#A-Network-design)
    * [B. Network management](#B-Network-management)
    * [C. Summary and lessons learned](#C-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [VI. LAMS FOR EDGE INTELLIGENCE](#VI-LAMS-FOR-EDGE-INTELLIGENCE)
    * [A. Edge training and application of LAMs](#A-Edge-training-and-application-of-LAMs)
    * [B. Edge resource scheduling meets LAMs](#B-Edge-resource-scheduling-meets-LAMs)
    * [C. Federated learning of LAMs](#C-Federated-learning-of-LAMs)
    * [D. Summary and lessons learned](#C-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [VII. LAMS FOR SEMANTIC COMMUNICATION](#VII-LAMS-FOR-SEMANTIC-COMMUNICATION)
    * [A. LLM-based semantic communication systems](#A-LLM-based-semantic-communication-systems)
    * [B. Other LAM-based semantic communication systems](#B-Other-LAM-based-semantic-communication-systems)
    * [C. Summary and lessons learned](#C-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [VIII. LAM-BASED AGENTIC SYSTEMS](#VIII-LAM-BASED-AGENTIC-SYSTEMS)
    * [A. Agentic systems based on LLMs](#Agentic-systems-based-on-LLMs)
    * [B. Agentic systems based on other GAI models](#B-Agentic-systems-based-on-other-GAI-models)
    * [C. Summary and lessons learned](#C-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [IX. LAMS FOR EMERGING APPLICATIONS](#IX-LAMS-FOR-EMERGING-APPLICATIONS)
    * [A. Smart healthcare](#A-Smart-healthcare)
    * [B. Carbon emissions](#B-Carbon-emissions)
    * [C. Digital twins](#C-Digital-twins)
    * [D. Artificial intelligence of things](#D-Artificial-intelligence-of-things)
    * [E. Integrated satellite, aerial, and terrestrial networks](#E-Integrated-satellite-aerial-and-terrestrial-networks)
    * [F. Integration of UAVs and LLMs](#F-Integration-of-UAVs-and-LLMs)
    * [G. Summary and lessons learned](#F-Summary-and-lessons-learned)
  * [X. RESEARCH CHALLENGES](#X-RESEARCH-CHALLENGES)
    * [1). Lack of high-quality communication data](#1-Lack-of-high-quality-communication-data)
    * [2). Lack of structured communication knowledge](#2-Lack-of-structured-communication-knowledge)
    * [3). Generative hallucination in communication](#3-Generative-hallucination-in-communication)
    * [4). Limitations of reasoning ability](#4-Limitations-of-reasoning-ability)
    * [5). Poor explainability in LAMs](#5-Poor-explainability-in-LAMs)
    * [6). Adaptability in dynamic environments](#6-Adaptability-in-dynamic-environments)
    * [7). Diversity of communication tasks](#7-Diversity-of-communication-tasks)
    * [8). Resource constraints at the edge](#8-Resource-constraints-at-the-edge)
    * [9). High inference latency](#9-High-inference-latency)
    * [10). Security and privacy](#10-Security-and-privacy)
  * [Communication datasets for LAMs](#Communication-datasets-for-LAMs)
  * [Classification of LAMs](#Classification-of-LAMs)
  * [Paper With Code](#共享代码的论文表)
  * [The Team](#The-Team)
  * [Acknowledgments](#Acknowledgments)
  * [Update log](#Update-log)

<div align="center">
 
![fig.png](https://github.com/jiangfeibo/ComLAM/blob/main/fig/fig.png)
<p align="center">Fig. 1: The development history of LAMs.</p>

</div>

<div align="center">

![fig2.png](https://github.com/jiangfeibo/ComLAM/blob/main/fig/fig2.png)
<p align="center">Fig. 2: The role of LAMs in AI.</p>

</div>

![fig3.png](https://github.com/jiangfeibo/ComLAM/blob/main/fig/fig3.png)
<p align="center">Fig. 3: Overall organization of the survey.</p>

![fig4.png](https://github.com/jiangfeibo/ComLAM/blob/main/fig/fig4.png)
<p align="center">Fig. 4: Applications of LAMs in Communication. LAMs can be applied across various domains in communication, including
physical layer design, resource allocation and optimization, network design and management, edge intelligence, semantic
communication, agentic systems, and emerging applications.</p>


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
    <td></td>
    <td><a href="http://commoncrawl.org/the-data/get-started/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Pile</td>
    <td>2023</td>
    <td></td>
    <td><a href="https://github.com/EleutherAI/the-pile" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Dolma</td>
    <td>2024</td>
    <td></td>
    <td><a href="https://huggingface.co/datasets/allenai/dolma" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
   <tr align="center">
    <td>RedPajama-data</td>
    <td>2024</td>
    <td></td>
    <td><a href="https://github.com/togethercomputer/RedPajama-Data" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Communication content filtering</td>
    <td>Common Crawl</td>
    <td>2024</td>
    <td></td>
    <td><a href="http://commoncrawl.org/the-data/get-started/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>RedPajama</td>
    <td>2024</td>
    <td></td>
    <td><a href="https://github.com/togethercomputer/RedPajama-Data" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=4>Communication pre-training datasets</td>
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
    <td>CommData-PT dataset</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2502.18763" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  <tr align="center">
    <td rowspan=3>Communication fine-tuning datasets</td>
    <td>TelecomInstruct dataset</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.09424" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>CSI dataset compliant with 3GPP standards</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2406.14440" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
 <tr align="center">
    <td>CommData-FT dataset</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2502.18763" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  <tr align="center">
    <td rowspan=1>Communication alignment datasets</td>
    <td>TelecomAlign dataset</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.09424" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
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
   <td><a href="https://github.com/openai/gpt-3/tree/master" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>GPT-4</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2312.00752" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
  <tr align="center">
    <td>OpenAI o1</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
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
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10378323/" target="_blank" rel="noopener noreferrer">Paper</a></td>
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
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
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
    <td><a href="https://arxiv.org/abs/2305.05665" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://facebookresearch.github.io/ImageBind" target="_blank" rel="noopener noreferrer">Code</a></td>
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
     <td>JEPA</td>
     <td>JEPA</td>
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


  <tr align="center">
    <td rowspan=3>Large Reasoning Model</td>
     <td>Mixtral</td>
     <td>Mixtral</td>
     <td>2024</td>
     <td><a href="https://arxiv.org/abs/2401.04088" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="https://github.com/mistralai/mistral-inference" target="_blank" rel="noopener noreferrer">Code</a></td>
   </tr>
   <tr align="center">
    <td>OpenAI o3-mini</td>
    <td>OpenAI o3-mini</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2501.17749" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Trust4AI/ASTRAL" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
   <tr align="center">
     <td>DeepSeek</td>
     <td>DeepSeek-R1</td>
     <td>2025</td>
     <td><a href="https://arxiv.org/abs/2502.12893" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="https://github.com/dukeceicenter/jailbreak-o1o3-deepseek-r1" target="_blank" rel="noopener noreferrer">Code</a></td>
   </tr>
</tbody></table>

## Paper with code
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
Here is the list of our student contributors in each section.
| Section 	| Student Contributors 	|
|:-------	|:--------------------	|
|   The whole paper  | Zhengyu Du , Yuhan Zhang |
|   Literature Search   | Jian Zou , Dandan Qi  |
|   Project Maintenance   | Xitao Pan  |



## Acknowledgments
Thank you to all contributors and readers for your support. Special thanks to everyone who helped with development, testing, and provided feedback. Your contributions made this project possible.


## Update Log
| Version 	| Time 	| Update Content 	|
|:---	|:---	|:---	|
| v1 	| 2024/12/09 	| The initial version. 	|
| v2 	| 2024/12/18 	| Improve the writing.<br>Correct some minor errors. 	|
| v2 	| 2025/05/07 	| Improve the writing.<br>Correct some minor errors. 	|

## Citation   
```
@ARTICLE{
  author={JiaFeibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Merouane Debbah, Dusit Niyato, Zhu Han},
  journal={}, 
  title={A Comprehensive Survey of Large AI Models for Future Communications: Foundations, Applications and Challenges}, 
  year={2025},
  volume={},
  number={},
  pages={},
  doi={}}
```
