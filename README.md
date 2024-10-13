
# ComLAM

简要介绍ComLAM



## 目录

* [ComLAM](#ComLAM)
  * [目录](#目录)
  * [ComLAM的数据、训练和评估](#ComLAM的数据训练和评估)
    * [LAM的发展阶段](#LAM的发展阶段)
    * [大模型的通信数据集](#大模型的通信数据集)
      * [通用数据集](#1通用数据集)
      * [通信专用数据集](#2通信专用数据集)
      * [用于预训练的通信数据集](#3用于预训练的通信数据集)
      * [用于微调的通信数据集](#4用于微调的通信数据集)
      * [用于对齐的通信数据集](#5用于对齐的通信数据集)
    * [ComLAM的预训练方法](#ComLAM的预训练方法)
      * [通用数据集上的预训练](#1通用数据集上的预训练)
      * [专用数据集上的持续预训练](#2专用数据集上的持续预训练)
      * [预训练优化策略](#3预训练优化策略)
    * [ComLAM的微调方法](#ComLAM的微调方法)
    * [ComLAM的对齐方法](#ComLAM的对齐方法)
    * [ComLAM的评估方法](#ComLAM的评估方法)
  * [ComLAM的关键架构、分类和优化方法](#ComLAM的关键架构分类和优化方法)
    * [大模型的关键架构](#1大模型的关键架构)
    * [大模型分类及其在通信中的应用](#2大模型分类及其在通信中的应用)
      * [大语言模型LLM](#大语言模型LLM)
      * [大视觉模型VLM](#大视觉模型VLM)
      * [视觉语言模型VLM](#视觉语言模型VLM)
      * [多模态大模型](#多模态大模型)
      * [世界模型](#世界模型)
      * [轻量级大模型](#轻量级大模型)
    * [大模型的优化方法](#3大模型的优化方法)
      * [In-context learning(ICL)](#In-contextlearning(ICL))
      * [XoT](#XoT)
      * [检索生成增强(RAG)](#检索生成增强(RAG))
      * [多智能体系统(MAS)](#多智能体系统(MAS))
      * [混合专家模型(MoE)](#混合专家模型(MoE))
  * [大模型在通信领域中的应用](#大模型在通信领域中的应用)
    * [大模型在PHY和MAC层中的设计](#大模型在PHY和MAC层中的设计)
      * [大模型在PHY层的设计](#1大模型在PHY层的设计)
      * [大模型在MAC层的设计](#2大模型在MAC层的设计)
    * [大模型在资源分配和优化的应用](#大模型在资源分配和优化的应用)
      * [大模型的计算资源分配](#1大模型的计算资源分配)
      * [大模型的频谱资源分配](#2大模型的频谱资源分配)
      * [大模型的能量资源优化](#3大模型的能量资源优化)
    * [大模型在网络的设计与管理的应用](#大模型在网络的设计与管理的应用)
      * [网络的设计](#1网络的设计)
      * [网络的管理](#2网络的管理)
    * [大模型在边缘智能的应用](#大模型在边缘智能的应用)
      * [边缘端 AIGC 的学习与应用](#1边缘端AIGC的学习与应用)
      * [边缘端大模型资源管理与调度](#2边缘端大模型资源管理与调度)
      * [边缘端大模型跨域协同与融合](#3边缘端大模型跨域协同与融合)
    * [大模型在语义通信的应用](#大模型在语义通信的应用)
      * [AIGC 增强的语义通信系统](#1AIGC增强的语义通信系统)
      * [智能体驱动的语义通信系统](#智能体驱动的语义通信系统)
      * [语义通信与无线感知](语义通信与无线感知)
    * [大模型在安全隐私的应用](#大模型在安全隐私的应用)
      * [网络安全威胁检测与防御](#1网络安全威胁检测与防御)
      * [通信网络中的可信 AI](#2通信网络中的可信AI)
    * [大模型在新兴应用的应用](#模型在新兴应用的应用)
      * [数字孪生](#1数字孪生)
      * [智慧医疗](#2智慧医疗)
      * [元宇宙](#3元宇宙)
      * [其他领域](#4其他领域)
  * [团队](#团队)
  * [致谢](#致谢)
  * [更新日志](#更新日志)

## ComLAM的数据、训练和评估
### LAM的发展阶段
### 大模型的通信数据集
#### 1、通用数据集
（1）Common Crawl数据集[[Download]](http://commoncrawl.org/the-data/get-started)  
（2）Pile数据集[[Download]]()  
（3）Dolma数据[[Download]]()  
（4）RedPajama-Data数据集[[Download]]()   
#### 2、通信专用数据集
（通过从通用数据集Common Crawl数据集和Redpajama数据集中提取与通信相关的内容来构成通信专用数据集）[[Download]]()  

#### 3、用于预训练的通信数据集
（1）TSpec-LLM数据集[[Download]]()  
（2）OpenTelecom数据集[[Download]]()  
（3）TeleQnA数据集[[Download]]()  

#### 4、用于微调的通信数据集
（1）elecomInstruct数据集[[Download]]()  
（2）符合3GPP标准的CSI数据集[[Download]]()  

#### 5、用于对齐的通信数据集
（1）TelecomAlign数据集[[Download]]()  

### ComLAM的预训练方法
#### 1、通用数据集上的预训练
（1）无监督学习[[Paper]]()    
（2）自监督学习[[Paper]]()    
（3）多任务学习[[Paper]]()    
#### 2、专用数据集上的持续预训练
（1）通信领域的持续预训练[[Paper]]()    
#### 3、预训练优化策略
（1）分布式训练[[Paper]]()    
（2）学习率调度[[Paper]]()    
（3）梯度剪裁[[Paper]]()    
### ComLAM的微调方法
1、ComLAM微调技术：  
 （1）电信指令微调[[Paper]]()  
2、大模型的微调技术：  
 （1）LoRA[[Paper]]()  
 （2）Adapters[[Paper]]()  
 （3）BitFit[[Paper]]()  
 （4）Prefix Tuning[[Paper]]()  
### ComLAM的对齐方法
1、大模型对齐微调技术：  
 （1）RLHF[[Paper]]()  
 （2）RLAIF[[Paper]]()  
 （3）PPO[[Paper]]()  
 （4）DPO[[Paper]]()  
### ComLAM的评估方法
1、通信问答与文档分类评测[[Paper]]()    
2、通信建模与代码生成能力评测[[Paper]]()    
3、通信推理能力评测[[Paper]]()    
4、通信工具学习能力评测[[Paper]]()    
5、通信安全评测[[Paper]]()    

## ComLAM的关键架构、分类和优化方法
### 大模型的关键架构
1、Transformer[[Paper]]()    
2、变分自编码器（VAE）[[Paper]]()    
3、扩散模型[[Paper]]()    
4、Mamba[[Paper]]()    
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
    <td><a href="https://splab.sdu.edu.cn/GPT3.pdf" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>GPT-4</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2312.00752" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2312.00752" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>OpenAI o1</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Gemma 系列</td>
    <td>Gemma 1</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2403.08295" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2403.08295" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>Gemma 2</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2408.00118" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2408.00118" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>LLaMA 系列</td>
    <td>LLaMA-2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2307.09288" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>LLaMA-3</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2407.21783" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
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
    <td><a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>SAM-2</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2408.08315" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2408.08315" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>DINO 系列</td>
    <td>DINO V1</td>
    <td>2021</td>
    <td><a href="https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>DINO V2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2304.07193" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2304.07193" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td rowspan=3>Stable Diffusion 系列</td>
    <td> Stable Diffusion V1</td>
    <td>2022</td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>Stable Diffusion V2</td>
    <td>2023</td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
   <td>Stable Diffusion V3</td>
   <td>2024</td>
   <td><a href="https://openreview.net/forum?id=FPnUhsQJ5B" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://openreview.net/pdf?id=FPnUhsQJ5B" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
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
    <td>2023</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Qwen-VL 系列</td>
    <td> Qwen-VL</td>
    <td>2023/6</td>
    <td><a href="https://arxiv.org/abs/2308.12966" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2308.12966 target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>Qwen-VL-Chat</td>
    <td>2023/9</td>
    <td><a href="https://arxiv.org/abs/2308.12966" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2308.12966" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>Mini-GPT4</td>
    <td>Mini-GPT4</td>
    <td>202_</td>
    <td><a href="https://arxiv.org/abs/2304.10592" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/pdf/2304.10592" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
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
    <td>2023</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/33edf072fe44f19079d66713a1831550-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/33edf072fe44f19079d66713a1831550-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>CoDi-2</td>
    <td>2024</td>
    <td><a href="http://openaccess.thecvf.com/content/CVPR2024/html/Tang_CoDi-2_In-Context_Interleaved_and_Interactive_Any-to-Any_Generation_CVPR_2024_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="http://openaccess.thecvf.com/content/CVPR2024/html/Tang_CoDi-2_In-Context_Interleaved_and_Interactive_Any-to-Any_Generation_CVPR_2024_paper.html" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
    <td>Meta-Transformer</td>
    <td>Meta-Transformer</td>
    <td>202_</td>
    <td><a href="https://arxiv.org/abs/2307.10802" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/abs/2307.10802" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
     <td>ImageBind</td>
     <td>ImageBind</td>
     <td>202_</td>
     <td><a href="http://openaccess.thecvf.com/content/CVPR2023/html/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="http://openaccess.thecvf.com/content/CVPR2023/html/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.html" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
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
    <td>202_</td>
    <td><a href="https://arxiv.org/abs/2402.17177" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/abs/2402.17177" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
     <td>JEPA </td>
     <td>JEPA </td>
     <td>202_</td>
     <td><a href="https://openreview.net/pdf?id=BZ5a1r-kVsf" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="https://openreview.net/pdf?id=BZ5a1r-kVsf" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
   </tr>
   <tr align="center">
      <td>Vista</td>
      <td>Vista</td>
      <td>202_</td>
      <td><a href="https://arxiv.org/abs/2405.17398" target="_blank" rel="noopener noreferrer">Paper</a></td>
      <td><a href="https://arxiv.org/abs/2405.17398" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
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
    <td>202_</td>
    <td><a href="https://arxiv.org/abs/2401.02385" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://arxiv.org/abs/2401.02385" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
  </tr>
  <tr align="center">
     <td>MobileVLM</td>
     <td>MobileVLM</td>
     <td>202_</td>
     <td><a href="https://arxiv.org/abs/2402.03766" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="https://arxiv.org/abs/2402.03766" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
   </tr>
   <tr align="center">
      <td>Mini-Gemini</td>
      <td>Mini-Gemini</td>
      <td>202_</td>
      <td><a href="https://arxiv.org/abs/2403.18814" target="_blank" rel="noopener noreferrer">Paper</a></td>
      <td><a href="https://arxiv.org/abs/2403.18814" target="_blank" rel="noopener noreferrer">Checkpoint</a></td>
    </tr>
</tbody></table>

### 大模型的优化方法
#### 1、In-context learning(ICL)
In-context learning(ICL)[[Paper]]()  
#### 2、XoT
XoT[[Paper]]()  
#### 3、检索生成增强(RAG)
检索生成增强(RAG)[[Paper]]()  
#### 4、多智能体系统(MAS)
多智能体系统(MAS)[[Paper]]()  
#### 5、混合专家模型(MoE)
混合专家模型(MoE)[[Paper]]()  



## 大模型在通信领域中的应用
### 大模型在 PHY 和 MAC 层中的设计
#### 1、大模型在 PHY 层的设计
[[Paper]]()  
#### 2、大模型在 MAC 层的设计
[[Paper]]()  


### 大模型在资源分配和优化的应用
#### 1、大模型的计算资源分配
[[Paper]]()  
#### 2、大模型的频谱资源分配
[[Paper]]()  
#### 3、大模型的能量资源优化
[[Paper]]()  


### 大模型在网络的设计与管理的应用
#### 1、网络的设计
[[Paper]]()  
#### 2、网络的管理
[[Paper]]()  


### 大模型在边缘智能的应用
#### 1、边缘端 AIGC 的学习与应用
[[Paper]]()  
#### 2、边缘端大模型资源管理与调度
[[Paper]]()  
#### 3、边缘端大模型跨域协同与融合
（1）与无线感知技术的融合[[Paper]]()  
（2）与强化学习技术的融合[[Paper]]()  
（3）与缓存和推理技术的融合[[Paper]]()  


### 大模型在语义通信的应用
#### 1、AIGC 增强的语义通信系统
（1）基于扩散模型的语义通信优化[[Paper]]()  
（2）基于 Transformer 的语义增强和推理[[Paper]]()  
（3）基于 LLM 的语义通信优化[[Paper]]()  
（4）基于大视觉模型的语义通信[[Paper]]()  
#### 2、智能体驱动的语义通信系统
[[Paper]]()  
#### 3、语义通信与无线感知
[[Paper]]()  


### 大模型在安全隐私的应用
#### 1、网络安全威胁检测与防御
（1）后门攻击防御[[Paper]]()  
（2）网络威胁检测[[Paper]]()  
（3）软件漏洞检测[[Paper]]()  
#### 2、通信网络中的可信 AI
[[Paper]]()  

### 大模型在新兴应用的应用
#### 1、数字孪生
[[Paper]]()  
#### 2、智慧医疗
[[Paper]]()  
#### 3、元宇宙
[[Paper]]()  
#### 4、其他领域
[[Paper]]()  

## 团队
| 部分 	| 学生贡献者 	|
|:----:	|:----------:	|
|      	|            	|
|      	|            	|
|      	|            	|


## 致谢



## 更新日志
| 版本 	| 时间 	| 更新内容 	|
|:----:	|:----:	|:--------:	|
|      	|      	|          	|
|      	|      	|          	|
|      	|      	|          	|












