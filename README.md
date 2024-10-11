

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
    * [大模型在新兴应用的应用](#模型在新兴应用的应用0)
      * [数字孪生](#数字孪生)
      * [智慧医疗](#2智慧医疗)
      * [元宇宙](#3元宇宙)
      * [其他领域](#4其他领域)

## ComLAM的数据、训练和评估
### LAM的发展阶段
### 大模型的通信数据集
#### 1、通用数据集
（1）Common Crawl数据集[[链接]](http://commoncrawl.org/the-data/get-started/)  
（2）Pile数据集[[链接]](https://github.com/EleutherAI/the-pile)  
（3）Dolma数据[[链接]](https://huggingface.co/datasets/allenai/dolma)  
（4）RedPajama-Data数据集[[链接]](https://github.com/togethercomputer/RedPajama-Data)  
#### 2、通信专用数据集
（通过从通用数据集Common Crawl数据集和Redpajama数据集中提取与通信相关的内容来构成通信专用数据集）

#### 3、用于预训练的通信数据集
（1）TSpec-LLM数据集[[链接]](https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM)  
（2）OpenTelecom数据集  
（3）TeleQnA数据集  

#### 4、用于微调的通信数据集
（1）elecomInstruct数据集  
（2）符合3GPP标准的CSI数据集  

#### 5、用于对齐的通信数据集
（1）TelecomAlign数据集

### ComLAM的预训练方法
#### 1、通用数据集上的预训练
（1）无监督学习  
（2）自监督学习  
（3）多任务学习  
#### 2、专用数据集上的持续预训练
（1）通信领域的持续预训练
#### 3、预训练优化策略
（1）分布式训练  
（2）学习率调度  
（3）梯度剪裁  
### ComLAM的微调方法
ComLAM微调技术：电信指令微调  
大模型的微调技术：LoRA、Adapters、BitFit、Prefix Tuning等  
### ComLAM的对齐方法
（RLHF、RLAIF、PPO、DPO等大模型对齐微调技术）
### ComLAM的评估方法
1、通信问答与文档分类评测  
2、通信建模与代码生成能力评测  
3、通信推理能力评测  
4、通信工具学习能力评测  
5、通信安全评测  

## ComLAM的关键架构、分类和优化方法
### 大模型的关键架构
1、Transformer  
2、变分自编码器（VAE）  
3、扩散模型  
4、Mamba  
### 大模型分类及其在通信中的应用
#### 1、大语言模型LLM

<table><thead>
  <tr>
    <th>Category</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td rowspan=3>GPT系列</td>
    <td>GPT-3</td>
    <td>2020</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td>GPT-4</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td>OpenAI o1</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Gemma 系列</td>
    <td>Gemma 1</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td>Gemma 2</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>LLaMA 系列</td>
    <td>LLaMA-2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td>LLaMA-3</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
</tbody></table>




#### 2、大视觉模型LVM
<table><thead>
  <tr>
    <th>Category</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td rowspan=2>SAM 系列</td>
    <td>SAM-1</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td>SAM-2</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>DINO 系列</td>
    <td>DINO V1</td>
    <td>2021</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td>DINO V2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td rowspan=3>Stable Diffusion 系列</td>
    <td> Stable Diffusion V1</td>
    <td>2022</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
    <td>Stable Diffusion V2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
  <tr align="center">
   <td>Stable Diffusion V3</td>
   <td>2024</td>
   <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
  </tr>
</tbody></table>

#### 3、视觉语言模型VLM
（1）LLaVA  
（2）Qwen-VL 系列  
（3）Mini-GPT4  



#### 4、多模态大模型
（1）CoDi 系列  
（2）Meta-Transformer  
（3）ImageBind  
#### 5、世界模型
（1）Sora  
（2）JEPA  
（3）Vista  
#### 6、轻量级大模型
（1）TinyLlama  
（2）MobileVLM  
（3）Mini-Gemini  


### 大模型的优化方法
#### 1、In-context learning(ICL)
#### 2、XoT
#### 3、检索生成增强(RAG)
#### 4、多智能体系统(MAS)
#### 5、混合专家模型(MoE)



## 大模型在通信领域中的应用
### 大模型在 PHY 和 MAC 层中的设计
#### 1、大模型在 PHY 层的设计
#### 2、大模型在 MAC 层的设计


### 大模型在资源分配和优化的应用
#### 1、大模型的计算资源分配
#### 2、大模型的频谱资源分配
#### 3、大模型的能量资源优化


### 大模型在网络的设计与管理的应用
#### 1、网络的设计
#### 2、网络的管理


### 大模型在边缘智能的应用
#### 1、边缘端 AIGC 的学习与应用
#### 2、边缘端大模型资源管理与调度
#### 3、边缘端大模型跨域协同与融合
（1）与无线感知技术的融合  
（2）与强化学习技术的融合  
（3）与缓存和推理技术的融合  


### 大模型在语义通信的应用
#### 1、AIGC 增强的语义通信系统
（1）基于扩散模型的语义通信优化  
（2）基于 Transformer 的语义增强和推理  
（3）基于 LLM 的语义通信优化  
（4）基于大视觉模型的语义通信  
#### 2、智能体驱动的语义通信系统
#### 3、语义通信与无线感知


### 大模型在安全隐私的应用
#### 1、网络安全威胁检测与防御
（1）后门攻击防御  
（2）网络威胁检测  
（3）软件漏洞检测  
#### 2、通信网络中的可信 AI


### 大模型在新兴应用的应用
#### 1、数字孪生
#### 2、智慧医疗
#### 3、元宇宙
#### 4、其他领域













