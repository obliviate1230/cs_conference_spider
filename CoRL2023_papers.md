# Expansive Latent Planning for Sparse Reward Offline Reinforcement Learning
**题目:** 宽松奖励 offline强化学习的扩展计划

**作者:** Robert Gieselmann, Florian T. Pokorny

**Abstract:** Sampling-based motion planning algorithms excel at searching global solution paths in geometrically complex settings. However, classical approaches, such as RRT, are difficult to scale beyond low-dimensional search spaces and rely on privileged knowledge e.g. about collision detection and underlying state distances. In this work, we take a step towards the integration of sampling-based planning into the reinforcement learning framework to solve sparse-reward control tasks from high-dimensional inputs. Our method, called VELAP, determines sequences of waypoints through sampling-based exploration in a learned state embedding. Unlike other sampling-based techniques, we iteratively expand a tree-based memory of visited latent areas, which is leveraged to explore a larger portion of the latent space for a given number of search iterations. We demonstrate state-of-the-art results in learning control from offline data in the context of vision-based manipulation under sparse reward feedback. Our method extends the set of available planning tools in model-based reinforcement learning by adding a latent planner that searches globally for feasible paths instead of being bound to a fixed prediction horizon.

**摘要:** 基于样本的运动规划算法在几何复杂设置中能够搜索全球解决方案路径。然而,传统的方法,如RRT,很难超越低维搜索空间,依赖于特权知识,例如关于碰撞检测和潜在状态距离。在这项工作中,我们迈向了从高维输入中解决稀有奖励控制任务的增强学习框架内集成样本基于规划的一步。我们的方法叫做VELAP,通过基于样本的探索在学习状态嵌入中确定路径序列。我们的方法通过添加一个全球搜索可行的路径的潜伏规划师来扩展基于模型强化学习中的现有规划工具,而不是被绑定到一个固定的预测水平。

**[Paper URL](https://proceedings.mlr.press/v229/gieselmann23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gieselmann23a/gieselmann23a.pdf)** 

# Expansive Latent Planning for Sparse Reward Offline Reinforcement Learning
**题目:** 宽松奖励 offline强化学习的扩展计划

**作者:** Robert Gieselmann, Florian T. Pokorny

**Abstract:** Sampling-based motion planning algorithms excel at searching global solution paths in geometrically complex settings. However, classical approaches, such as RRT, are difficult to scale beyond low-dimensional search spaces and rely on privileged knowledge e.g. about collision detection and underlying state distances. In this work, we take a step towards the integration of sampling-based planning into the reinforcement learning framework to solve sparse-reward control tasks from high-dimensional inputs. Our method, called VELAP, determines sequences of waypoints through sampling-based exploration in a learned state embedding. Unlike other sampling-based techniques, we iteratively expand a tree-based memory of visited latent areas, which is leveraged to explore a larger portion of the latent space for a given number of search iterations. We demonstrate state-of-the-art results in learning control from offline data in the context of vision-based manipulation under sparse reward feedback. Our method extends the set of available planning tools in model-based reinforcement learning by adding a latent planner that searches globally for feasible paths instead of being bound to a fixed prediction horizon.

**摘要:** 基于样本的运动规划算法在几何复杂设置中能够搜索全球解决方案路径。然而,传统的方法,如RRT,很难超越低维搜索空间,依赖于特权知识,例如关于碰撞检测和潜在状态距离。在这项工作中,我们迈向了从高维输入中解决稀有奖励控制任务的增强学习框架内集成样本基于规划的一步。我们的方法叫做VELAP,通过基于样本的探索在学习状态嵌入中确定路径序列。我们的方法通过添加一个全球搜索可行的路径的潜伏规划师来扩展基于模型强化学习中的现有规划工具,而不是被绑定到一个固定的预测水平。

**[Paper URL](https://proceedings.mlr.press/v229/gieselmann23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gieselmann23a/gieselmann23a.pdf)** 

# SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning
**题目:** SayPlan:基于可尺度机器人任务规划的3D场景图的大型语言模型

**作者:** Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, Niko Suenderhauf

**Abstract:** Large language models (LLMs) have demonstrated impressive results in developing generalist planning agents for diverse tasks. However, grounding these plans in expansive, multi-floor, and multi-room environments presents a significant challenge for robotics. We introduce SayPlan, a scalable approach to LLM-based, large-scale task planning for robotics using 3D scene graph (3DSG) representations. To ensure the scalability of our approach, we: (1) exploit the hierarchical nature of 3DSGs to allow LLMs to conduct a "semantic search" for task-relevant subgraphs from a smaller, collapsed representation of the full graph; (2) reduce the planning horizon for the LLM by integrating a classical path planner and (3) introduce an "iterative replanning" pipeline that refines the initial plan using feedback from a scene graph simulator, correcting infeasible actions and avoiding planning failures. We evaluate our approach on two large-scale environments spanning up to 3 floors and 36 rooms with 140 assets and objects and show that our approach is capable of grounding large-scale, long-horizon task plans from abstract, and natural language instruction for a mobile manipulator robot to execute. We provide real robot video demonstrations on our project page https://sayplan.github.io.

**摘要:** 大型语言模型(LLM)在开发各种任务的一般性规划代理方面取得了令人印象深刻的成果。然而,在扩展性、多层及多室环境下建立这些计划对于机器人来说是一个重大的挑战。我们引入了SyPlan,一种基于LLM的可扩展的方法,用于机器人的大规模任务规划,使用3D场景图(3DSG)表示。为了确保我们的方法的可扩展性,我们:(一)利用3DSG的层次性,允许LLM进行从一个较小、倒塌的全图的任务相关子图的“语义搜索”;(二)通过集成经典路径规划者来减少LLM的规划地平线;(三)引入了一种“迭代重新规划”管道,利用场景图模拟器的反馈来精细最初的计划,纠正无法执行的行动,避免规划失败。我们评估了在最大3层和36室的140个资产和对象的两个大型环境上的方法,并证明了我们的方法能够从抽象和自然语言的教学中对移动操纵机器人执行大规模、长期的任务计划。

**[Paper URL](https://proceedings.mlr.press/v229/rana23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rana23a/rana23a.pdf)** 

# SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning
**题目:** SayPlan:基于可尺度机器人任务规划的3D场景图的大型语言模型

**作者:** Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, Niko Suenderhauf

**Abstract:** Large language models (LLMs) have demonstrated impressive results in developing generalist planning agents for diverse tasks. However, grounding these plans in expansive, multi-floor, and multi-room environments presents a significant challenge for robotics. We introduce SayPlan, a scalable approach to LLM-based, large-scale task planning for robotics using 3D scene graph (3DSG) representations. To ensure the scalability of our approach, we: (1) exploit the hierarchical nature of 3DSGs to allow LLMs to conduct a "semantic search" for task-relevant subgraphs from a smaller, collapsed representation of the full graph; (2) reduce the planning horizon for the LLM by integrating a classical path planner and (3) introduce an "iterative replanning" pipeline that refines the initial plan using feedback from a scene graph simulator, correcting infeasible actions and avoiding planning failures. We evaluate our approach on two large-scale environments spanning up to 3 floors and 36 rooms with 140 assets and objects and show that our approach is capable of grounding large-scale, long-horizon task plans from abstract, and natural language instruction for a mobile manipulator robot to execute. We provide real robot video demonstrations on our project page https://sayplan.github.io.

**摘要:** 大型语言模型(LLM)在开发各种任务的一般性规划代理方面取得了令人印象深刻的成果。然而,在扩展性、多层及多室环境下建立这些计划对于机器人来说是一个重大的挑战。我们引入了SyPlan,一种基于LLM的可扩展的方法,用于机器人的大规模任务规划,使用3D场景图(3DSG)表示。为了确保我们的方法的可扩展性,我们:(一)利用3DSG的层次性,允许LLM进行从一个较小、倒塌的全图的任务相关子图的“语义搜索”;(二)通过集成经典路径规划者来减少LLM的规划地平线;(三)引入了一种“迭代重新规划”管道,利用场景图模拟器的反馈来精细最初的计划,纠正无法执行的行动,避免规划失败。我们评估了在最大3层和36室的140个资产和对象的两个大型环境上的方法,并证明了我们的方法能够从抽象和自然语言的教学中对移动操纵机器人执行大规模、长期的任务计划。

**[Paper URL](https://proceedings.mlr.press/v229/rana23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rana23a/rana23a.pdf)** 

# Robot Parkour Learning
**题目:** 机器人皮毛学习

**作者:** Ziwen Zhuang, Zipeng Fu, Jianren Wang, Christopher G. Atkeson, Sören Schwertfeger, Chelsea Finn, Hang Zhao

**Abstract:** Parkour is a grand challenge for legged locomotion that requires robots to overcome various obstacles rapidly in complex environments. Existing methods can generate either diverse but blind locomotion skills or vision-based but specialized skills by using reference animal data or complex rewards. However, autonomous parkour requires robots to learn generalizable skills that are both vision-based and diverse to perceive and react to various scenarios. In this work, we propose a system for learning a single end-to-end vision-based parkour policy of diverse parkour skills using a simple reward without any reference motion data. We develop a reinforcement learning method inspired by direct collocation to generate parkour skills, including climbing over high obstacles, leaping over large gaps, crawling beneath low barriers, squeezing through thin slits, and running. We distill these skills into a single vision-based parkour policy and transfer it to a quadrupedal robot using its egocentric depth camera. We demonstrate that our system can empower low-cost quadrupedal robots to autonomously select and execute appropriate parkour skills to traverse challenging environments in the real world. Project website: https://robot-parkour.github.io/

**摘要:** 在复杂环境下,机器人快速克服各种障碍是腿部运动的一大挑战。现有的方法可通过参考动物数据或复杂奖励来产生多种但盲运动技能或基于视觉而专门技能。然而,自主机器人需要学习基于视觉和多样的通用技能,以感知和对各种场景作出反应。本研究中,我们提出了一种基于视觉的单个终点到终点的基于各种障碍的机器人策略,以使用简单的奖励,而不使用任何参考运动数据。我们开发了基于直接定位的强化学习方法,以产生各种障碍攀爬、跳过大空隙、爬下低障碍、挤穿细缝和跑。我们将这些技能转化为一种基于视觉的parkour策略,并将其应用于自我中心的深度摄像机转移到四肢机器人中。我们证明,我们的系统能够使低成本四肢机器人自主选择和执行适当的parkour技能,从而在现实世界中跨越挑战的环境。

**[Paper URL](https://proceedings.mlr.press/v229/zhuang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhuang23a/zhuang23a.pdf)** 

# Task-Oriented Koopman-Based Control with Contrastive Encoder
**题目:** 基于任务的 Koopman 控件与反向编码器

**作者:** Xubo Lyu, Hanyang Hu, Seth Siriya, Ye Pu, Mo Chen

**Abstract:** We present task-oriented Koopman-based control that utilizes end-to-end reinforcement learning and contrastive encoder to simultaneously learn the Koopman latent embedding, operator, and associated linear controller within an iterative loop. By prioritizing the task cost as the main objective for controller learning, we reduce the reliance of controller design on a well-identified model, which, for the first time to the best of our knowledge, extends Koopman control from low to high-dimensional, complex nonlinear systems, including pixel-based tasks and a real robot with lidar observations. Code and videos are available: https://sites.google.com/view/kpmlilatsupp/.

**摘要:** 我们提出了面向任务的 Koopman-based控制,它利用end-to-end增强学习和对比编码器同时学习 Koopman隐形嵌入、操作器和关联的线性控制器在迭代循环中。通过优先考虑任务成本作为控制器学习的主要目标,我们减少了控制器设计的依赖于明确的模型,这首次将我们的知识最大化,从低到高维的复杂非线性系统,包括像素-based任务和一个有线性观察的真正的机器人。

**[Paper URL](https://proceedings.mlr.press/v229/lyu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lyu23a/lyu23a.pdf)** 

# On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills
**题目:**  Koopman操作理论在熟练操纵技能学习中的应用

**作者:** Yunhai Han, Mandy Xie, Ye Zhao, Harish Ravichandar

**Abstract:** Despite impressive dexterous manipulation capabilities enabled by learning-based approaches, we are yet to witness widespread adoption beyond well-resourced laboratories. This is likely due to practical limitations, such as significant computational burden, inscrutable learned behaviors, sensitivity to initialization, and the considerable technical expertise required for implementation. In this work, we investigate the utility of Koopman operator theory in alleviating these limitations. Koopman operators are simple yet powerful control-theoretic structures to represent complex nonlinear dynamics as linear systems in higher dimensions. Motivated by the fact that complex nonlinear dynamics underlie dexterous manipulation, we develop a Koopman operator-based imitation learning framework to learn the desired motions of both the robotic hand and the object simultaneously. We show that Koopman operators are surprisingly effective for dexterous manipulation and offer a number of unique benefits. Notably, policies can be learned analytically, drastically reducing computation burden and eliminating sensitivity to initialization and the need for painstaking hyperparameter optimization. Our experiments reveal that a Koopman operator-based approach can perform comparably to state-of-the-art imitation learning algorithms in terms of success rate and sample efficiency, while being an order of magnitude faster. Policy videos can be viewed at https://sites.google.com/view/kodex-corl.

**摘要:** 尽管基于学习的操作方法能够实现令人印象深刻的敏捷操作能力,但我们仍未看到在资源充足的实验室以外的广泛应用。这可能是由于实际的局限性,例如重大的计算负担,不可思议的学习行为,对初始化敏感性,以及实现所需的大量技术专门知识。在此工作中,我们研究了 Koopman操作理论在减轻这些局限性方面的作用。 Koopman操作是简单但强大的控制理论结构,以代表高维线性系统中复杂的非线性动态。我们证明 Koopman操作员对于敏捷操作具有惊人的有效性,并提供了一系列独特的好处。 尤其是,政策可以分析学习,大幅减少计算负担,消除对初始化敏感性,以及对痛苦的超参数优化的需要。

**[Paper URL](https://proceedings.mlr.press/v229/han23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/han23a/han23a.pdf)** 

# Rearrangement Planning for General Part Assembly
**题目:** 大会全体会议重新安排规划

**作者:** Yulong Li, Andy Zeng, Shuran Song

**Abstract:** Most successes in autonomous robotic assembly have been restricted to single target or category. We propose to investigate general part assembly, the task of creating novel target assemblies with unseen part shapes. As a fundamental step to a general part assembly system, we tackle the task of determining the precise poses of the parts in the target assembly, which we term “rearrangement planning". We present General Part Assembly Transformer (GPAT), a transformer-based model architecture that accurately predicts part poses by inferring how each part shape corresponds to the target shape. Our experiments on both 3D CAD models and real-world scans demonstrate GPAT’s generalization abilities to novel and diverse target and part shapes.

**摘要:** 自动机器人组装的大部分成功都局限于单一目标或类别。我们建议研究通用部件组装,即创建具有未知部件形状的新目标组装的任务。作为通用部件组装系统的一个基本步骤,我们解决了确定目标组装中的部件的准确姿态的任务,我们称之为“重新安排规划”。我们介绍通用部件组装变换器(GPAT),一种基于变换器的模型架构,通过推导每个部件形状如何与目标形状相符来准确预测部件姿态。

**[Paper URL](https://proceedings.mlr.press/v229/li23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/li23a/li23a.pdf)** 

# Language-Guided Traffic Simulation via Scene-Level Diffusion
**题目:** 基于语言的场景级扩散交通仿真

**作者:** Ziyuan Zhong, Davis Rempe, Yuxiao Chen, Boris Ivanovic, Yulong Cao, Danfei Xu, Marco Pavone, Baishakhi Ray

**Abstract:** Realistic and controllable traffic simulation is a core capability that is necessary to accelerate autonomous vehicle (AV) development. However, current approaches for controlling learning-based traffic models require significant domain expertise and are difficult for practitioners to use. To remedy this, we present CTG++, a scene-level conditional diffusion model that can be guided by language instructions. Developing this requires tackling two challenges: the need for a realistic and controllable traffic model backbone, and an effective method to interface with a traffic model using language. To address these challenges, we first propose a scene-level diffusion model equipped with a spatio-temporal transformer backbone, which generates realistic and controllable traffic. We then harness a large language model (LLM) to convert a user’s query into a loss function, guiding the diffusion model towards query-compliant generation. Through comprehensive evaluation, we demonstrate the effectiveness of our proposed method in generating realistic, query-compliant traffic simulations.

**摘要:** 现实和可控制的交通仿真是加速自主车辆(AV)发展所需的核心能力。然而,当前控制基于学习的交通模型的方法需要大量领域专业知识,对于实践者来说很难使用。为此,我们提出了CTG++,一种可由语言指令指导的场景条件扩散模型。通过综合评价,我们证明了我们提出的方法在生成现实、符合查询的交通仿真方面的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/zhong23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhong23a/zhong23a.pdf)** 

# Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping
**题目:** 语言嵌入射线场用于零射程任务 Oriented Grasping

**作者:** Adam Rashid, Satvik Sharma, Chung Min Kim, Justin Kerr, Lawrence Yunliang Chen, Angjoo Kanazawa, Ken Goldberg

**Abstract:** Grasping objects by a specific subpart is often crucial for safety and for executing downstream tasks. We propose LERF-TOGO, Language Embedded Radiance Fields for Task-Oriented Grasping of Objects, which uses vision-language models zero-shot to output a grasp distribution over an object given a natural language query. To accomplish this, we first construct a LERF of the scene, which distills CLIP embeddings into a multi-scale 3D language field queryable with text. However, LERF has no sense of object boundaries, so its relevancy outputs often return incomplete activations over an object which are insufficient for grasping. LERF-TOGO mitigates this lack of spatial grouping by extracting a 3D object mask via DINO features and then conditionally querying LERF on this mask to obtain a semantic distribution over the object to rank grasps from an off-the-shelf grasp planner. We evaluate LERF-TOGO’s ability to grasp task-oriented object parts on 31 physical objects, and find it selects grasps on the correct part in $81%$ of trials and grasps successfully in $69%$. Code, data, appendix, and details are available at: lerftogo.github.io

**摘要:** 通过一个特定的子部分对对象进行抓取通常对安全和执行下游任务至关重要。我们建议LERF-TOGO,语言嵌入射线场用于任务导向对象的抓取,它使用视觉语言模型零射出给自然语言查询的对象上的抓取分布。为此,我们首先构建场景的LERF,它将CLIP嵌入在可用文本查询的多尺度3D语言字段中。然而,LERF没有对象边界感,因此其相关性输出往往返回在对象上不完整的抓取活动。我们评估了LERF-TOGO在31个物理对象上的任务目标部分的把握能力,并发现它在81%的测试中选择正确部分的把握,并在69%的测试中成功地把握。代码、数据、附录和详细资料可于 lerftogo.github.io

**[Paper URL](https://proceedings.mlr.press/v229/rashid23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rashid23a/rashid23a.pdf)** 

# MimicPlay: Long-Horizon Imitation Learning by Watching Human Play
**题目:** 模拟游戏:通过观察人类游戏进行长期模拟学习

**作者:** Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, Anima Anandkumar

**Abstract:** Imitation learning from human demonstrations is a promising paradigm for teaching robots manipulation skills in the real world. However, learning complex long-horizon tasks often requires an unattainable amount of demonstrations. To reduce the high data requirement, we resort to human play data - video sequences of people freely interacting with the environment using their hands. Even with different morphologies, we hypothesize that human play data contain rich and salient information about physical interactions that can readily facilitate robot policy learning. Motivated by this, we introduce a hierarchical learning framework named MimicPlay that learns latent plans from human play data to guide low-level visuomotor control trained on a small number of teleoperated demonstrations. With systematic evaluations of 14 long-horizon manipulation tasks in the real world, we show that MimicPlay outperforms state-of-the-art imitation learning methods in task success rate, generalization ability, and robustness to disturbances. Code and videos are available at https://mimic-play.github.io.

**摘要:** 仿真学习是机器人在现实世界中操纵技能教学的一个有前途的范式。然而,学习复杂长视线任务往往需要不可实现的大量仿真。为了减少高数据需求,我们诉诸人类游戏数据--人们自由使用手来与环境相互作用的视频序列。即使在不同的形态学中,我们假设人类游戏数据包含丰富的和突出的信息,可以轻易地促进机器人政策学习。通过在现实世界中对14个长期操纵任务的系统评估,我们证明MimicPlay在任务成功率、推广能力和干扰的鲁棒性方面比最先进的仿真学习方法高。

**[Paper URL](https://proceedings.mlr.press/v229/wang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23a/wang23a.pdf)** 

# Continual Vision-based Reinforcement Learning with Group Symmetries
**题目:** 基于视觉的持续强化学习与群交配

**作者:** Shiqi Liu, Mengdi Xu, Peide Huang, Xilun Zhang, Yongkang Liu, Kentaro Oguchi, Ding Zhao

**Abstract:** Continual reinforcement learning aims to sequentially learn a variety of tasks, retaining the ability to perform previously encountered tasks while simultaneously developing new policies for novel tasks. However, current continual RL approaches overlook the fact that certain tasks are identical under basic group operations like rotations or translations, especially with visual inputs. They may unnecessarily learn and maintain a new policy for each similar task, leading to poor sample efficiency and weak generalization capability. To address this, we introduce a unique Continual Vision-based Reinforcement Learning method that recognizes Group Symmetries, called COVERS, cultivating a policy for each group of equivalent tasks rather than an individual task. COVERS employs a proximal-policy-gradient-based (PPO-based) algorithm to train each policy, which contains an equivariant feature extractor and takes inputs with different modalities, including image observations and robot proprioceptive states. It also utilizes an unsupervised task grouping mechanism that relies on 1-Wasserstein distance on the extracted invariant features. We evaluate COVERS on a sequence of table-top manipulation tasks in simulation and on a real robot platform. Our results show that COVERS accurately assigns tasks to their respective groups and significantly outperforms baselines by generalizing to unseen but equivariant tasks in seen task groups. Demos are available on our project page: https://sites.google.com/view/rl-covers/.

**摘要:** 持续增强学习的目标是连续学习多种任务,同时保持执行以前遇到的任务的能力,同时开发新任务的新策略。然而,当前持续增强学习方法忽略某些任务在基本群操作下,如旋转或翻译,特别是在视觉输入下,是相同的事实。它们可能不必要地学习和维持每个相似任务的新策略,导致样本效率低和一般化能力弱。COVERS使用近似策略梯度(PPO)算法来训练每个策略,它包含一个等效特征提取器,并采用不同的模式输入,包括图像观察和机器人受感状态。它还利用一个不受监督的任务聚类机制,它依赖于抽取的不等效特征的1-Wasserstein距离。我们评估COVERS在模拟和真实机器人平台的表顶操纵任务的序列。我们的结果表明, COVERS准确地分配任务给各自的组别,并通过推广到未见但在已见任务组中等效任务,大大超过了基线。

**[Paper URL](https://proceedings.mlr.press/v229/liu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23a/liu23a.pdf)** 

# HACMan: Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation
**题目:** HACMan:学习6D非隐形操作的混合行动者-临界地图

**作者:** Wenxuan Zhou, Bowen Jiang, Fan Yang, Chris Paxton, David Held

**Abstract:** Manipulating objects without grasping them is an essential component of human dexterity, referred to as non-prehensile manipulation. Non-prehensile manipulation may enable more complex interactions with the objects, but also presents challenges in reasoning about gripper-object interactions. In this work, we introduce Hybrid Actor-Critic Maps for Manipulation (HACMan), a reinforcement learning approach for 6D non-prehensile manipulation of objects using point cloud observations. HACMan proposes a temporally-abstracted and spatially-grounded object-centric action representation that consists of selecting a contact location from the object point cloud and a set of motion parameters describing how the robot will move after making contact. We modify an existing off-policy RL algorithm to learn in this hybrid discrete-continuous action representation. We evaluate HACMan on a 6D object pose alignment task in both simulation and in the real world. On the hardest version of our task, with randomized initial poses, randomized 6D goals, and diverse object categories, our policy demonstrates strong generalization to unseen object categories without a performance drop, achieving an $89%$ success rate on unseen objects in simulation and $50%$ success rate with zero-shot transfer in the real world. Compared to alternative action representations, HACMan achieves a success rate more than three times higher than the best baseline. With zero-shot sim2real transfer, our policy can successfully manipulate unseen objects in the real world for challenging non-planar goals, using dynamic and contact-rich non-prehensile skills. Videos can be found on the project website: https://hacman-2023.github.io.

**摘要:** 不把握对象的操作是人类机能的基本组成部分,称为“不理解对象的操作”。不理解对象的操作可能使与对象进行更复杂的交互,但也提出了关于握手对象交互的推理的挑战。本研究中,我们介绍了基于点云观测的混合行动行为映射(HACMan),一种基于点云观测的6D不理解对象的增强学习方法。HACMan提出了一种由点云中选择接触位置和描述机器人在接触后如何移动的运动参数组成的时空抽象和空间地基的对象中心动作表示。在最困难的任务版本中,通过随机化初始姿态、随机化6D目标和多种对象类别,我们的政策显示了无视对象类别的强一般化,无性能下降,在模拟中无视对象的成功率达到89%,在现实世界中的零射击转移成功率达到50%。与其他行动表现相比,HACMan的成功率比最佳基线高三倍。

**[Paper URL](https://proceedings.mlr.press/v229/zhou23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhou23a/zhou23a.pdf)** 

# Hijacking Robot Teams Through Adversarial Communication
**题目:** 通过敌对沟通劫持机器人团队

**作者:** Zixuan Wu, Sean Charles Ye, Byeolyi Han, Matthew Gombolay

**Abstract:** Communication is often necessary for robot teams to collaborate and complete a decentralized task. Multi-agent reinforcement learning (MARL) systems allow agents to learn how to collaborate and communicate to complete a task. These domains are ubiquitous and include safety-critical domains such as wildfire fighting, traffic control, or search and rescue missions. However, critical vulnerabilities may arise in communication systems as jamming the signals can interrupt the robot team. This work presents a framework for applying black-box adversarial attacks to learned MARL policies by manipulating only the communication signals between agents. Our system only requires observations of MARL policies after training is complete, as this is more realistic than attacking the training process. To this end, we imitate a learned policy of the targeted agents without direct interaction with the environment or ground truth rewards. Instead, we infer the rewards by only observing the behavior of the targeted agents. Our framework reduces reward by $201%$ compared to an equivalent baseline method and also shows favorable results when deployed in real swarm robots. Our novel attack methodology within MARL systems contributes to the field by enhancing our understanding on the reliability of multi-agent systems.

**摘要:** 多代理强化学习(MARL)系统允许代理人学习如何进行协作和通信完成任务。这些领域是无处不在的,包括野火战斗、交通控制、搜索和救援任务等安全关键领域。然而,通信系统可能出现关键漏洞,因为干扰信号可以干扰机器人团队。该工作提出了应用黑箱敌对攻击的框架,通过操纵代理人之间的通信信号来学习MARL政策。我们的系统仅需要在训练完成后观察MARL政策,因为这比攻击训练过程更现实。为此目的,我们模仿目标代理人学习的政策,而不直接与环境或地面真理奖励互动。相反,我们只通过观察目标代理人的行为来推导报酬。我们的框架降低了与等效的基线方法相比的报酬201%$,并且在实际的蜂窝机器人部署时也显示出有利的结果。

**[Paper URL](https://proceedings.mlr.press/v229/wu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wu23a/wu23a.pdf)** 

# GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields
**题目:** GNFactor:通用神经特征域的多任务实物机器人学习

**作者:** Yanjie Ze, Ge Yan, Yueh-Hua Wu, Annabella Macaluso, Yuying Ge, Jianglong Ye, Nicklas Hansen, Li Erran Li, Xiaolong Wang

**Abstract:** It is a long-standing problem in robotics to develop agents capable of executing diverse manipulation tasks from visual observations in unstructured real-world environments. To achieve this goal, the robot will need to have a comprehensive understanding of the 3D structure and semantics of the scene. In this work, we present GNFactor, a visual behavior cloning agent for multi-task robotic manipulation with Generalizable Neural feature Fields. GNFactor jointly optimizes a neural radiance field (NeRF) as a reconstruction module and a Perceiver Transformer as a decision-making module, leveraging a shared deep 3D voxel representation. To incorporate semantics in 3D, the reconstruction module incorporates a vision-language foundation model (e.g., Stable Diffusion) to distill rich semantic information into the deep 3D voxel. We evaluate GNFactor on 3 real-robot tasks and perform detailed ablations on 10 RLBench tasks with a limited number of demonstrations. We observe a substantial improvement of GNFactor over current state-of-the-art methods in seen and unseen tasks, demonstrating the strong generalization ability of GNFactor. Project website: https://yanjieze.com/GNFactor/

**摘要:** 为了实现这一目标,机器人需要对场景的3D结构和语义有全面了解。本课题中,我们介绍GNFactor,一种具有通用神经特征场的多任务机器人操作的视觉行为克隆剂。GNFactor联合优化了神经辐射场(NeRF)作为重建模块和 perceiver Transformer作为决策模块,利用共享的深3D voxel表示。为了将语义纳入3D,重建模块 incorporates a vision-language foundation model (e.g., Stable Diffusion) to distill rich semantic information into the deep 3D voxel。我们观察到GNFactor在可视和不可视任务上比当前最先进的方法有很大改善,这表明GNFactor具有很强的一般化能力。

**[Paper URL](https://proceedings.mlr.press/v229/ze23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ze23a/ze23a.pdf)** 

# Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance
**题目:** 引导自己的技能:用大型语言模型指导学习解决新任务

**作者:** Jesse Zhang, Jiahui Zhang, Karl Pertsch, Ziyi Liu, Xiang Ren, Minsuk Chang, Shao-Hua Sun, Joseph J. Lim

**Abstract:** We propose BOSS, an approach that automatically learns to solve new long-horizon, complex, and meaningful tasks by growing a learned skill library with minimal supervision. Prior work in reinforcement learning require expert supervision, in the form of demonstrations or rich reward functions, to learn long-horizon tasks. Instead, our approach BOSS (BOotStrapping your own Skills) learns to accomplish new tasks by performing "skill bootstrapping," where an agent with a set of primitive skills interacts with the environment to practice new skills without receiving reward feedback for tasks outside of the initial skill set. This bootstrapping phase is guided by large language models (LLMs) that inform the agent of meaningful skills to chain together. Through this process, BOSS builds a wide range of complex and useful behaviors from a basic set of primitive skills. We demonstrate through experiments in realistic household environments that agents trained with our LLM-guided bootstrapping procedure outperform those trained with naive bootstrapping as well as prior unsupervised skill acquisition methods on zero-shot execution of unseen, long-horizon tasks in new environments. Website at clvrai.com/boss.

**摘要:** 我们建议BOSS,一种自动学习解决新的长期、复杂和有意义的任务的方法,通过建立一个学习技能库,并进行最小限度的监督。在增强学习中以前的工作需要专家监督,以示例或丰富的奖励功能来学习长期任务。相反,我们的方法BOSS(BOotStrapping your own Skills)通过执行"技能引导"来学习完成新的任务,其中一个具有原始技能的代理人与环境互动,以实践新的技能,而不接受从初始技能集以外的任务的奖励反馈。该引导阶段由大型语言模型(LLMs)指导,通知具有意义的技能的代理人进行联链。通过在现实的家庭环境中进行实验,我们证明,用LLM引导的启动过程训练的代理人比那些用直观的启动过程训练的代理人胜过,以及在新的环境中在无视、远景任务的零射击执行上事先没有监督的技能获取方法。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23a/zhang23a.pdf)** 

# DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control
**题目:** DATT:quadrotor控制的深度自适应跟踪

**作者:** Kevin Huang, Rwik Rana, Alexander Spitzer, Guanya Shi, Byron Boots

**Abstract:** Precise arbitrary trajectory tracking for quadrotors is challenging due to unknown nonlinear dynamics, trajectory infeasibility, and actuation limits. To tackle these challenges, we present DATT, a learning-based approach that can precisely track arbitrary, potentially infeasible trajectories in the presence of large disturbances in the real world. DATT builds on a novel feedforward-feedback-adaptive control structure trained in simulation using reinforcement learning. When deployed on real hardware, DATT is augmented with a disturbance estimator using $\mathcal{L}_1$ adaptive control in closed-loop, without any fine-tuning. DATT significantly outperforms competitive adaptive nonlinear and model predictive controllers for both feasible smooth and infeasible trajectories in unsteady wind fields, including challenging scenarios where baselines completely fail. Moreover, DATT can efficiently run online with an inference time less than 3.2ms, less than 1/4 of the adaptive nonlinear model predictive control baseline.

**摘要:** 由于未知的非线性动力学、轨迹不可行性和操作限制,对四轮机的精密任意轨迹跟踪具有挑战性。为了解决这些挑战,我们提出了一种基于学习的方法,它能够准确地跟踪在真实世界出现重大扰动时的任意、潜在不可能的轨迹。DATT建立在采用增强学习的仿真训练的新型反馈后反馈适应控制结构上。当部署在实际硬件上,DATT在闭环中使用$\mathcal{L}_1$的适应控制估计器增强,而无需微调。DATT大大超过了竞争性的适应非线性和模型预测控制器,用于在不稳定风场的可行的平滑和不可能的轨迹,包括在基线完全失败的挑战性场景。此外,DATT可以有效运行在线,推导时间少于3.2ms,低于适应性非线性模型预测控制基线的1/4。

**[Paper URL](https://proceedings.mlr.press/v229/huang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/huang23a/huang23a.pdf)** 

# HANDLOOM: Learned Tracing of One-Dimensional Objects for Inspection and Manipulation
**题目:** 操作室:对一维物体进行检测和操作的学问跟踪

**作者:** Vainavi Viswanath, Kaushik Shivakumar, Mallika Parulekar, Jainil Ajmera, Justin Kerr, Jeffrey Ichnowski, Richard Cheng, Thomas Kollar, Ken Goldberg

**Abstract:** Tracing – estimating the spatial state of – long deformable linear objects such as cables, threads, hoses, or ropes, is useful for a broad range of tasks in homes, retail, factories, construction, transportation, and healthcare. For long deformable linear objects (DLOs or simply cables) with many (over 25) crossings, we present HANDLOOM (Heterogeneous Autoregressive Learned Deformable Linear Object Observation and Manipulation) a learning-based algorithm that fits a trace to a greyscale image of cables. We evaluate HANDLOOM on semi-planar DLO configurations where each crossing involves at most 2 segments. HANDLOOM makes use of neural networks trained with 30,000 simulated examples and 568 real examples to autoregressively estimate traces of cables and classify crossings. Experiments find that in settings with multiple identical cables, HANDLOOM can trace each cable with $80%$ accuracy. In single-cable images, HANDLOOM can trace and identify knots with $77%$ accuracy. When HANDLOOM is incorporated into a bimanual robot system, it enables state-based imitation of knot tying with $80%$ accuracy, and it successfully untangles $64%$ of cable configurations across 3 levels of difficulty. Additionally, HANDLOOM demonstrates generalization to knot types and materials (rubber, cloth rope) not present in the training dataset with $85%$ accuracy. Supplementary material, including all code and an annotated dataset of RGB-D images of cables along with ground-truth traces, is at https://sites.google.com/view/cable-tracing.

**摘要:** 跟踪 — — 估算长度变形线性物体的空间状态,如电缆、线条、软管或绳索,对于家庭、零售、工厂、建筑、运输和医疗等广泛的任务有用。 对于长度变形线性物体(DLOs或简单的电缆)有多个(超过25个)交叉点,我们提出了一种基于学习的算法HANDLOOM(Heterogeneous Autoregressive Learned Deformable Linear Object Observation and Manipulation),该算法将一个跟踪映射到灰色的电缆图像中。当HANDLOOM被纳入一个双人机器人系统时,它允许基于状态的仿真与$80%精度的结扎,并成功地解锁在3个难度级别的$64%的电缆配置。此外,HANDLOOM展示了与$85%精度的训练数据集中没有的结扎类型和材料(橡胶、布绳)的一般化。补充材料包括所有代码和带有地面真实迹象的 RGB-D电缆图像的注释数据集,在 https://sites.google.com/view/cable-tracing。

**[Paper URL](https://proceedings.mlr.press/v229/viswanath23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/viswanath23a/viswanath23a.pdf)** 

# Predicting Object Interactions with Behavior Primitives: An Application in Stowing Tasks
**题目:** 预测对象与行为 Primitives的相互作用:在堆栈任务中的应用

**作者:** Haonan Chen, Yilong Niu, Kaiwen Hong, Shuijing Liu, Yixuan Wang, Yunzhu Li, Katherine Rose Driggs-Campbell

**Abstract:** Stowing, the task of placing objects in cluttered shelves or bins, is a common task in warehouse and manufacturing operations. However, this task is still predominantly carried out by human workers as stowing is challenging to automate due to the complex multi-object interactions and long-horizon nature of the task. Previous works typically involve extensive data collection and costly human labeling of semantic priors across diverse object categories. This paper presents a method to learn a generalizable robot stowing policy from predictive model of object interactions and a single demonstration with behavior primitives. We propose a novel framework that utilizes Graph Neural Networks (GNNs) to predict object interactions within the parameter space of behavioral primitives. We further employ primitive-augmented trajectory optimization to search the parameters of a predefined library of heterogeneous behavioral primitives to instantiate the control action. Our framework enables robots to proficiently execute long-horizon stowing tasks with a few keyframes (3-4) from a single demonstration. Despite being solely trained in a simulation, our framework demonstrates remarkable generalization capabilities. It efficiently adapts to a broad spectrum of real-world conditions, including various shelf widths, fluctuating quantities of objects, and objects with diverse attributes such as sizes and shapes.

**摘要:** 堆积物置于杂货架或垃圾箱的任务是仓库和制造作业中常见的任务。然而,由于堆积物的复杂多目标相互作用和任务的长期地平线性质,堆积物的自动化仍然是人类劳动者的主要任务。以往的工作通常涉及广泛的数据收集和各种对象类别的语义优先级的昂贵的人类标记。本文提出了一种从对象相互作用预测模型和行为原始物的单个演示中学习可推广的机器人堆积物政策的方法。我们的框架使机器人能够从单一的演示中精确地执行几个关键框架(3-4)的长地平线抛物任务。尽管仅在模拟中进行训练,我们的框架展示了显著的一般化能力。它有效地适应了广泛的现实环境,包括各种架幅、物体的波动数量以及物体的大小和形状等多种属性。

**[Paper URL](https://proceedings.mlr.press/v229/chen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23a/chen23a.pdf)** 

# Language to Rewards for Robotic Skill Synthesis
**题目:** 语言对机器人技能合成的奖励

**作者:** Wenhao Yu, Nimrod Gileadi, Chuyuan Fu, Sean Kirmani, Kuang-Huei Lee, Montserrat Gonzalez Arenas, Hao-Tien Lewis Chiang, Tom Erez, Leonard Hasenclever, Jan Humplik, Brian Ichter, Ted Xiao, Peng Xu, Andy Zeng, Tingnan Zhang, Nicolas Heess, Dorsa Sadigh, Jie Tan, Yuval Tassa, Fei Xia

**Abstract:** Large language models (LLMs) have demonstrated exciting progress in acquiring diverse new capabilities through in-context learning, ranging from logical reasoning to code-writing. Robotics researchers have also explored using LLMs to advance the capabilities of robotic control. However, since low-level robot actions are hardware-dependent and underrepresented in LLM training corpora, existing efforts in applying LLMs to robotics have largely treated LLMs as semantic planners or relied on human-engineered control primitives to interface with the robot. On the other hand, reward functions are shown to be flexible representations that can be optimized for control policies to achieve diverse tasks, while their semantic richness makes them suitable to be specified by LLMs. In this work, we introduce a new paradigm that harnesses this realization by utilizing LLMs to define reward parameters that can be optimized and accomplish variety of robotic tasks. Using reward as the intermediate interface generated by LLMs, we can effectively bridge the gap between high-level language instructions or corrections to low-level robot actions. Meanwhile, combining this with a real-time optimizer, MuJoCo MPC, empowers an interactive behavior creation experience where users can immediately observe the results and provide feedback to the system. To systematically evaluate the performance of our proposed method, we designed a total of 17 tasks for a simulated quadruped robot and a dexterous manipulator robot. We demonstrate that our proposed method reliably tackles $90%$ of the designed tasks, while a baseline using primitive skills as the interface with Code-as-policies achieves $50%$ of the tasks. We further validated our method on a real robot arm where complex manipulation skills such as non-prehensile pushing emerge through our interactive system.

**摘要:** 大型语言模型(英语:LLMs)显示了从逻辑推理到代码编写等各个方面,通过上下文学习获取各种新功能的令人兴奋的进展。机器人研究者还利用LLM来推进机器人控制的能力。然而,由于低级机器人行动依赖硬件,在LLM培训公司中缺乏代表性,目前在应用LLMs对机器人技术方面的现有努力大多把LLMs当作语义规划者或依赖于人造控制原型来与机器人接口。同时,结合实时优化器MuJoCo MPC,赋予用户可立即观察结果并向系统提供反馈的交互行为创建经验。为了系统地评估我们提出的方法的性能,我们设计了模拟四倍机器人和敏捷操纵机器人的17个任务。

**[Paper URL](https://proceedings.mlr.press/v229/yu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yu23a/yu23a.pdf)** 

# Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation
**题目:** 精制特征字段可启用少击语言指导的操作

**作者:** William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack Kaelbling, Phillip Isola

**Abstract:** Self-supervised and language-supervised image models contain rich knowledge of the world that is important for generalization. Many robotic tasks, however, require a detailed understanding of 3D geometry, which is often lacking in 2D image features. This work bridges this 2D-to-3D gap for robotic manipulation by leveraging distilled feature fields to combine accurate 3D geometry with rich semantics from 2D foundation models. We present a few-shot learning method for 6-DOF grasping and placing that harnesses these strong spatial and semantic priors to achieve in-the-wild generalization to unseen objects. Using features distilled from a vision-language model, CLIP, we present a way to designate novel objects for manipulation via free-text natural language, and demonstrate its ability to generalize to unseen expressions and novel categories of objects. Project website: https://f3rm.csail.mit.edu

**摘要:** 自我监督和语言监督的图像模型包含了对广义化的丰富知识。然而,许多机器人任务需要对3D几何的详细了解,这往往缺乏2D图像特征。这项工作通过利用蒸馏特征场结合精确的3D几何与2D基础模型中丰富的语义相结合,为机器人操纵提供了2D到3D的桥梁。我们提出了6-DOF把握和配置的几个射击学习方法,利用这些强的空间和语义预先来实现无形对象的野外广义化。使用从视觉语言模型中蒸馏特征,我们提出了一种通过自由文本自然语言进行操纵的新对象的设计方法,并展示它对无形表达和新对象类别的广义化的能力。

**[Paper URL](https://proceedings.mlr.press/v229/shen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shen23a/shen23a.pdf)** 

# Finetuning Offline World Models in the Real World
**题目:** 在现实世界中精细在线世界模型

**作者:** Yunhai Feng, Nicklas Hansen, Ziyan Xiong, Chandramouli Rajagopalan, Xiaolong Wang

**Abstract:** Reinforcement Learning (RL) is notoriously data-inefficient, which makes training on a real robot difficult. While model-based RL algorithms (world models) improve data-efficiency to some extent, they still require hours or days of interaction to learn skills. Recently, offline RL has been proposed as a framework for training RL policies on pre-existing datasets without any online interaction. However, constraining an algorithm to a fixed dataset induces a state-action distribution shift between training and inference, and limits its applicability to new tasks. In this work, we seek to get the best of both worlds: we consider the problem of pretraining a world model with offline data collected on a real robot, and then finetuning the model on online data collected by planning with the learned model. To mitigate extrapolation errors during online interaction, we propose to regularize the planner at test-time by balancing estimated returns and (epistemic) model uncertainty. We evaluate our method on a variety of visuo-motor control tasks in simulation and on a real robot, and find that our method enables few-shot finetuning to seen and unseen tasks even when offline data is limited. Videos are available at https://yunhaifeng.com/FOWM

**摘要:** 强化学习(RL)是众所周知的数据效率低,使得在真正的机器人上进行训练变得困难。虽然基于模型的RL算法(世界模型)在某种程度上提高了数据效率,但它们仍然需要几个小时或几天的交互才能学习技能。最近,非线性RL被提议作为培训RL政策的框架,在没有在线交互的情况下在现有的数据集上进行训练。然而,将一个算法限制在固定的数据集中,会诱导训练和推理之间的状态行动分布转变,并限制其适用于新任务的适用性。我们对仿真和实物机器人的各种视力运动控制任务进行了评估,发现我们的方法能够在有限的非线性数据的情况下,对视力和非视力任务进行少量 Shot Finetuning。

**[Paper URL](https://proceedings.mlr.press/v229/feng23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/feng23a/feng23a.pdf)** 

# Intent-Aware Planning in Heterogeneous Traffic via Distributed Multi-Agent Reinforcement Learning
**题目:** 通过分布式多代理强化学习在异种交通中具有意识的规划

**作者:** Xiyang Wu, Rohan Chandra, Tianrui Guan, Amrit Bedi, Dinesh Manocha

**Abstract:** Navigating safely and efficiently in dense and heterogeneous traffic scenarios is challenging for autonomous vehicles (AVs) due to their inability to infer the behaviors or intentions of nearby drivers. In this work, we introduce a distributed multi-agent reinforcement learning (MARL) algorithm for joint trajectory and intent prediction for autonomous vehicles in dense and heterogeneous environments. Our approach for intent-aware planning, iPLAN, allows agents to infer nearby drivers’ intents solely from their local observations. We model an explicit representation of agents’ private incentives: Behavioral Incentive for high-level decision-making strategy that sets planning sub-goals and Instant Incentive for low-level motion planning to execute sub-goals. Our approach enables agents to infer their opponents’ behavior incentives and integrate this inferred information into their decision-making and motion-planning processes. We perform experiments on two simulation environments, Non-Cooperative Navigation and Heterogeneous Highway. In Heterogeneous Highway, results show that, compared with centralized training decentralized execution (CTDE) MARL baselines such as QMIX and MAPPO, our method yields a $4.3%$ and $38.4%$ higher episodic reward in mild and chaotic traffic, with $48.1%$ higher success rate and $80.6%$ longer survival time in chaotic traffic. We also compare with a decentralized training decentralized execution (DTDE) baseline IPPO and demonstrate a higher episodic reward of $12.7%$ and $6.3%$ in mild traffic and chaotic traffic, $25.3%$ higher success rate, and $13.7%$ longer survival time.

**摘要:** 在密集 heterogeneous 的 交通 场景 中 安全 和 有效 地 航行 是 由于 汽车 无法 推断 附近 司机 的 行为 或 意图 而 对 汽车 造成 挑战 。 在 这项 工作 中, 我们 介绍 分布式 多 代理 增强 学习 ( MARL ) 算法, 用于 在 密集 和  heterogeneous 环境 中 对 汽车 的 联合 轨迹 和 意图 预测 。 我们 对 意图 意识 规划 的 方法, iPLAN, 允许 代理 只 从 他们 的 当地 观察 中 推断 附近 司机 的 意图 。我们对两个仿真环境,非合作导航和异性公路进行了实验。在异性公路中,结果表明,与集中训练分散执行(CTDE)MARL基线(如QMIX和MAPPO)相比,我们的方法在温和混沌交通中产生4.3%和38.4%的 episodic reward,在混沌交通中获得48.1%的成功率和80.6%的长寿时间。我们还与集中训练分散执行(DTDE)基线IPPO进行了比较,显示在温和混沌交通中获得12.7%和6.3%的 episodic reward,成功率高25.3%,长寿时间长 13.7%。

**[Paper URL](https://proceedings.mlr.press/v229/wu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wu23b/wu23b.pdf)** 

# PreCo: Enhancing Generalization in Co-Design of Modular Soft Robots via Brain-Body Pre-Training
**题目:** PreCo:通过脑-身体预训练,增强模块化软机器人共同设计的通用化

**作者:** Yuxing Wang, Shuang Wu, Tiantian Zhang, Yongzhe Chang, Haobo Fu, QIANG FU, Xueqian Wang

**Abstract:** Brain-body co-design, which involves the collaborative design of control strategies and morphologies, has emerged as a promising approach to enhance a robot’s adaptability to its environment. However, the conventional co-design process often starts from scratch, lacking the utilization of prior knowledge. This can result in time-consuming and costly endeavors. In this paper, we present PreCo, a novel methodology that efficiently integrates brain-body pre-training into the co-design process of modular soft robots. PreCo is based on the insight of embedding co-design principles into models, achieved by pre-training a universal co-design policy on a diverse set of tasks. This pre-trained co-designer is utilized to generate initial designs and control policies, which are then fine-tuned for specific co-design tasks. Through experiments on a modular soft robot system, our method demonstrates zero-shot generalization to unseen co-design tasks, facilitating few-shot adaptation while significantly reducing the number of policy iterations required.

**摘要:** 大脑-身体共设计,涉及控制策略和形态的协同设计,已成为提高机器人适应环境的有前途的途径。然而,传统的共设计过程往往从零开始,缺乏事前知识的利用。这可能导致耗费时间和昂贵的努力。本论文介绍一种新型方法,有效地将大脑-身体预训练纳入模块化软机器人共设计过程。通过在模块化软机器人系统中进行实验,我们的方法将零射程推广到未见的共同设计任务,从而有利于少射程的适应,同时大大减少所需的政策迭代。

**[Paper URL](https://proceedings.mlr.press/v229/wang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23b/wang23b.pdf)** 

# Diff-LfD: Contact-aware Model-based Learning from Visual Demonstration for Robotic Manipulation via Differentiable Physics-based Simulation and Rendering
**题目:** Diff-LfD:基于接触的模型学习,基于可变物理的仿真和渲染实现机器人操作的视觉演示

**作者:** Xinghao Zhu, JingHan Ke, Zhixuan Xu, Zhixin Sun, Bizhe Bai, Jun Lv, Qingtao Liu, Yuwei Zeng, Qi Ye, Cewu Lu, Masayoshi Tomizuka, Lin Shao

**Abstract:** Learning from Demonstration (LfD) is an efficient technique for robots to acquire new skills through expert observation, significantly mitigating the need for laborious manual reward function design. This paper introduces a novel framework for model-based LfD in the context of robotic manipulation. Our proposed pipeline is underpinned by two primary components: self-supervised pose and shape estimation and contact sequence generation. The former utilizes differentiable rendering to estimate object poses and shapes from demonstration videos, while the latter iteratively optimizes contact points and forces using differentiable simulation, consequently effectuating object transformations. Empirical evidence demonstrates the efficacy of our LfD pipeline in acquiring manipulation actions from human demonstrations. Complementary to this, ablation studies focusing on object tracking and contact sequence inference underscore the robustness and efficiency of our approach in generating long-horizon manipulation actions, even amidst environmental noise. Validation of our results extends to real-world deployment of the proposed pipeline. Supplementary materials and videos are available on our webpage.

**摘要:** LfD是机器人通过专家观察获得新技能的一种有效技术,大大缓解了人工奖励函数设计的必要性。本文介绍了基于模型的LfD在机器人操作中的新框架。我们提出的管道由两个主要组成部分支持:自监督的姿态和形状估计和接触序列生成。前者利用可微分渲染来估计对象姿态和形状,后者通过可微分模拟迭代优化接触点和力量,从而影响对象变换。除此之外,集中于对象跟踪和接触序列推导的研究突出了我们在制造长期操纵行动中的方法的鲁棒性和效率,即使在环境噪声中。我们的结果的验证可扩展到拟议的管道的实际部署。

**[Paper URL](https://proceedings.mlr.press/v229/zhu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhu23a/zhu23a.pdf)** 

# Surrogate Assisted Generation of Human-Robot Interaction Scenarios
**题目:** 人与机器人交互场景的替代辅助生成

**作者:** Varun Bhatt, Heramb Nemlekar, Matthew Christopher Fontaine, Bryon Tjanaka, Hejia Zhang, Ya-Chuan Hsu, Stefanos Nikolaidis

**Abstract:** As human-robot interaction (HRI) systems advance, so does the difficulty of evaluating and understanding the strengths and limitations of these systems in different environments and with different users. To this end, previous methods have algorithmically generated diverse scenarios that reveal system failures in a shared control teleoperation task. However, these methods require directly evaluating generated scenarios by simulating robot policies and human actions. The computational cost of these evaluations limits their applicability in more complex domains. Thus, we propose augmenting scenario generation systems with surrogate models that predict both human and robot behaviors. In the shared control teleoperation domain and a more complex shared workspace collaboration task, we show that surrogate assisted scenario generation efficiently synthesizes diverse datasets of challenging scenarios. We demonstrate that these failures are reproducible in real-world interactions.

**摘要:** 随着人类-机器人交互(HRI)系统的发展,在不同环境和不同用户之间对这些系统的优势和局限性进行评价和理解的难度也随之增加。为此目的,以前的方法通过算法生成了在共享控制远程操作任务中系统故障的多种场景。然而,这些方法需要通过模拟机器人政策和人类行动直接评价生成的场景。这些评估的计算成本限制了它们在更复杂的领域的应用。因此,我们建议用替代模型预测人类和机器人行为的增强场景生成系统。在共享控制远程操作领域和更复杂的共享工作空间协作任务中,我们证明替代辅助场景生成能有效地合成各种挑战场景的数据集。

**[Paper URL](https://proceedings.mlr.press/v229/bhatt23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/bhatt23a/bhatt23a.pdf)** 

# VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models
**题目:** VoxPoser:基于语言模型的机器人操作可编译的3D值图

**作者:** Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei

**Abstract:** Large language models (LLMs) are shown to possess a wealth of actionable knowledge that can be extracted for robot manipulation in the form of reasoning and planning. Despite the progress, most still rely on pre-defined motion primitives to carry out the physical interactions with the environment, which remains a major bottleneck. In this work, we aim to synthesize robot trajectories, i.e., a dense sequence of 6-DoF end-effector waypoints, for a large variety of manipulation tasks given an open-set of instructions and an open-set of objects. We achieve this by first observing that LLMs excel at inferring affordances and constraints given a free-form language instruction. More importantly, by leveraging their code-writing capabilities, they can interact with a vision-language model (VLM) to compose 3D value maps to ground the knowledge into the observation space of the agent. The composed value maps are then used in a model-based planning framework to zero-shot synthesize closed-loop robot trajectories with robustness to dynamic perturbations. We further demonstrate how the proposed framework can benefit from online experiences by efficiently learning a dynamics model for scenes that involve contact-rich interactions. We present a large-scale study of the proposed method in both simulated and real-robot environments, showcasing the ability to perform a large variety of everyday manipulation tasks specified in free-form natural language.

**摘要:** 大型语言模型(LLMs)显示,它们拥有大量可操作的知识,可以以推理和规划的形式提取用于机器人操作。尽管取得了进展,但大多数仍依靠预定义的运动原型来与环境进行物理交互,这仍然是一个主要瓶颈。在这项工作中,我们的目标是合成机器人轨迹,即6-DoF终点效应道点的密集序列,用于给一个开放式指令和一个开放式对象的大量操纵任务。我们通过首先观察LLMs excel at inferring affordances and constraints given a free-form language instruction实现这一点。本文通过对基于模型规划框架的复合值映射进行实例分析,以零射击对动态扰动具有鲁棒性的闭环机器人轨迹进行合成,并进一步说明该框架如何有效地学习接触丰富的交互作用场景的动态模型,从而从在线经验中获益。本文对该方法在模拟和实例机器人环境中进行了大规模研究,展示了能够在自由形式自然语言中完成多种日常操作任务的能力。

**[Paper URL](https://proceedings.mlr.press/v229/huang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/huang23b/huang23b.pdf)** 

# Stabilize to Act: Learning to Coordinate for Bimanual Manipulation
**题目:** 稳定行动:学习协调双人操纵

**作者:** Jennifer Grannen, Yilin Wu, Brandon Vu, Dorsa Sadigh

**Abstract:** Key to rich, dexterous manipulation in the real world is the ability to coordinate control across two hands. However, while the promise afforded by bimanual robotic systems is immense, constructing control policies for dual arm autonomous systems brings inherent difficulties. One such difficulty is the high-dimensionality of the bimanual action space, which adds complexity to both model-based and data-driven methods. We counteract this challenge by drawing inspiration from humans to propose a novel role assignment framework: a stabilizing arm holds an object in place to simplify the environment while an acting arm executes the task. We instantiate this framework with BimanUal Dexterity from Stabilization (BUDS), which uses a learned restabilizing classifier to alternate between updating a learned stabilization position to keep the environment unchanged, and accomplishing the task with an acting policy learned from demonstrations. We evaluate BUDS on four bimanual tasks of varying complexities on real-world robots, such as zipping jackets and cutting vegetables. Given only 20 demonstrations, BUDS achieves $76.9%$ task success across our task suite, and generalizes to out-of-distribution objects within a class with a $52.7%$ success rate. BUDS is $56.0%$ more successful than an unstructured baseline that instead learns a BC stabilizing policy due to the precision required of these complex tasks. Supplementary material and videos can be found at https://tinyurl.com/stabilizetoact.

**摘要:** 在现实世界中的丰富、敏捷操作的关键是双手控制的协调能力。然而,虽然双手机器人系统所提供的承诺是巨大的,但双手自主系统制订控制政策带来了固有困难。其中的一个困难是双手行动空间的高维度,这增加了模型和数据驱动方法的复杂性。我们通过从人类那里获得灵感来对付这一挑战,提出一种新的角色分配框架:一个稳定臂在执行任务时保持一个对象以简化环境。我们对实际机器人的四个双人任务的复杂性进行评估,例如夹克夹克和切菜。仅给出20个演示,BUDS在我们的任务套件中实现了$76.9%的任务成功,并将其推广到与$52.7%的成功率的类别内的非分配对象。BUDS比一个没有结构的基线更成功,因为这些复杂任务所需的精确性,因此学习了BC稳定政策。

**[Paper URL](https://proceedings.mlr.press/v229/grannen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/grannen23a/grannen23a.pdf)** 

# How to Learn and Generalize From Three Minutes of Data: Physics-Constrained and Uncertainty-Aware Neural Stochastic Differential Equations
**题目:** 如何从三分钟的数据中学习和概括:物理约束和不确定性意识的神经随机微分方程

**作者:** Franck Djeumou, Cyrus Neary, Ufuk Topcu

**Abstract:** We present a framework and algorithms to learn controlled dynamics models using neural stochastic differential equations (SDEs)—SDEs whose drift and diffusion terms are both parametrized by neural networks. We construct the drift term to leverage a priori physics knowledge as inductive bias, and we design the diffusion term to represent a distance-aware estimate of the uncertainty in the learned model’s predictions—it matches the system’s underlying stochasticity when evaluated on states near those from the training dataset, and it predicts highly stochastic dynamics when evaluated on states beyond the training regime. The proposed neural SDEs can be evaluated quickly enough for use in model predictive control algorithms, or they can be used as simulators for model-based reinforcement learning. Furthermore, they make accurate predictions over long time horizons, even when trained on small datasets that cover limited regions of the state space. We demonstrate these capabilities through experiments on simulated robotic systems, as well as by using them to model and control a hexacopter’s flight dynamics: A neural SDE trained using only three minutes of manually collected flight data results in a model-based control policy that accurately tracks aggressive trajectories that push the hexacopter’s velocity and Euler angles to nearly double the maximum values observed in the training dataset.

**摘要:** 我们提出了一种基于神经随机微分方程(SDEs)的控制动力学模型的框架和算法,该模型的漂移和扩散术语均由神经网络参数化。我们构造了漂移术语以利用先验物理知识作为诱导性偏见,并设计了扩散术语以表示学习模型的预测中不确定程度的距离感知估计,在训练数据集中接近状态的评估时与系统的基本随机性匹配,在训练制度以外的状态的评估时预测高度随机动力学。我们通过模拟机器人系统实验,以及通过它们来模拟和控制六架直升机的飞行动力学,证明了这些能力。 只使用三分钟的手动收集飞行数据,训练的神经元SDE实现了基于模型的控制政策,它准确地跟踪攻击性轨迹,使六架直升机的速度和欧勒角几乎达到训练数据集中观察到的最大值的两倍。

**[Paper URL](https://proceedings.mlr.press/v229/djeumou23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/djeumou23a/djeumou23a.pdf)** 

# Measuring Interpretability of Neural Policies of Robots with Disentangled Representation
**题目:** 分散表现机器人神经政策的解释性测定

**作者:** Tsun-Hsuan Wang, Wei Xiao, Tim Seyde, Ramin Hasani, Daniela Rus

**Abstract:** The advancement of robots, particularly those functioning in complex human-centric environments, relies on control solutions that are driven by machine learning. Understanding how learning-based controllers make decisions is crucial since robots are mostly safety-critical systems. This urges a formal and quantitative understanding of the explanatory factors in the interpretability of robot learning. In this paper, we aim to study interpretability of compact neural policies through the lens of disentangled representation. We leverage decision trees to obtain factors of variation [1] for disentanglement in robot learning; these encapsulate skills, behaviors, or strategies toward solving tasks. To assess how well networks uncover the underlying task dynamics, we introduce interpretability metrics that measure disentanglement of learned neural dynamics from a concentration of decisions, mutual information and modularity perspective. We showcase the effectiveness of the connection between interpretability and disentanglement consistently across extensive experimental analysis.

**摘要:** 机器人的发展,特别是在复杂的人中心环境中运行的机器人,依赖于由机器学习驱动的控制解决方案。理解基于学习的控制器如何作出决策是至关重要的,因为机器人大多是安全关键系统。这促使人们对机器人学习的解释因素的正式和定量理解。本论文的目标是通过分散表示的镜头研究紧凑神经政策的解释性。我们利用决策树来获取机器人学习中的分散因素,这些因素包含技能、行为或解决任务的战略。在广泛的实验分析中,我们不断展示了解释性和分离之间的联系的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/wang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23c/wang23c.pdf)** 

# RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools
**题目:** 罗博库克:多种工具来操纵长边形弹性物体

**作者:** Haochen Shi, Huazhe Xu, Samuel Clarke, Yunzhu Li, Jiajun Wu

**Abstract:** Humans excel in complex long-horizon soft body manipulation tasks via flexible tool use: bread baking requires a knife to slice the dough and a rolling pin to flatten it. Often regarded as a hallmark of human cognition, tool use in autonomous robots remains limited due to challenges in understanding tool-object interactions. Here we develop an intelligent robotic system, RoboCook, which perceives, models, and manipulates elasto-plastic objects with various tools. RoboCook uses point cloud scene representations, models tool-object interactions with Graph Neural Networks (GNNs), and combines tool classification with self-supervised policy learning to devise manipulation plans. We demonstrate that from just 20 minutes of real-world interaction data per tool, a general-purpose robot arm can learn complex long-horizon soft object manipulation tasks, such as making dumplings and alphabet letter cookies. Extensive evaluations show that RoboCook substantially outperforms state-of-the-art approaches, exhibits robustness against severe external disturbances, and demonstrates adaptability to different materials.

**摘要:** 通过灵活的工具使用,人类 excel in complex long-horizon soft body manipulation tasks: bread baking requires a knife to slice the dough and a rolling pin to flatten it. 通常被认为是人类认知的标志,自主机器人的工具使用仍然受到理解工具-对象交互的挑战限制。广泛的评估表明,RoboCook大大超过了最先进的方法,表现出对严重外部干扰的鲁棒性,并表现出对不同材料的适应性。

**[Paper URL](https://proceedings.mlr.press/v229/shi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shi23a/shi23a.pdf)** 

# Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners
**题目:** 请求帮助的机器人:大型语言模型规划人员的不确定性配置

**作者:** Allen Z. Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, Zhenjia Xu, Dorsa Sadigh, Andy Zeng, Anirudha Majumdar

**Abstract:** Large language models (LLMs) exhibit a wide range of promising capabilities — from step-by-step planning to commonsense reasoning — that may provide utility for robots, but remain prone to confidently hallucinated predictions. In this work, we present KnowNo, a framework for measuring and aligning the uncertainty of LLM-based planners, such that they know when they don’t know, and ask for help when needed. KnowNo builds on the theory of conformal prediction to provide statistical guarantees on task completion while minimizing human help in complex multi-step planning settings. Experiments across a variety of simulated and real robot setups that involve tasks with different modes of ambiguity (for example, from spatial to numeric uncertainties, from human preferences to Winograd schemas) show that KnowNo performs favorably over modern baselines (which may involve ensembles or extensive prompt tuning) in terms of improving efficiency and autonomy, while providing formal assurances. KnowNo can be used with LLMs out-of-the-box without model-finetuning, and suggests a promising lightweight approach to modeling uncertainty that can complement and scale with the growing capabilities of foundation models.

**摘要:** 大型语言模型(LLMs)展示了一系列有前途的能力 — — 从逐步规划到常识推理 — — 可为机器人提供实用性,但仍易于自信地幻觉预测。 本文介绍KnowNo,一种基于LLM的规划者测量和调整不确定性的框架,使得他们知道他们不知道时,并在需要时求助。KnowNo建立在符合预测理论之上,以提供完成任务的统计保证,同时在复杂的多步骤规划设置中尽量减少人类的帮助。KnowNo可以在无模型精确化的情况下使用LLM,并提出一种可以补充和扩大基础模型的增长能力的轻量方法来建模不确定性。

**[Paper URL](https://proceedings.mlr.press/v229/ren23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ren23a/ren23a.pdf)** 

# Robot Learning with Sensorimotor Pre-training
**题目:** 感应运动预训练的机器人学习

**作者:** Ilija Radosavovic, Baifeng Shi, Letian Fu, Ken Goldberg, Trevor Darrell, Jitendra Malik

**Abstract:** We present a self-supervised sensorimotor pre-training approach for robotics. Our model, called RPT, is a Transformer that operates on sequences of sensorimotor tokens. Given a sequence of camera images, proprioceptive robot states, and actions, we encode the sequence into tokens, mask out a subset, and train a model to predict the missing content from the rest. We hypothesize that if a robot can predict the masked-out content it will have acquired a good model of the physical world that can enable it to act. RPT is designed to operate on latent visual representations which makes prediction tractable, enables scaling to larger models, and allows fast inference on a real robot. To evaluate our approach, we collected a dataset of 20,000 real-world trajectories over 9 months using a combination of motion planning and grasping algorithms. We find that sensorimotor pre-training consistently outperforms training from scratch, has favorable scaling properties, and enables transfer across different tasks, environments, and robots.

**摘要:** 我们提出了一种自我监督的感官运动预训练方法,即RPT(英语:Sensorimotor pre-training approach),是一种基于感官运动符号序列的变换器。 基于摄像机图像、感官机器人状态和动作的序列,我们将序列编码成符号,掩盖一个子集,并训练一个模型来预测其他部分的不足内容。我们发现,感应运动预训练能持续超越从头开始的训练,具有有利的尺度特性,并可通过不同的任务、环境和机器人进行转移。

**[Paper URL](https://proceedings.mlr.press/v229/radosavovic23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/radosavovic23a/radosavovic23a.pdf)** 

# RVT: Robotic View Transformer for 3D Object Manipulation
**题目:** RVT:用于3D对象操作的机器人视图变换器

**作者:** Ankit Goyal, Jie Xu, Yijie Guo, Valts Blukis, Yu-Wei Chao, Dieter Fox

**Abstract:** For 3D object manipulation, methods that build an explicit 3D representation perform better than those relying only on camera images. But using explicit 3D representations like voxels comes at large computing cost, adversely affecting scalability. In this work, we propose RVT, a multi-view transformer for 3D manipulation that is both scalable and accurate. Some key features of RVT are an attention mechanism to aggregate information across views and re-rendering of the camera input from virtual views around the robot workspace. In simulations, we find that a single RVT model works well across 18 RLBench tasks with 249 task variations, achieving $26%$ higher relative success than the existing state-of-the-art method (PerAct). It also trains 36X faster than PerAct for achieving the same performance and achieves 2.3X the inference speed of PerAct. Further, RVT can perform a variety of manipulation tasks in the real world with just a few ($\sim$10) demonstrations per task. Visual results, code, and trained model are provided at: https://robotic-view-transformer.github.io/.

**摘要:** 对于3D对象的操作,建立一个明确的3D表示方法比仅依靠摄像机图像做的更好。但是使用明确的3D表示方法如 voxels会带来巨大的计算成本,不利于可扩展性。在这个工作中,我们提出了RVT,一种可扩展和准确的3D操作的多视变换器。RVT的一些关键特征是通过视图集聚信息并从机器人工作空间周围的虚拟视图重新提供摄像机输入的注意力机制。在仿真中,我们发现一个单个RVT模型在18个RLBench任务中使用249个任务变异,获得比现有的最先进的方法(PerAct)高26美元的相对成功。可视结果、代码和训练模型提供于: https://robotic-view-transformer.github.io/。

**[Paper URL](https://proceedings.mlr.press/v229/goyal23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/goyal23a/goyal23a.pdf)** 

# ViNT: A Foundation Model for Visual Navigation
**题目:** ViNT:视觉导航的基础模型

**作者:** Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachowicz, Kevin Black, Noriaki Hirose, Sergey Levine

**Abstract:** General-purpose pre-trained models (“foundation models”) have enabled practitioners to produce generalizable solutions for individual machine learning problems with datasets that are significantly smaller than those required for learning from scratch. Such models are typically trained on large and diverse datasets with weak supervision, consuming much more training data than is available for any individual downstream application. In this paper, we describe the Visual Navigation Transformer (ViNT), a foundation model that aims to bring the success of general-purpose pre-trained models to vision-based robotic navigation. ViNT is trained with a general goal-reaching objective that can be used with any navigation dataset, and employs a flexible Transformer-based architecture to learn navigational affordances and enable efficient adaptation to a variety of downstream navigational tasks. ViNT is trained on a number of existing navigation datasets, comprising hundreds of hours of robotic navigation from a variety of different robotic platforms, and exhibits positive transfer, outperforming specialist models trained on narrower datasets. ViNT can be augmented with diffusion-based goal proposals to explore novel environments, and can solve kilometer-scale navigation problems when equipped with long-range heuristics. ViNT can also be adapted to novel task specifications with a technique inspired by prompt-tuning, where the goal encoder is replaced by an encoding of another task modality (e.g., GPS waypoints or turn-by-turn directions) embedded into the same space of goal tokens. This flexibility and ability to accommodate a variety of downstream problem domains establish ViNT as an effective foundation model for mobile robotics.

**摘要:** 通用预训练模型(英语:General-purpose pre-trained models,缩写为“基础模型”)使专业人员能够为基于视觉的机器人导航实现通用化解决方案,其数据集比从头到尾学习所需的数据集要小得多。 这类模型通常被训练在大型和多样的数据集中,并受弱监督,消耗的训练数据比任何单独的下游应用都多。ViNT由多个现有的导航数据集训练,包括来自各种不同机器人平台的数百小时的机器人导航,并展示了积极的传输,超过了对较窄的数据集训练的专家模型。ViNT可以增加基于扩散的目标提议,以探索新环境,并在装备长程回旋技术时解决 kilometer-scale导航问题。ViNT也可以以快速调制的灵感技术适应新任务规格,目标编码器被嵌入相同目标符号空间的另一个任务模式(例如GPS路径点或转转方向)的编码器所取代。

**[Paper URL](https://proceedings.mlr.press/v229/shah23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shah23a/shah23a.pdf)** 

# What Went Wrong? Closing the Sim-to-Real Gap via Differentiable Causal Discovery
**题目:** 什么是错误?通过可辨别的因果发现关闭模拟到真实差距

**作者:** Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis, Bingqing Chen, Ding Zhao

**Abstract:** Training control policies in simulation is more appealing than on real robots directly, as it allows for exploring diverse states in an efficient manner. Yet, robot simulators inevitably exhibit disparities from the real-world \rebut{dynamics}, yielding inaccuracies that manifest as the dynamical simulation-to-reality (sim-to-real) gap. Existing literature has proposed to close this gap by actively modifying specific simulator parameters to align the simulated data with real-world observations. However, the set of tunable parameters is usually manually selected to reduce the search space in a case-by-case manner, which is hard to scale up for complex systems and requires extensive domain knowledge. To address the scalability issue and automate the parameter-tuning process, we introduce COMPASS, which aligns the simulator with the real world by discovering the causal relationship between the environment parameters and the sim-to-real gap. Concretely, our method learns a differentiable mapping from the environment parameters to the differences between simulated and real-world robot-object trajectories. This mapping is governed by a simultaneously learned causal graph to help prune the search space of parameters, provide better interpretability, and improve generalization on unseen parameters. We perform experiments to achieve both sim-to-sim and sim-to-real transfer, and show that our method has significant improvements in trajectory alignment and task success rate over strong baselines in several challenging manipulation tasks. Demos are available on our project website: https://sites.google.com/view/sim2real-compass.

**摘要:** 在仿真中,训练控制政策比直接仿真机器人更吸引人,因为它能有效探索各种状态。然而,机器人仿真器必然表现出现实世界\rebut{dynamics}的差异,产生动态仿真-现实(sim-to-real)差距的误差。现有文献提出了通过积极修改特定仿真器参数来实现仿真数据与现实世界观测的匹配,以消除这一差距。然而,可调动参数的集合通常是手动选择的,以减少搜索空间,这对于复杂系统来说是很难提高的,需要广泛的领域知识。具体地说,我们的方法从环境参数学习可区分的映射到模拟和现实机器人-对象轨迹之间的差异。该映射由同时学习的因果图控制,帮助缩小参数的搜索空间,提供更好的解释性,并改进未见参数的一般化。我们进行了实验,以实现模拟和模拟的转换,并表明我们的方法在几个挑战性操作任务中具有显著的轨迹调整和任务成功率上的优势。

**[Paper URL](https://proceedings.mlr.press/v229/huang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/huang23c/huang23c.pdf)** 

# Scalable Deep Kernel Gaussian Process for Vehicle Dynamics in Autonomous Racing
**题目:** 自动赛车车辆动力学的可尺度深核高斯过程

**作者:** Jingyun Ning, Madhur Behl

**Abstract:** Autonomous racing presents a challenging environment for testing the limits of autonomous vehicle technology. Accurately modeling the vehicle dynamics (with all forces and tires) is critical for high-speed racing, but it remains a difficult task and requires an intricate balance between run-time computational demands and modeling complexity. Researchers have proposed utilizing learning-based methods such as Gaussian Process (GP) for learning vehicle dynamics. However, current approaches often oversimplify the modeling process or apply strong assumptions, leading to unrealistic results that cannot translate to real-world settings. In this paper, we proposed DKL-SKIP method for vehicle dynamics modeling. Our approach outperforms standard GP methods and the N4SID system identification technique in terms of prediction accuracy. In addition to evaluating DKL-SKIP on real-world data, we also evaluate its performance using a high-fidelity autonomous racing AutoVerse simulator. The results highlight the potential of DKL-SKIP as a promising tool for modeling complex vehicle dynamics in both real-world and simulated environments.

**摘要:** 自主赛车是测试自主赛车技术局限性的一个挑战性环境。准确地建模车辆动力学(包括所有力量和轮胎)对于高速赛车具有关键意义,但仍是一个艰巨的任务,需要在运行时间计算要求和建模复杂性之间实现复杂的平衡。研究人员提出了使用基于学习的方法,例如高斯过程(GP)来学习车辆动力学。然而,当前的方法往往过于简化建模过程或应用强假设,导致无法转化为现实环境的结果。结果突出了DKL-SKIP作为现实和模拟环境中复杂车辆动力学建模的有前途工具的潜力。

**[Paper URL](https://proceedings.mlr.press/v229/ning23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ning23a/ning23a.pdf)** 

# Autonomous Robotic Reinforcement Learning with Asynchronous Human Feedback
**题目:** 基于异步人反馈的自主机器人增强学习

**作者:** Max Balsells, Marcel Torne Villasevil, Zihan Wang, Samedh Desai, Pulkit Agrawal, Abhishek Gupta

**Abstract:** Ideally, we would place a robot in a real-world environment and leave it there improving on its own by gathering more experience autonomously. However, algorithms for autonomous robotic learning have been challenging to realize in the real world. While this has often been attributed to the challenge of sample complexity, even sample-efficient techniques are hampered by two major challenges - the difficulty of providing well “shaped" rewards, and the difficulty of continual reset-free training. In this work, we describe a system for real-world reinforcement learning that enables agents to show continual improvement by training directly in the real world without requiring painstaking effort to hand-design reward functions or reset mechanisms. Our system leverages occasional non-expert human-in-the-loop feedback from remote users to learn informative distance functions to guide exploration while leveraging a simple self-supervised learning algorithm for goal-directed policy learning. We show that in the absence of resets, it is particularly important to account for the current “reachability" of the exploration policy when deciding which regions of the space to explore. Based on this insight, we instantiate a practical learning system - GEAR, which enables robots to simply be placed in real-world environments and left to train autonomously without interruption. The system streams robot experience to a web interface only requiring occasional asynchronous feedback from remote, crowdsourced, non-expert humans in the form of binary comparative feedback. We evaluate this system on a suite of robotic tasks in simulation and demonstrate its effectiveness at learning behaviors both in simulation and the real world. Project website https://guided-exploration-autonomous-rl.github.io/GEAR/.

**摘要:** 理想情况下,我们将机器人放置在现实环境中,并通过自力更生地收集更多经验来改善它。然而,自力更生机器人学习算法在现实环境中难以实现。虽然这常常被归因于样品复杂度的挑战,但即使样品高效的技术也受到两个主要挑战的阻碍——提供良好“形状”的奖励的困难,以及持续的无重置训练的困难。我们证明,在没有重新设置的情况下,在决定探索空间的哪些区域时,特别重要的是考虑到探索政策的当前“可达性”。基于这一洞察力,我们建立了一种实用的学习系统-GEAR,它使机器人可以简单地放置在现实环境中,并且可以不受干扰地自主训练。该系统将机器人的经验流入Web接口,仅需要从远程、集群、非专家人提供偶尔的异步反馈,以二进制比较反馈的形式。我们对该系统在模拟中的一系列机器人任务进行评估,并证明它在模拟和现实世界中的学习行为中具有有效性。

**[Paper URL](https://proceedings.mlr.press/v229/balsells23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/balsells23a/balsells23a.pdf)** 

# Learning Realistic Traffic Agents in Closed-loop
**题目:** 在封闭循环中学习现实交通代理

**作者:** Chris Zhang, James Tu, Lunjun Zhang, Kelvin Wong, Simon Suo, Raquel Urtasun

**Abstract:** Realistic traffic simulation is crucial for developing self-driving software in a safe and scalable manner prior to real-world deployment. Typically, imitation learning (IL) is used to learn human-like traffic agents directly from real-world observations collected offline, but without explicit specification of traffic rules, agents trained from IL alone frequently display unrealistic infractions like collisions and driving off the road. This problem is exacerbated in out-of-distribution and long-tail scenarios. On the other hand, reinforcement learning (RL) can train traffic agents to avoid infractions, but using RL alone results in unhuman-like driving behaviors. We propose Reinforcing Traffic Rules (RTR), a holistic closed-loop learning objective to match expert demonstrations under a traffic compliance constraint, which naturally gives rise to a joint IL + RL approach, obtaining the best of both worlds. Our method learns in closed-loop simulations of both nominal scenarios from real-world datasets as well as procedurally generated long-tail scenarios. Our experiments show that RTR learns more realistic and generalizable traffic simulation policies, achieving significantly better tradeoffs between human-like driving and traffic compliance in both nominal and long-tail scenarios. Moreover, when used as a data generation tool for training prediction models, our learned traffic policy leads to considerably improved downstream prediction metrics compared to baseline traffic agents.

**摘要:** 现实交通仿真对于开发自驾驶软件在现实世界部署之前具有重要的安全性和可扩展性。通常,仿真学习(IL)用于直接从网络收集的现实世界观测中学习类似人类的交通代理人,但没有明确的交通规则规范,由IL训练的代理人经常显示不现实的违背行为,如碰撞和离开道路的驾驶。这一问题在非分布和长尾场景中加剧。研究结果表明,RTR学习的交通仿真策略具有更现实和可推广性,使人型驾驶和交通遵守在名义和长尾的交通仿真策略中具有显著的优越性。此外,当当当用作训练预测模型的数据生成工具时,我们学习的交通政策可以大大改善下游预测指标,与基线交通代理相比。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23b/zhang23b.pdf)** 

# Leveraging 3D Reconstruction for Mechanical Search on Cluttered Shelves
**题目:** 利用三维重建技术对夹杂架的机械搜索

**作者:** Seungyeon Kim, Young Hun Kim, Yonghyeon Lee, Frank C. Park

**Abstract:** Finding and grasping a target object on a cluttered shelf, especially when the target is occluded by other unknown objects and initially invisible, remains a significant challenge in robotic manipulation. While there have been advances in finding the target object by rearranging surrounding objects using specialized tools, developing algorithms that work with standard robot grippers remains an unresolved issue. In this paper, we introduce a novel framework for finding and grasping the target object using a standard gripper, employing pushing and pick and-place actions. To achieve this, we introduce two indicator functions: (i) an existence function, determining the potential presence of the target, and (ii) a graspability function, assessing the feasibility of grasping the identified target. We then formulate a model-based optimal control problem. The core component of our approach involves leveraging a 3D recognition model, enabling efficient estimation of the proposed indicator functions and their associated dynamics models. Our method succeeds in finding and grasping the target object using a standard robot gripper in both simulations and real-world settings. In particular, we demonstrate the adaptability and robustness of our method in the presence of noise in real-world vision sensor data. The code for our framework is available at https://github.com/seungyeon-k/Search-for-Grasp-public.

**摘要:** 基于模型的优化控制问题,本文提出了一种基于模型的优化控制问题,即基于模型的优化控制问题。我们的方法成功地在模拟和现实环境中使用标准机器人握手找到和把握目标对象,特别是在现实世界视觉传感器数据中存在噪声时,我们证明了该方法的适应性和鲁棒性。我们框架的代码可于 https://github.com/seungyeon-k/Search-for-Grasp-public。

**[Paper URL](https://proceedings.mlr.press/v229/kim23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kim23a/kim23a.pdf)** 

# SCONE: A Food Scooping Robot Learning Framework with Active Perception
**题目:** SCONE:一种具有主动感知的食品收集机器人学习框架

**作者:** Yen-Ling Tai, Yu Chien Chiu, Yu-Wei Chao, Yi-Ting Chen

**Abstract:** Effectively scooping food items poses a substantial challenge for current robotic systems, due to the intricate states and diverse physical properties of food. To address this challenge, we believe in the importance of encoding food items into meaningful representations for effective food scooping. However, the distinctive properties of food items, including deformability, fragility, fluidity, or granularity, pose significant challenges for existing representations. In this paper, we investigate the potential of active perception for learning meaningful food representations in an implicit manner. To this end, we present SCONE, a food-scooping robot learning framework that leverages representations gained from active perception to facilitate food scooping policy learning. SCONE comprises two crucial encoding components: the interactive encoder and the state retrieval module. Through the encoding process, SCONE is capable of capturing properties of food items and vital state characteristics. In our real-world scooping experiments, SCONE excels with a $71%$ success rate when tasked with 6 previously unseen food items across three different difficulty levels, surpassing state-of-theart methods. This enhanced performance underscores SCONE’s stability, as all food items consistently achieve task success rates exceeding $50%$. Additionally, SCONE’s impressive capacity to accommodate diverse initial states enables it to precisely evaluate the present condition of the food, resulting in a compelling scooping success rate. For further information, please visit our website: https://sites.google.com/view/corlscone/home.

**摘要:** 由于食品的复杂状态和多种物理特性,有效捕食食品是当前机器人系统面临的一个重大挑战。为了解决这一挑战,我们相信将食物物品编码成有效的捕食食品的有意义表示的重要性。然而,食品物品的独特特性,包括变形性、脆弱性、流动性或粒度,对现有的表示构成了重大挑战。本论文研究了主动感知的潜在性,以隐含的方式学习有意义的捕食食品的表示。为此,我们提出了一种捕食食品机器人学习框架,它利用主动感知获得的表示来促进捕食食品的政策学习。通过编码过程,SCONE能够捕捉食物物品的特性和关键状态特征。在我们的实物捕食实验中,SCONE以71%的成功率胜出6件 previously unseen food items across three different difficulty levels,超越心脏状态的方法。这增强的性能突出了SCONE的稳定性,因为所有食物物品均能达到超过50%的任务成功率。此外,SCONE的适应各种初始状态的能力使它能够准确地评估食物的现况,从而达到令人引人注目的捕食成功率。

**[Paper URL](https://proceedings.mlr.press/v229/tai23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tai23a/tai23a.pdf)** 

# Fine-Tuning Generative Models as an Inference Method for Robotic Tasks
**题目:** 微调生成模型作为机器人任务的迭代方法

**作者:** Orr Krupnik, Elisei Shafer, Tom Jurgenson, Aviv Tamar

**Abstract:** Adaptable models could greatly benefit robotic agents operating in the real world, allowing them to deal with novel and varying conditions. While approaches such as Bayesian inference are well-studied frameworks for adapting models to evidence, we build on recent advances in deep generative models which have greatly affected many areas of robotics. Harnessing modern GPU acceleration, we investigate how to quickly adapt the sample generation of neural network models to observations in robotic tasks. We propose a simple and general method that is applicable to various deep generative models and robotic environments. The key idea is to quickly fine-tune the model by fitting it to generated samples matching the observed evidence, using the cross-entropy method. We show that our method can be applied to both autoregressive models and variational autoencoders, and demonstrate its usability in object shape inference from grasping, inverse kinematics calculation, and point cloud completion.

**摘要:** 适应性模型可以极大地有利于在现实世界中运行的机器人代理人,使他们能够处理新颖和变化的条件。虽然贝叶斯推理等方法是对模型适应证据的深入研究框架,但我们建立在深层生成模型的最近进步中,这些进展极大地影响了许多机器人领域。利用现代 GPU加速,我们研究如何快速适应神经网络模型的样本生成对机器人任务的观察。我们提出了一种简单和通用的方法,适用于各种深层生成模型和机器人环境。

**[Paper URL](https://proceedings.mlr.press/v229/krupnik23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/krupnik23a/krupnik23a.pdf)** 

# Learning to Design and Use Tools for Robotic Manipulation
**题目:** 学习设计和使用机器人操纵工具

**作者:** Ziang Liu, Stephen Tian, Michelle Guo, Karen Liu, Jiajun Wu

**Abstract:** When limited by their own morphologies, humans and some species of animals have the remarkable ability to use objects from the environment toward accomplishing otherwise impossible tasks. Robots might similarly unlock a range of additional capabilities through tool use. Recent techniques for jointly optimizing morphology and control via deep learning are effective at designing locomotion agents. But while outputting a single morphology makes sense for locomotion, manipulation involves a variety of strategies depending on the task goals at hand. A manipulation agent must be capable of rapidly prototyping specialized tools for different goals. Therefore, we propose learning a designer policy, rather than a single design. A designer policy is conditioned on task information and outputs a tool design that helps solve the task. A design-conditioned controller policy can then perform manipulation using these tools. In this work, we take a step towards this goal by introducing a reinforcement learning framework for jointly learning these policies. Through simulated manipulation tasks, we show that this framework is more sample efficient than prior methods in multi-goal or multi-variant settings, can perform zero-shot interpolation or fine-tuning to tackle previously unseen goals, and allows tradeoffs between the complexity of design and control policies under practical constraints. Finally, we deploy our learned policies onto a real robot. Please see our supplementary video and website at https://robotic-tool-design.github.io/ for visualizations.

**摘要:** 人类和一些动物的形态被限制时,它们具有利用环境中的物体完成其他不可能的任务的能力。机器人也可以通过工具使用来解锁一系列额外的能力。最近联合优化形态和通过深层学习控制的技术在设计运动代理时有效。但是,在输出单一的形态为运动有意义时,操纵涉及不同的策略,取决于所掌握的任务目标。操纵代理必须能够快速制备不同目标的专门工具的原型。因此,我们建议学习设计策略,而不是单一的设计。设计策略取决于任务信息,输出工具设计,帮助解决任务。设计条件的控制策略可以使用这些工具进行操纵。在这项工作中,我们通过引入加强学习框架来共同学习这些政策,迈向这一目标。通过模拟操纵任务,我们表明,这个框架比多目标或多变量设置的先前方法更有效,可以执行零射击插值或微调,以解决以前未见的目标,并允许在实际约束下设计和控制政策的复杂性之间进行交换。最后,我们将我们的学习政策部署到真正的机器人上。请参阅我们的补充视频和网站 https://robotic-tool-design.github.io/ 来进行可视化。

**[Paper URL](https://proceedings.mlr.press/v229/liu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23b/liu23b.pdf)** 

# CLUE: Calibrated Latent Guidance for Offline Reinforcement Learning
**题目:**  CLUE:校正在线强化学习的潜在指导

**作者:** Jinxin Liu, Lipeng Zu, Li He, Donglin Wang

**Abstract:** Offline reinforcement learning (RL) aims to learn an optimal policy from pre-collected and labeled datasets, which eliminates the time-consuming data collection in online RL. However, offline RL still bears a large burden of specifying/handcrafting extrinsic rewards for each transition in the offline data. As a remedy for the labor-intensive labeling, we propose to endow offline RL tasks with a few expert data and utilize the limited expert data to drive intrinsic rewards, thus eliminating the need for extrinsic rewards. To achieve that, we introduce Calibrated Latent gUidancE (CLUE), which utilizes a conditional variational auto-encoder to learn a latent space such that intrinsic rewards can be directly qualified over the latent space. CLUE’s key idea is to align the intrinsic rewards consistent with the expert intention via enforcing the embeddings of expert data to a calibrated contextual representation. We instantiate the expert-driven intrinsic rewards in sparse-reward offline RL tasks, offline imitation learning (IL) tasks, and unsupervised offline RL tasks. Empirically, we find that CLUE can effectively improve the sparse-reward offline RL performance, outperform the state-of-the-art offline IL baselines, and discover diverse skills from static reward-free offline data.

**摘要:** 非线性增强学习(Offline reinforcement learning,RL)旨在从预收集和标记数据集学习最佳政策,从而消除在在线RL中耗时的数据收集。然而,非线性学习(Offline RL)仍承受着在非线性数据中的每个过渡中指定/手工制作外部奖励的巨大负担。作为劳力密集的标记的补救,我们建议用少数专家数据为非线性学习任务提供非线性任务,并利用有限的专家数据驱动内部奖励,从而消除外部奖励的需求。通过实例分析,我们发现CLUE能够有效地提高低收益的低收益的 offline RL性能,超越最先进的 offline IL 基线,并从静态无奖励的 offline 数据中发现各种技能。

**[Paper URL](https://proceedings.mlr.press/v229/liu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23c/liu23c.pdf)** 

# DEFT: Dexterous Fine-Tuning for Hand Policies
**题目:** DEFT:手动策略的精细调制

**作者:** Aditya Kannan, Kenneth Shaw, Shikhar Bahl, Pragna Mannam, Deepak Pathak

**Abstract:** Dexterity is often seen as a cornerstone of complex manipulation. Humans are able to perform a host of skills with their hands, from making food to operating tools. In this paper, we investigate these challenges, especially in the case of soft, deformable objects as well as complex, relatively long-horizon tasks. Although, learning such behaviors from scratch can be data inefficient. To circumvent this, we propose a novel approach, DEFT (DExterous Fine-Tuning for Hand Policies), that leverages human-driven priors, which are executed directly in the real world. In order to improve upon these priors, DEFT involves an efficient online optimization procedure. With the integration of human-based learning and online fine-tuning, coupled with a soft robotic hand, DEFT demonstrates success across various tasks, establishing a robust, data-efficient pathway toward general dexterous manipulation. Please see our website at https://dexterousfinetuning.github.io for video results.

**摘要:** 敏捷性通常被看作是复杂操纵的基石。人类能够用自己的手进行一系列技能,从制造食物到操作工具。在这个论文中,我们研究了这些挑战,特别是软变形的对象以及复杂、相对长的水平任务。尽管,从零开始学习这些行为是数据效率低下的,但为了绕过这一问题,我们提出了一种新方法,DEFT(DExterous Fine-Tuning for Hand Policies),它利用人为驱动的预先,直接执行在现实世界中。为了改善这些预先,DEFT涉及有效的在线优化程序。

**[Paper URL](https://proceedings.mlr.press/v229/kannan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kannan23a/kannan23a.pdf)** 

# One-Shot Imitation Learning: A Pose Estimation Perspective
**题目:** 一击模仿学习:动作估计视角

**作者:** Pietro Vitiello, Kamil Dreczkowski, Edward Johns

**Abstract:** In this paper, we study imitation learning under the challenging setting of: (1) only a single demonstration, (2) no further data collection, and (3) no prior task or object knowledge. We show how, with these constraints, imitation learning can be formulated as a combination of trajectory transfer and unseen object pose estimation. To explore this idea, we provide an in-depth study on how state-of-the-art unseen object pose estimators perform for one-shot imitation learning on ten real-world tasks, and we take a deep dive into the effects that camera calibration, pose estimation error, and spatial generalisation have on task success rates. For videos, please visit www.robot-learning.uk/pose-estimation-perspective.

**摘要:** 本文研究了一种具有挑战性条件的仿真学习方法,即:(一)只进行一次演示,(二)没有进一步数据采集,(三)没有先前的任务或对象知识。我们展示了仿真学习如何以轨迹转移和未见物体姿态估计相结合的方式进行仿真学习。为了探索这一想法,我们提供了关于最先进的未见物体姿态估计器如何在10个实物任务中进行一次仿真学习的深入研究,并深入探讨了摄像机校正、姿态估计误差和空间推广对任务成功率的影响。

**[Paper URL](https://proceedings.mlr.press/v229/vitiello23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/vitiello23a/vitiello23a.pdf)** 

# Semantic Mechanical Search with Large Vision and Language Models
**题目:** 大型视觉和语言模型的语义机械搜索

**作者:** Satvik Sharma, Huang Huang, Kaushik Shivakumar, Lawrence Yunliang Chen, Ryan Hoque, Brian Ichter, Ken Goldberg

**Abstract:** Moving objects to find a fully-occluded target object, known as mechanical search, is a challenging problem in robotics. As objects are often organized semantically, we conjecture that semantic information about object relationships can facilitate mechanical search and reduce search time. Large pretrained vision and language models (VLMs and LLMs) have shown promise in generalizing to uncommon objects and previously unseen real-world environments. In this work, we propose a novel framework called Semantic Mechanical Search (SMS). SMS conducts scene understanding and generates a semantic occupancy distribution explicitly using LLMs. Compared to methods that rely on visual similarities offered by CLIP embeddings, SMS leverages the deep reasoning capabilities of LLMs. Unlike prior work that uses VLMs and LLMs as end-to-end planners, which may not integrate well with specialized geometric planners, SMS can serve as a plug-in semantic module for downstream manipulation or navigation policies. For mechanical search in closed-world settings such as shelves, we compare with a geometric-based planner and show that SMS improves mechanical search performance by $24%$ across the pharmacy, kitchen, and office domains in simulation and $47.1%$ in physical experiments. For open-world real environments, SMS can produce better semantic distributions compared to CLIP-based methods, with the potential to be integrated with downstream navigation policies to improve object navigation tasks. Code, data, videos, and Appendix are available here.

**摘要:** 移动对象以寻找完全封闭的目标对象,称为机械搜索,是机器人中的一个挑战性问题。由于对象经常以语义方式组织,我们推测有关对象关系的语义信息可以促进机械搜索和减少搜索时间。大型预留视觉和语言模型(VLMs和LLMs)在一般化非常见对象和以前未见的现实环境方面显示了前景。与以前使用VLM和LLM为终点规划师的工作不同,这些可能与专门的几何规划师不兼容,SMS可以作为下游操作或导航政策的插件语义模块服务。 对于封闭世界设置中的机械搜索,如架子,我们与基于几何规划师比较,并显示SMS在模拟和物理实验中在药房、厨房和办公室领域提高机械搜索性能24%和47.1%。 对于开放世界真实环境,SMS可以与CLIP-based方法相比产生更好的语义分布,其潜力可以与下游导航政策结合,以改善对象导航任务。

**[Paper URL](https://proceedings.mlr.press/v229/sharma23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sharma23a/sharma23a.pdf)** 

# KITE: Keypoint-Conditioned Policies for Semantic Manipulation
**题目:** 库:语义操纵的关键条件政策

**作者:** Priya Sundaresan, Suneel Belkhale, Dorsa Sadigh, Jeannette Bohg

**Abstract:** While natural language offers a convenient shared interface for humans and robots, enabling robots to interpret and follow language commands remains a longstanding challenge in manipulation. A crucial step to realizing a performant instruction-following robot is achieving semantic manipulation – where a robot interprets language at different specificities, from high-level instructions like "Pick up the stuffed animal" to more detailed inputs like "Grab the left ear of the elephant." To tackle this, we propose Keypoints + Instructions to Execution, a two-step framework for semantic manipulation which attends to both scene semantics (distinguishing between different objects in a visual scene) and object semantics (precisely localizing different parts within an object instance). KITE first grounds an input instruction in a visual scene through 2D image keypoints, providing a highly accurate object-centric bias for downstream action inference. Provided an RGB-D scene observation, KITE then executes a learned keypoint-conditioned skill to carry out the instruction. The combined precision of keypoints and parameterized skills enables fine-grained manipulation with generalization to scene and object variations. Empirically, we demonstrate KITE in 3 real-world environments: long-horizon 6-DoF tabletop manipulation, semantic grasping, and a high-precision coffee-making task. In these settings, KITE achieves a $75%$, $70%$, and $71%$ overall success rate for instruction-following, respectively. KITE outperforms frameworks that opt for pre-trained visual language models over keypoint-based grounding, or omit skills in favor of end-to-end visuomotor control, all while being trained from fewer or comparable amounts of demonstrations. Supplementary material, datasets, code, and videos can be found on our website: https://tinyurl.com/kite-site.

**摘要:** 虽然自然语言为人类和机器人提供了方便的共享接口,但使机器人能够解释和遵循语言命令仍然是一个长期的操纵挑战。实现高性能的指令跟踪机器人的关键步骤是实现语义操纵 — — 机器人在不同的特定方面解释语言,从高级别的指令,如“拾起填充动物”到更详细的输入,如“抓住象的左耳朵”。提供RGB-D场景观察,KITE然后执行一个学习的键点条件技能执行指令。键点和参数化技能的综合精确性允许对场景和对象变异的一般化进行细粒化操作。我们以实例的方式在3个真实环境中展示KITE:长视线6-DoF平板操作、语义把握和高精度咖啡制作任务。在这些设置中,KITE分别达到$75%$、$70%$和$71%$的instruction-following总体成功率。KITE超越了从键点基础上选择预训练的视觉语言模型的框架,或者忽略了最终到终点视觉运动控制的技能,同时从较少或比较数量的演示中进行训练。

**[Paper URL](https://proceedings.mlr.press/v229/sundaresan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sundaresan23a/sundaresan23a.pdf)** 

# BM2CP: Efficient Collaborative Perception with LiDAR-Camera Modalities
**题目:** BM2CP:利达-摄像机模式下的高效协作感知

**作者:** Binyu Zhao, Wei ZHANG, Zhaonian Zou

**Abstract:** Collaborative perception enables agents to share complementary perceptual information with nearby agents. This can significantly benefit the perception performance and alleviate the issues of single-view perception, such as occlusion and sparsity. Most proposed approaches mainly focus on single modality (especially LiDAR), and not fully exploit the superiority of multi-modal perception. We propose an collaborative perception paradigm, BM2CP, which employs LiDAR and camera to achieve efficient multi-modal perception. BM2CP utilizes LiDAR-guided modal fusion, cooperative depth generation and modality-guided intermediate fusion to acquire deep interactions between modalities and agents. Moreover, it is capable to cope with the special case that one of the sensors is unavailable. Extensive experiments validate that it outperforms the state-of-the-art methods with 50X lower communication volumes in real-world autonomous driving scenarios. Our code is available at supplementary materials.

**摘要:** 合作感知使代理人能够与邻近代理人共享互补感知信息。这可以显著提高感知性能,缓解单视感知问题,例如闭塞和稀疏性。大多数提议的方法主要集中在单模式(特别是LiDAR)上,而不是充分利用多模式感知的优越性。我们提议一种合作感知范式,BM2CP,它使用LiDAR和摄像机实现高效多模式感知。BM2CP利用LiDAR引导的模态融合、合作深度生成和模态引导的中间融合来获取模态和代理之间的深度交互。

**[Paper URL](https://proceedings.mlr.press/v229/zhao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhao23a/zhao23a.pdf)** 

# That Sounds Right: Auditory Self-Supervision for Dynamic Robot Manipulation
**题目:** 听起来不错:动态机器人操纵的听觉自我监督

**作者:** Abitha Thankaraj, Lerrel Pinto

**Abstract:** Learning to produce contact-rich, dynamic behaviors from raw sensory data has been a longstanding challenge in robotics. Prominent approaches primarily focus on using visual and tactile sensing. However, pure vision often fails to capture high-frequency interaction, while current tactile sensors can be too delicate for large-scale data collection. In this work, we propose a data-centric approach to dynamic manipulation that uses an often ignored source of information – sound. We first collect a dataset of 25k interaction-sound pairs across five dynamic tasks using contact microphones. Then, given this data, we leverage self-supervised learning to accelerate behavior prediction from sound. Our experiments indicate that this self-supervised ‘pretraining’ is crucial to achieving high performance, with a $34.5%$ lower MSE than plain supervised learning and a $54.3%$ lower MSE over visual training. Importantly, we find that when asked to generate desired sound profiles, online rollouts of our models on a UR10 robot can produce dynamic behavior that achieves an average of $11.5%$ improvement over supervised learning on audio similarity metrics. Videos and audio data are best seen on our project website: aurl-anon.github.io

**摘要:** 学习从原始感官数据产生接触丰富、动态行为是机器人领域长期以来的一个挑战。主要的途径是使用视觉和触觉感知。然而,纯视觉往往无法捕捉高频交互,而当前触觉感知器对于大规模数据收集来说过于敏感。在这个工作中,我们提出了一种数据中心的动态操作方法,它使用了一个经常忽略的信息源——声音。我们首先收集了5个动态任务中25k交互声对的数据集,使用触觉麦克风。然后,根据这些数据,我们利用自监督学习来加速从声音中预测行为。重要的是,我们发现当被要求生成所需的音频 профил时,在UR10机器人上在线推出我们的模型能够产生动态行为,在音频相似度指标上的监督学习上达到平均$11.5%的改善。视频和音频数据最好看到我们的项目网站: aurl-anon.github.io

**[Paper URL](https://proceedings.mlr.press/v229/thankaraj23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/thankaraj23a/thankaraj23a.pdf)** 

# ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting
**题目:** ManiCast:基于成本意识的人类预测的协作操纵

**作者:** Kushal Kedia, Prithwish Dan, Atiksh Bhardwaj, Sanjiban Choudhury

**Abstract:** Seamless human-robot manipulation in close proximity relies on accurate forecasts of human motion. While there has been significant progress in learning forecast models at scale, when applied to manipulation tasks, these models accrue high errors at critical transition points leading to degradation in downstream planning performance. Our key insight is that instead of predicting the most likely human motion, it is sufficient to produce forecasts that capture how future human motion would affect the cost of a robot’s plan. We present ManiCast, a novel framework that learns cost-aware human forecasts and feeds them to a model predictive control planner to execute collaborative manipulation tasks. Our framework enables fluid, real-time interactions between a human and a 7-DoF robot arm across a number of real-world tasks such as reactive stirring, object handovers, and collaborative table setting. We evaluate both the motion forecasts and the end-to-end forecaster-planner system against a range of learned and heuristic baselines while additionally contributing new datasets. We release our code and datasets at https://portal-cornell.github.io/manicast/.

**摘要:** 近距离的无缝人-机器人操纵依赖于人类运动的准确预测。虽然在大规模学习预测模型方面取得了重大进展,但当应用于操纵任务时,这些模型在关键的过渡点产生高误差,导致下游规划性能的恶化。我们的关键洞察是,而不是预测人类运动的最可能,它足以产生预测,捕捉未来人类运动会如何影响机器人计划的成本。我们对运动预测和端到端的预测者-规划者系统进行评估,同时对各种学习和启发性基础线进行评估,并提供新的数据集。我们在 https://portal-cornell.github.io/manicast/上发布我们的代码和数据集。

**[Paper URL](https://proceedings.mlr.press/v229/kedia23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kedia23a/kedia23a.pdf)** 

# Predicting Routine Object Usage for Proactive Robot Assistance
**题目:** 预测主动机器人辅助的常规对象使用

**作者:** Maithili Patel, Aswin Gururaj Prakash, Sonia Chernova

**Abstract:** Proactivity in robot assistance refers to the robot’s ability to anticipate user needs and perform assistive actions without explicit requests. This requires understanding user routines, predicting consistent activities, and actively seeking information to predict inconsistent behaviors. We propose SLaTe-PRO (Sequential Latent Temporal model for Predicting Routine Object usage), which improves upon prior state-of-the-art by combining object and user action information, and conditioning object usage predictions on past history. Additionally, we find some human behavior to be inherently stochastic and lacking in contextual cues that the robot can use for proactive assistance. To address such cases, we introduce an interactive query mechanism that can be used to ask queries about the user’s intended activities and object use to improve prediction. We evaluate our approach on longitudinal data from three households, spanning 24 activity classes. SLaTe-PRO performance raises the F1 score metric to 0.57 without queries, and 0.60 with user queries, over a score of 0.43 from prior work. We additionally present a case study with a fully autonomous household robot.

**摘要:** 在机器人辅助中,主动性是指机器人能够预知用户需要并无明确要求执行辅助行动的能力。这需要理解用户日常活动,预测一致活动,并积极寻找信息来预测不一致的行为。我们提出了 SLaTe-PRO(Sequential Latent Temporal model for Predicting Routine Object usage),它通过将对象和用户动作信息结合起来,在过去的历史中改善了对象使用预测。此外,我们发现某些人的行为是固有随机性的,缺乏机器人可以用于主动性援助的上下文提示。SLaTe-PRO的性能提高了F1分数计量值到0.57,没有查询,和0.60的用户查询,超过前作的0.43分。此外,我们还提出了一个完全自主的家庭机器人的案例研究。

**[Paper URL](https://proceedings.mlr.press/v229/patel23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/patel23a/patel23a.pdf)** 

# Grounding Complex Natural Language Commands for Temporal Tasks in Unseen Environments
**题目:** 无视环境中临时任务的复杂自然语言命令

**作者:** Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex, Ankit Shah

**Abstract:** Grounding navigational commands to linear temporal logic (LTL) leverages its unambiguous semantics for reasoning about long-horizon tasks and verifying the satisfaction of temporal constraints. Existing approaches require training data from the specific environment and landmarks that will be used in natural language to understand commands in those environments. We propose Lang2LTL, a modular system and a software package that leverages large language models (LLMs) to ground temporal navigational commands to LTL specifications in environments without prior language data. We comprehensively evaluate Lang2LTL for five well-defined generalization behaviors. Lang2LTL demonstrates the state-of-the-art ability of a single model to ground navigational commands to diverse temporal specifications in 21 city-scaled environments. Finally, we demonstrate a physical robot using Lang2LTL can follow 52 semantically diverse navigational commands in two indoor environments.

**摘要:** 基于线性时间逻辑(LTL)的地面导航命令利用其明确的语义来推理长期目标任务和验证时间限制的满足。现有的方法需要从特定环境和标记中获取训练数据,这些数据将在自然语言中用于理解这些环境中的命令。我们提出了一种模块化系统和软件包,它利用大型语言模型(LLMs)实现在没有语言数据的情况下在地面时间导航命令的LTL规范。我们全面评估了 Lang2LTL的五个明确的一般化行为。

**[Paper URL](https://proceedings.mlr.press/v229/liu23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23d/liu23d.pdf)** 

# HOI4ABOT: Human-Object Interaction Anticipation for Human Intention Reading Collaborative roBOTs
**题目:** HOI4ABOT:人类-对象相互作用预测人类意向阅读合作机器人

**作者:** Esteve Valls Mascaro, Daniel Sliwowski, Dongheui Lee

**Abstract:** Robots are becoming increasingly integrated into our lives, assisting us in various tasks. To ensure effective collaboration between humans and robots, it is essential that they understand our intentions and anticipate our actions. In this paper, we propose a Human-Object Interaction (HOI) anticipation framework for collaborative robots. We propose an efficient and robust transformer-based model to detect and anticipate HOIs from videos. This enhanced anticipation empowers robots to proactively assist humans, resulting in more efficient and intuitive collaborations. Our model outperforms state-of-the-art results in HOI detection and anticipation in VidHOI dataset with an increase of $1.76%$ and $1.04%$ in mAP respectively while being 15.4 times faster. We showcase the effectiveness of our approach through experimental results in a real robot, demonstrating that the robot’s ability to anticipate HOIs is key for better Human-Robot Interaction.

**摘要:** 为了确保人与机器人之间的有效协作,他们必须了解我们的意图和预期我们的行动。本论文中,我们为合作机器人提出了一种人-目标交互(HOI)预期框架。我们提出了一种基于变换器的高效和鲁棒模型,以检测和预测视频中的HOI。这种增强的预期授权机器人主动协助人类,从而产生更高效和直观的协作。我们的模型在VidHOI数据集中HOI检测和预期的最先进的结果,分别达到$1.76%和$1.04%,同时达到15.4倍的速度。我们通过实物机器人的实验结果展示了我们的方法的有效性,证明了机器人预测HOI的能力是更好的人-机器人交互的关键。

**[Paper URL](https://proceedings.mlr.press/v229/mascaro23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mascaro23a/mascaro23a.pdf)** 

# Reinforcement Learning Enables Real-Time Planning and Control of Agile Maneuvers for Soft Robot Arms
**题目:** 增强学习能实现软机器人武器敏捷手柄的实时规划和控制

**作者:** Rianna Jitosho, Tyler Ga Wei Lum, Allison Okamura, Karen Liu

**Abstract:** Control policies for soft robot arms typically assume quasi-static motion or require a hand-designed motion plan. To achieve real-time planning and control for tasks requiring highly dynamic maneuvers, we apply deep reinforcement learning to train a policy entirely in simulation, and we identify strategies and insights that bridge the gap between simulation and reality. In particular, we strengthen the policy’s tolerance for inaccuracies with domain randomization and implement crucial simulator modifications that improve actuation and sensor modeling, enabling zero-shot sim-to-real transfer without requiring high-fidelity soft robot dynamics. We demonstrate the effectiveness of this approach with experiments on physical hardware and show that our soft robot can reach target positions that require dynamic swinging motions. This is the first work to achieve such agile maneuvers on a physical soft robot, advancing the field of soft robot arm planning and control. Our code and videos are publicly available at https://sites.google.com/view/rl-soft-robot.

**摘要:** 软机器人手臂的控制策略通常采用准静态运动或需要手设计的运动计划。为了实现需要高度动态操纵的任务的实时规划和控制,我们应用深层强化学习来完全在模拟中训练政策,并确定了解决模拟与现实之间的差距的策略和洞察力。我们的代码和视频在 https://sites.google.com/view/rl-soft-robot上公开。

**[Paper URL](https://proceedings.mlr.press/v229/jitosho23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/jitosho23a/jitosho23a.pdf)** 

# A Policy Optimization Method Towards Optimal-time Stability
**题目:** 政策优化方法,实现最佳时间稳定

**作者:** Shengjie Wang, Lan Fengb, Xiang Zheng, Yuxue Cao, Oluwatosin OluwaPelumi Oseni, Haotian Xu, Tao Zhang, Yang Gao

**Abstract:** In current model-free reinforcement learning (RL) algorithms, stability criteria based on sampling methods are commonly utilized to guide policy optimization. However, these criteria only guarantee the infinite-time convergence of the system’s state to an equilibrium point, which leads to sub-optimality of the policy. In this paper, we propose a policy optimization technique incorporating sampling-based Lyapunov stability. Our approach enables the system’s state to reach an equilibrium point within an optimal time and maintain stability thereafter, referred to as "optimal-time stability". To achieve this, we integrate the optimization method into the Actor-Critic framework, resulting in the development of the Adaptive Lyapunov-based Actor-Critic (ALAC) algorithm. Through evaluations conducted on ten robotic tasks, our approach outperforms previous studies significantly, effectively guiding the system to generate stable patterns.

**摘要:** 在当前的无模型增强学习(RL)算法中,基于抽样方法的稳定性标准通常用于指导政策优化。然而,这些标准只保证系统状态的无限时间收敛到平衡点,从而导致政策的次优性。本文提出了一种基于抽样的拉普诺夫稳定性的策略优化技术。我们的方法使系统状态在最佳时间内达到平衡点,并在此后保持稳定,称为“最佳时间稳定性”。为此,我们将优化方法集成到行为者-行为者框架中,从而开发了基于拉普诺夫的适应性行为者-行为者(ALAC)算法。

**[Paper URL](https://proceedings.mlr.press/v229/wang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23d/wang23d.pdf)** 

# An Unbiased Look at Datasets for Visuo-Motor Pre-Training
**题目:** 无偏见的视力运动预训练数据集

**作者:** Sudeep Dasari, Mohan Kumar Srirama, Unnat Jain, Abhinav Gupta

**Abstract:** Visual representation learning hold great promise for robotics, but is severely hampered by the scarcity and homogeneity of robotics datasets. Recent works address this problem by pre-training visual representations on large-scale but out-of-domain data (e.g., videos of egocentric interactions) and then transferring them to target robotics tasks. While the field is heavily focused on developing better pre-training algorithms, we find that dataset choice is just as important to this paradigm’s success. After all, the representation can only learn the structures or priors present in the pre-training dataset. To this end, we flip the focus on algorithms, and instead conduct a dataset centric analysis of robotic pre-training. Our findings call into question some common wisdom in the field. We observe that traditional vision datasets (like ImageNet, Kinetics and 100 Days of Hands) are surprisingly competitive options for visuo-motor representation learning, and that the pre-training dataset’s image distribution matters more than its size. Finally, we show that common simulation benchmarks are not a reliable proxy for real world performance and that simple regularization strategies can dramatically improve real world policy learning.

**摘要:** 视觉表现学习对于机器人具有巨大的前景,但由于机器人数据集的稀缺和均匀性严重阻碍。最近的工作解决了这一问题,通过在大规模但外域数据(例如自我中心交互视频)上进行预训练视觉表现,然后将其转移到机器人目标任务。虽然该领域着重发展更好的预训练算法,但我们发现数据集选择对于这一模式的成功同样重要。我们观察到传统的视觉数据集(如ImageNet、Kinetics和100 Days of Hands)对于视觉运动表现学习具有惊人的竞争性,并且预训练数据集的图像分布比其大小更重要。

**[Paper URL](https://proceedings.mlr.press/v229/dasari23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/dasari23a/dasari23a.pdf)** 

# Equivariant Motion Manifold Primitives
**题目:** 等价运动奇异 Primitives

**作者:** Byeongho Lee, Yonghyeon Lee, Seungyeon Kim, MinJun Son, Frank C. Park

**Abstract:** Existing movement primitive models for the most part focus on representing and generating a single trajectory for a given task, limiting their adaptability to situations in which unforeseen obstacles or new constraints may arise. In this work we propose Motion Manifold Primitives (MMP), a movement primitive paradigm that encodes and generates, for a given task, a continuous manifold of trajectories each of which can achieve the given task. To address the challenge of learning each motion manifold from a limited amount of data, we exploit inherent symmetries in the robot task by constructing motion manifold primitives that are equivariant with respect to given symmetry groups. Under the assumption that each of the MMPs can be smoothly deformed into each other, an autoencoder framework is developed to encode the MMPs and also generate solution trajectories. Experiments involving synthetic and real-robot examples demonstrate that our method outperforms existing manifold primitive methods by significant margins. Code is available at https://github.com/dlsfldl/EMMP-public.

**摘要:** 现有的运动原型模型主要集中在给定任务的一个单轨迹的表示和生成上,限制其适应性到可能出现的未知障碍或新约束的情况。本研究中,我们提出了一种运动原型范式Motion Manifold Primitives(MMP),该范式为给定任务编码并生成一个连续的多轨迹,其中每个可以实现给定任务。为了解决从有限的数据中学习每个运动多轨迹的挑战,我们利用机器人任务中的固有对称性,通过构造与给定对称群相等的运动多轨迹原型。包含合成和实际机器人例子的实验表明,我们的方法比现有的多种原始方法有显著的差距。代码可于 https://github.com/dlsfldl/EMMP-public。

**[Paper URL](https://proceedings.mlr.press/v229/lee23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lee23a/lee23a.pdf)** 

# FlowBot++: Learning Generalized Articulated Objects Manipulation via Articulation Projection
**题目:** FlowBot++: 通过线性投影学习一般化的线性对象操作

**作者:** Harry Zhang, Ben Eisner, David Held

**Abstract:** Understanding and manipulating articulated objects, such as doors and drawers, is crucial for robots operating in human environments. We wish to develop a system that can learn to articulate novel objects with no prior interaction, after training on other articulated objects. Previous approaches for articulated object manipulation rely on either modular methods which are brittle or end-to-end methods, which lack generalizability. This paper presents FlowBot++, a deep 3D vision-based robotic system that predicts dense per-point motion and dense articulation parameters of articulated objects to assist in downstream manipulation tasks. FlowBot++ introduces a novel per-point representation of the articulated motion and articulation parameters that are combined to produce a more accurate estimate than either method on their own. Simulated experiments on the PartNet-Mobility dataset validate the performance of our system in articulating a wide range of objects, while real-world experiments on real objects’ point clouds and a Sawyer robot demonstrate the generalizability and feasibility of our system in real-world scenarios. Videos are available on our anonymized website https://sites.google.com/view/flowbotpp/home

**摘要:** 本文介绍了一种基于深度3D视觉的机器人系统FlowBot++,该系统可预测单点运动和单点运动参数,以协助下游操作任务。FlowBot++引入了单点运动和单点参数的单点表示,结合起来可以比单点方法更准确的估计。PartNet-Mobility数据集的仿真实验验证了我们系统在表达多种对象方面的表现,而实物对象点云和S Sawyer机器人的实物实验则证明了我们系统在实物场景中具有通用性和可行性。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23c/zhang23c.pdf)** 

# Geometry Matching for Multi-Embodiment Grasping
**题目:** 多体格图形的几何匹配

**作者:** Maria Attarian, Muhammad Adil Asif, Jingzhou Liu, Ruthrash Hari, Animesh Garg, Igor Gilitschenski, Jonathan Tompson

**Abstract:** While significant progress has been made on the problem of generating grasps, many existing learning-based approaches still concentrate on a single embodiment, provide limited generalization to higher DoF end-effectors and cannot capture a diverse set of grasp modes. In this paper, we tackle the problem of grasping multi-embodiments through the viewpoint of learning rich geometric representations for both objects and end-effectors using Graph Neural Networks (GNN). Our novel method – GeoMatch – applies supervised learning on grasping data from multiple embodiments, learning end-to-end contact point likelihood maps as well as conditional autoregressive prediction of grasps keypoint-by-keypoint. We compare our method against 3 baselines that provide multi-embodiment support. Our approach performs better across 3 end-effectors, while also providing competitive diversity of grasps. Examples can be found at geomatch.github.io.

**摘要:** 虽然在生成把握问题上取得了重大进展,但许多现有的基于学习的方法仍然集中于单一的把握,对较高的DoFend-effectors提供有限的一般化,并不能捕捉多种把握模式。本文,我们通过学习 Graph Neural Networks(GNN)的对象和end-effectors rich geometric representations的视角来解决把握多姿态的问题。我们的新方法--GeoMatch--应用于从多姿态把握数据的监督学习,学习end-to-end接触点概率地图以及条件自回归预测 grasps keypoint-by-keypoint。我们与提供多姿态支持的3个基线相比,我们的方法在3个end-effectors中的表现更好,同时提供竞争性的把握多样性。

**[Paper URL](https://proceedings.mlr.press/v229/attarian23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/attarian23a/attarian23a.pdf)** 

# Contrastive Value Learning: Implicit Models for Simple Offline RL
**题目:** 反向价值学习:简单的非线性RL的隐形模型

**作者:** Bogdan Mazoure, Benjamin Eysenbach, Ofir Nachum, Jonathan Tompson

**Abstract:** Model-based reinforcement learning (RL) methods are appealing in the offline setting because they allow an agent to reason about the consequences of actions without interacting with the environment. While conventional model-based methods learn a 1-step model, predicting the immediate next state, these methods must be plugged into larger planning or RL systems to yield a policy. Can we model the environment dynamics in a different way, such that the learned model directly indicates the value of each action? In this paper, we propose Contrastive Value Learning (CVL), which learns an implicit, multi-step dynamics model. This model can be learned without access to reward functions, but nonetheless can be used to directly estimate the value of each action, without requiring any TD learning. Because this model represents the multi-step transitions implicitly, it avoids having to predict high-dimensional observations and thus scales to high-dimensional tasks. Our experiments demonstrate that CVL outperforms prior offline RL methods on complex robotics benchmarks.

**摘要:** 基于模型的强化学习(RL)方法在非线性设置中具有吸引力,因为它们允许代理人与环境无关地推理行为的后果。传统的基于模型的方法学习一步模型,预测下一个状态,这些方法必须被插入更大的规划或RL系统,以产生政策。我们能以不同的方式模拟环境动力学,使得学习模型直接显示每个行为的价值吗?本论文提出了反向价值学习(CVL),它学习了一个隐含的多步动力学模型。该模型可以不访问奖励函数,但仍可用于直接估计每个行为的价值,而不需要任何TD学习。我们的实验表明,CVL在复杂的机器人基准上比以前的非线性RL方法高。

**[Paper URL](https://proceedings.mlr.press/v229/mazoure23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mazoure23a/mazoure23a.pdf)** 

# Parting with Misconceptions about Learning-based Vehicle Motion Planning
**题目:** 与基于学习的车辆运动规划的误解分开

**作者:** Daniel Dauner, Marcel Hallgarten, Andreas Geiger, Kashyap Chitta

**Abstract:** The release of nuPlan marks a new era in vehicle motion planning research, offering the first large-scale real-world dataset and evaluation schemes requiring both precise short-term planning and long-horizon ego-forecasting. Existing systems struggle to simultaneously meet both requirements. Indeed, we find that these tasks are fundamentally misaligned and should be addressed independently. We further assess the current state of closed-loop planning in the field, revealing the limitations of learning-based methods in complex real-world scenarios and the value of simple rule-based priors such as centerline selection through lane graph search algorithms. More surprisingly, for the open-loop sub-task, we observe that the best results are achieved when using only this centerline as scene context (i.e., ignoring all information regarding the map and other agents). Combining these insights, we propose an extremely simple and efficient planner which outperforms an extensive set of competitors, winning the nuPlan planning challenge 2023.

**摘要:** nuPlan的发布标志着汽车运动规划研究的新纪元,提供了第一个大规模的现实世界数据集和评估方案,需要准确的短期规划和长期自我预测。现有系统同时满足这两个要求,事实上,我们发现这些任务基本不一致,应该独立处理。我们进一步评估了现场封闭循环规划的现状,揭示了复杂现实世界场景中基于学习的方法的局限性以及通过路面图搜索算法选择中心线等简单的规则基础的优先次序的价值。结合这些洞察力,我们提出了一个非常简单的高效的规划师,它超过了大量的竞争对手,赢得nuPlan规划挑战2023。

**[Paper URL](https://proceedings.mlr.press/v229/dauner23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/dauner23a/dauner23a.pdf)** 

# Learning Sequential Acquisition Policies for Robot-Assisted Feeding
**题目:** 机器人辅助喂养学习序列获取策略

**作者:** Priya Sundaresan, Jiajun Wu, Dorsa Sadigh

**Abstract:** A robot providing mealtime assistance must perform specialized maneuvers with various utensils in order to pick up and feed a range of food items. Beyond these dexterous low-level skills, an assistive robot must also plan these strategies in sequence over a long horizon to clear a plate and complete a meal. Previous methods in robot-assisted feeding introduce highly specialized primitives for food handling without a means to compose them together. Meanwhile, existing approaches to long-horizon manipulation lack the flexibility to embed highly specialized primitives into their frameworks. We propose Visual Action Planning OveR Sequences (VAPORS), a framework for long-horizon food acquisition. VAPORS learns a policy for high-level action selection by leveraging learned latent plate dynamics in simulation. To carry out sequential plans in the real world, VAPORS delegates action execution to visually parameterized primitives. We validate our approach on complex real-world acquisition trials involving noodle acquisition and bimanual scooping of jelly beans. Across 38 plates, VAPORS acquires much more efficiently than baselines, generalizes across realistic plate variations such as toppings and sauces, and qualitatively appeals to user feeding preferences in a survey conducted across 49 individuals. Code, datasets, videos, and supplementary materials can be found on our website: https://sites.google.com/view/vaporsbot.

**摘要:** 一个提供膳食时间援助的机器人必须用各种器具进行专门的操纵,以获取和喂食一系列食物物品。除了这些熟练的低级技能外,辅助机器人还必须在长期范围内规划这些策略来清理一个盘子并完成膳食。以前的机器人辅助喂食方法引入高度专门的原始食品处理方法,没有办法将它们组合起来。与此同时,现有的长期操作方法缺乏在其框架内嵌入高度专门的原始食品的灵活性。我们对涉及面条收购和双人咀嚼的复杂真实世界收购试验的办法进行了验证。在38个盘子中, VAPORS比基线取得的效率高得多,在49个个人进行的调查中,对实际的盘子变异,如 toppings 和 sauces, 以及对用户喂食的偏好具有定性的吸引力。代码、数据集、视频和补充材料可以找到我们的网站: https://sites.google.com/view/vaporsbot。

**[Paper URL](https://proceedings.mlr.press/v229/sundaresan23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sundaresan23b/sundaresan23b.pdf)** 

# Composable Part-Based Manipulation
**题目:** 可编译的基于部件操作

**作者:** Weiyu Liu, Jiayuan Mao, Joy Hsu, Tucker Hermans, Animesh Garg, Jiajun Wu

**Abstract:** In this paper, we propose composable part-based manipulation (CPM), a novel approach that leverages object-part decomposition and part-part correspondences to improve learning and generalization of robotic manipulation skills. By considering the functional correspondences between object parts, we conceptualize functional actions, such as pouring and constrained placing, as combinations of different correspondence constraints. CPM comprises a collection of composable diffusion models, where each model captures a different inter-object correspondence. These diffusion models can generate parameters for manipulation skills based on the specific object parts. Leveraging part-based correspondences coupled with the task decomposition into distinct constraints enables strong generalization to novel objects and object categories. We validate our approach in both simulated and real-world scenarios, demonstrating its effectiveness in achieving robust and generalized manipulation capabilities.

**摘要:** 本文提出了一种利用对象部件分解和部分部件对应的方法,改进机器人操作技能的学习和推广。通过考虑对象部件间的函数对应,我们将函数行为,如注水和约束置位,作为不同对应约束的结合概念化。CPM包括一套可编译扩散模型,每个模型捕捉不同的对象间的对应关系。这些扩散模型可以根据特定对象部件生成操作技能的参数。

**[Paper URL](https://proceedings.mlr.press/v229/liu23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23e/liu23e.pdf)** 

# Adv3D: Generating Safety-Critical 3D Objects through Closed-Loop Simulation
**题目:** Adv3D:通过闭路模拟生成安全-临界3D对象

**作者:** Jay Sarva, Jingkang Wang, James Tu, Yuwen Xiong, Sivabalan Manivasagam, Raquel Urtasun

**Abstract:** Self-driving vehicles (SDVs) must be rigorously tested on a wide range of scenarios to ensure safe deployment. The industry typically relies on closed-loop simulation to evaluate how the SDV interacts on a corpus of synthetic and real scenarios and to verify good performance. However, they primarily only test the motion planning module of the system, and only consider behavior variations. It is key to evaluate the full autonomy system in closed-loop, and to understand how variations in sensor data based on scene appearance, such as the shape of actors, affect system performance. In this paper, we propose a framework, Adv3D, that takes real world scenarios and performs closed-loop sensor simulation to evaluate autonomy performance, and finds vehicle shapes that make the scenario more challenging, resulting in autonomy failures and uncomfortable SDV maneuvers. Unlike prior work that add contrived adversarial shapes to vehicle roof-tops or roadside to harm perception performance, we optimize a low-dimensional shape representation to modify the vehicle shape itself in a realistic manner to degrade full autonomy performance (e.g., perception, prediction, motion planning). Moreover, we find that the shape variations found with Adv3D optimized in closed-loop are much more effective than open-loop, demonstrating the importance of finding and testing scene appearance variations that affect full autonomy performance.

**摘要:** 自走车辆(SDV)必须在广泛的场景中进行严格测试,以确保安全部署。工业通常依靠闭环模拟来评估SDV在合成和实际场景中如何相互作用,并验证良好的性能。然而,它们主要只测试系统运动规划模块,并只考虑行为变化。与以往在车辆屋顶或路面上添加拟合的敌对形状,以损害视觉性能不同,我们优化了低维形状表示,以使车辆形状本身在现实的方式修改,从而降低全自动性能(例如视觉、预测、运动规划)。此外,我们发现在封闭循环中优化的Adv3D的形状变异比开放循环更加有效,证明发现和测试影响全自动性能的场景外观变异的重要性。

**[Paper URL](https://proceedings.mlr.press/v229/sarva23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sarva23a/sarva23a.pdf)** 

# FindThis: Language-Driven Object Disambiguation in Indoor Environments
**题目:** FindThis:室内环境中语言驱动对象的区分

**作者:** Arjun Majumdar, Fei Xia, Brian Ichter, Dhruv Batra, Leonidas Guibas

**Abstract:** Natural language is naturally ambiguous. In this work, we consider interactions between a user and a mobile service robot tasked with locating a desired object, specified by a language utterance. We present a task FindThis, which addresses the problem of how to disambiguate and locate the particular object instance desired through a dialog with the user. To approach this problem we propose an algorithm, GoFind, which exploits visual attributes of the object that may be intrinsic (e.g., color, shape), or extrinsic (e.g., location, relationships to other entities), expressed in an open vocabulary. GoFind leverages the visual common sense learned by large language models to enable fine-grained object localization and attribute differentiation in a zero-shot manner. We also provide a new visio-linguistic dataset, 3D Objects in Context (3DOC), for evaluating agents on this task consisting of Google Scanned Objects placed in Habitat-Matterport 3D scenes. Finally, we validate our approach on a real robot operating in an unstructured physical office environment using complex fine-grained language instructions.

**摘要:** 自然语言是自然的含糊不清的。在这个工作中,我们考虑了用户与移动服务机器人之间的交互作用,任务是定位一个需要的对象,由语言表达来指定。我们提出了FindThis任务,该任务解决如何通过与用户对话来区分和定位特定对象实例的问题。为了解决这个问题,我们提出了GoFind算法,该算法利用对象的视觉属性,这些属性可能是内在的(例如颜色、形状)或外在的(例如位置、与其他实体的关系),表达在公开的词汇中。GoFind利用由大型语言模型学习的视觉常识,以使精细的对象定位和属性分化以零射击的方式。最后,我们验证了在非结构的办公室环境中使用复杂的精细语言指令的实际机器人的处理方法。

**[Paper URL](https://proceedings.mlr.press/v229/majumdar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/majumdar23a/majumdar23a.pdf)** 

# Action-Quantized Offline Reinforcement Learning for Robotic Skill Learning
**题目:** 机器人技能学习的行动量化 offline增强学习

**作者:** Jianlan Luo, Perry Dong, Jeffrey Wu, Aviral Kumar, Xinyang Geng, Sergey Levine

**Abstract:** The offline reinforcement learning (RL) paradigm provides a general recipe to convert static behavior datasets into policies that can perform better than the policy that collected the data. While policy constraints, conservatism, and other methods for mitigating distributional shifts have made offline reinforcement learning more effective, the continuous action setting often necessitates various approximations for applying these techniques. Many of these challenges are greatly alleviated in discrete action settings, where offline RL constraints and regularizers can often be computed more precisely or even exactly. In this paper, we propose an adaptive scheme for action quantization. We use a VQ-VAE to learn state- conditioned action quantization, avoiding the exponential blowup that comes with naïve discretization of the action space. We show that several state-of-the-art offline RL methods such as IQL, CQL, and BRAC improve in performance on benchmarks when combined with our proposed discretization scheme. We further validate our approach on a set of challenging long-horizon complex robotic manipulation tasks in the Robomimic environment, where our discretized offline RL algorithms are able to improve upon their continuous counterparts by 2-3x. Our project page is at saqrl.github.io

**摘要:** 网络增强学习(英语:Offline reinforcement learning,简称RL)是将静态行为数据集转换成能比收集数据的政策更好的政策的一种一般方法。虽然政策约束、保守主义和其他方法来缓解分布性转变使得网络增强学习更加有效,但持续行动设置经常需要对这些技术进行各种近似。我们进一步验证了我们提出的离散化方案在罗博米mic环境中一系列具有挑战性的长期复杂机器人操纵任务中的方法,其中离散化的离散RL算法能够提高其连续的同类算法的 2-3倍。

**[Paper URL](https://proceedings.mlr.press/v229/luo23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/luo23a/luo23a.pdf)** 

# Batch Differentiable Pose Refinement for In-The-Wild Camera/LiDAR Extrinsic Calibration
**题目:** 野外摄像机/LiDAR外型校正批量可辨别的姿势精细

**作者:** Lanke Frank Tarimo Fu, Maurice Fallon

**Abstract:** Accurate camera to LiDAR (Light Detection and Ranging) extrinsic calibration is important for robotic tasks carrying out tight sensor fusion — such as target tracking and odometry. Calibration is typically performed before deployment in controlled conditions using calibration targets, however, this limits scalability and subsequent recalibration. We propose a novel approach for target-free camera-LiDAR calibration using end-to-end direct alignment which doesn’t need calibration targets. Our batched formulation enhances sample efficiency during training and robustness at inference time. We present experimental results, on publicly available real-world data, demonstrating 1.6cm/$0.07^{\circ}$ median accuracy when transferred to unseen sensors from held-out data sequences. We also show state-of-the-art zero-shot transfer to unseen cameras, LiDARs, and environments.

**摘要:** 对LiDAR(光检测和测距)外部校正的精确摄像机对于执行紧密传感器融合的机器人任务至关重要 — — 如目标跟踪和仪表学。校正通常在使用校正目标在控制条件下部署之前进行,然而,这限制了可扩展性和随后的重新校正。我们提出了一种新的目标自由摄像机-LiDAR校正方法,使用端到端直接校正,不需要校正目标。

**[Paper URL](https://proceedings.mlr.press/v229/fu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/fu23a/fu23a.pdf)** 

# Fleet Active Learning: A Submodular Maximization Approach
**题目:** 舰队主动学习:一种子模块化最大化方法

**作者:** Oguzhan Akcin, Orhan Unuvar, Onat Ure, Sandeep P. Chinchali

**Abstract:** In multi-robot systems, robots often gather data to improve the performance of their deep neural networks (DNNs) for perception and planning. Ideally, these robots should select the most informative samples from their local data distributions by employing active learning approaches. However, when the data collection is distributed among multiple robots, redundancy becomes an issue as different robots may select similar data points. To overcome this challenge, we propose a fleet active learning (FAL) framework in which robots collectively select informative data samples to enhance their DNN models. Our framework leverages submodular maximization techniques to prioritize the selection of samples with high information gain. Through an iterative algorithm, the robots coordinate their efforts to collectively select the most valuable samples while minimizing communication between robots. We provide a theoretical analysis of the performance of our proposed framework and show that it is able to approximate the NP-hard optimal solution. We demonstrate the effectiveness of our framework through experiments on real-world perception and classification datasets, which include autonomous driving datasets such as Berkeley DeepDrive. Our results show an improvement by up to $25.0 %$ in classification accuracy, $9.2 %$ in mean average precision and $48.5 %$ in the submodular objective value compared to a completely distributed baseline.

**摘要:** 在多机器人系统中,机器人经常收集数据以提高其深度神经网络(DNN)的感知和规划性能。理想情况下,这些机器人应通过采用主动学习方法从其本地数据分布中选择最有信息的样品。然而,当数据收集在多个机器人之间分布时,冗余成为问题,因为不同的机器人可能选择相似的数据点。为了克服这一挑战,我们提出了一种舰队主动学习(FAL)框架,在该框架中,机器人集体选择有信息的数据样品以增强其DNN模型。我们提供了我们提出的框架的性能的理论分析,并证明它能够近似NP-hard最佳解决方案。我们通过对真实世界感知和分类数据集的实验证明了我们的框架的有效性,包括Berkeley DeepDrive等自主驱动数据集。我们的结果显示了分类精度的提高高达$25.0%,平均精度的提高$9.2%,副模态目标值的提高$48.5,与完全分布的基线相比。

**[Paper URL](https://proceedings.mlr.press/v229/akcin23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/akcin23a/akcin23a.pdf)** 

# Robust Reinforcement Learning in Continuous Control Tasks with Uncertainty Set Regularization
**题目:** 基于不确定性的持续控制任务的鲁棒强化学习

**作者:** Yuan Zhang, Jianhong Wang, Joschka Boedecker

**Abstract:** Reinforcement learning (RL) is recognized as lacking generalization and robustness under environmental perturbations, which excessively restricts its application for real-world robotics. Prior work claimed that adding regularization to the value function is equivalent to learning a robust policy under uncertain transitions. Although the regularization-robustness transformation is appealing for its simplicity and efficiency, it is still lacking in continuous control tasks. In this paper, we propose a new regularizer named Uncertainty Set Regularizer (USR), to formulate the uncertainty set on the parametric space of a transition function. To deal with unknown uncertainty sets, we further propose a novel adversarial approach to generate them based on the value function. We evaluate USR on the Real-world Reinforcement Learning (RWRL) benchmark and the Unitree A1 Robot, demonstrating improvements in the robust performance of perturbed testing environments and sim-to-real scenarios.

**摘要:** 强化学习(英语:Reinforcement learning,简称RL)被认为是在环境扰动下缺乏一般化和鲁棒性,这极大地限制了其应用于真实机器人领域。以前的研究声称将规则化添加到值函数,相当于在不确定的转变下学习鲁棒政策。虽然规则化-鲁棒性转换具有简单性和效率,但仍缺乏连续控制任务。本论文提出了一种名为“不确定集规则化器”的新规则化器,以制订一个变换函数参数空间上的不确定集。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23d/zhang23d.pdf)** 

# Context-Aware Deep Reinforcement Learning for Autonomous Robotic Navigation in Unknown Area
**题目:** 无知区域自主机器人导航的深层次强化学习

**作者:** Jingsong Liang, Zhichen Wang, Yuhong Cao, Jimmy Chiun, Mengqi Zhang, Guillaume Adrien Sartoretti

**Abstract:** Mapless navigation refers to a challenging task where a mobile robot must rapidly navigate to a predefined destination using its partial knowledge of the environment, which is updated online along the way, instead of a prior map of the environment. Inspired by the recent developments in deep reinforcement learning (DRL), we propose a learning-based framework for mapless navigation, which employs a context-aware policy network to achieve efficient decision-making (i.e., maximize the likelihood of finding the shortest route towards the target destination), especially in complex and large-scale environments. Specifically, our robot learns to form a context of its belief over the entire known area, which it uses to reason about long-term efficiency and sequence show-term movements. Additionally, we propose a graph rarefaction algorithm to enable more efficient decision-making in large-scale applications. We empirically demonstrate that our approach reduces average travel time by up to $61.4%$ and average planning time by up to $88.2%$ compared to benchmark planners (D*lite and BIT) on hundreds of test scenarios. We also validate our approach both in high-fidelity Gazebo simulations as well as on hardware, highlighting its promising applicability in the real world without further training/tuning.

**摘要:** 基于 deep reinforcement learning(DRL)的最新发展,我们提出了一种基于学习的无地图导航框架,该框架采用具有上下文意识的政策网络实现有效的决策(即最大化寻找目标目的地的最短路径的可能性),特别是在复杂和大规模环境中。具体而言,我们的机器人学习在整个已知领域形成其信念的上下文,以推理长期效率和序列示例运动。此外,我们提出了一种图形稀释算法,使在大规模应用中更有效的决策。在数百个测试场景中,我们通过实验证明,我们的方法在高可靠性的Gazebo仿真和硬件上降低了平均旅行时间到61.4%$和平均规划时间到88.2%$,与基准规划者(D*lite和BIT)相比。

**[Paper URL](https://proceedings.mlr.press/v229/liang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liang23a/liang23a.pdf)** 

# Learning to Discern: Imitating Heterogeneous Human Demonstrations with Preference and Representation Learning
**题目:** 学习发现:与偏好和代表学习模仿异性人类演示

**作者:** Sachit Kuhar, Shuo Cheng, Shivang Chopra, Matthew Bronars, Danfei Xu

**Abstract:** Practical Imitation Learning (IL) systems rely on large human demonstration datasets for successful policy learning. However, challenges lie in maintaining the quality of collected data and addressing the suboptimal nature of some demonstrations, which can compromise the overall dataset quality and hence the learning outcome. Furthermore, the intrinsic heterogeneity in human behavior can produce equally successful but disparate demonstrations, further exacerbating the challenge of discerning demonstration quality. To address these challenges, this paper introduces Learning to Discern (L2D), an offline imitation learning framework for learning from demonstrations with diverse quality and style. Given a small batch of demonstrations with sparse quality labels, we learn a latent representation for temporally embedded trajectory segments. Preference learning in this latent space trains a quality evaluator that generalizes to new demonstrators exhibiting different styles. Empirically, we show that L2D can effectively assess and learn from varying demonstrations, thereby leading to improved policy performance across a range of tasks in both simulations and on a physical robot.

**摘要:** 实践仿真学习(IL)系统依靠大规模的人类演示数据集进行成功政策学习。然而,挑战在于保持收集数据的质量和解决某些演示的次优性质,这可能影响整个数据集质量,从而影响学习结果。此外,人类行为的内在异质性可以产生同样成功的但不同的演示,进一步加剧辨识演示质量的挑战。为了解决这些挑战,本文介绍了基于不同质量和风格的演示学习的在线仿真学习框架Learning to Discern(L2D)。实验表明,L2D能够有效地评估和从不同演示中学习,从而在仿真和实物机器人的一系列任务中提高政策性能。

**[Paper URL](https://proceedings.mlr.press/v229/kuhar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kuhar23a/kuhar23a.pdf)** 

# Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter
**题目:** 基于语言的机器人测距: Clutter中基于CLIP的参考测距合成

**作者:** Georgios Tziafas, Yucheng XU, Arushi Goel, Mohammadreza Kasaei, Zhibin Li, Hamidreza Kasaei

**Abstract:** Robots operating in human-centric environments require the integration of visual grounding and grasping capabilities to effectively manipulate objects based on user instructions. This work focuses on the task of referring grasp synthesis, which predicts a grasp pose for an object referred through natural language in cluttered scenes. Existing approaches often employ multi-stage pipelines that first segment the referred object and then propose a suitable grasp, and are evaluated in private datasets or simulators that do not capture the complexity of natural indoor scenes. To address these limitations, we develop a challenging benchmark based on cluttered indoor scenes from OCID dataset, for which we generate referring expressions and connect them with 4-DoF grasp poses. Further, we propose a novel end-to-end model (CROG) that leverages the visual grounding capabilities of CLIP to learn grasp synthesis directly from image-text pairs. Our results show that vanilla integration of CLIP with pretrained models transfers poorly in our challenging benchmark, while CROG achieves significant improvements both in terms of grounding and grasping. Extensive robot experiments in both simulation and hardware demonstrate the effectiveness of our approach in challenging interactive object grasping scenarios that include clutter.

**摘要:** 在人类中心环境中运行的机器人需要整合视觉基础和把握能力,以有效操作基于用户指令的对象。该工作重点在于参照抓捕合成的任务,该任务预测通过自然语言在杂乱场景中引用对象的抓捕姿态。现有的方法经常采用多阶段管道,首先分割引用对象,然后提出合适的抓捕姿态,并在私人数据集或模拟器中评估,这些数据集不能捕捉自然室内场景的复杂性。为了解决这些限制,我们开发了一个基于OCID数据集杂乱室内场景的挑战性指标,为此我们生成引用表达式并与4DoF抓捕姿态连接。结果表明,CLIP与预制模型的瓦尼拉整合在我们的挑战性指标中表现不佳,而CROG在接地和抓取方面取得了显著的改进。在模拟和硬件方面的广泛的机器人实验显示了我们对包括杂乱在内的挑战性交互对象抓取场景的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/tziafas23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tziafas23a/tziafas23a.pdf)** 

# Learning Reusable Manipulation Strategies
**题目:** 学习可重复操作策略

**作者:** Jiayuan Mao, Tomás Lozano-Pérez, Joshua B. Tenenbaum, Leslie Pack Kaelbling

**Abstract:** Humans demonstrate an impressive ability to acquire and generalize manipulation “tricks.” Even from a single demonstration, such as using soup ladles to reach for distant objects, we can apply this skill to new scenarios involving different object positions, sizes, and categories (e.g., forks and hammers). Additionally, we can flexibly combine various skills to devise long-term plans. In this paper, we present a framework that enables machines to acquire such manipulation skills, referred to as “mechanisms,” through a single demonstration and self-play. Our key insight lies in interpreting each demonstration as a sequence of changes in robot-object and object-object contact modes, which provides a scaffold for learning detailed samplers for continuous parameters. These learned mechanisms and samplers can be seamlessly integrated into standard task and motion planners, enabling their compositional use.

**摘要:** 人类具有获取和推广操纵“技巧”的能力,甚至从单一的演示中,例如使用汤棒来获取遥远的对象,我们可以将这种技能应用到涉及不同物体位置、大小和类别的新场景(例如叉子和锤子)。此外,我们可以灵活地结合各种技能来制定长期计划。本论文提出了一种框架,允许机器通过单一演示和自玩获得这些操纵技能,称为“机制”。我们的关键洞察在于将每个演示解释为机器人-物体和物体-物体接触模式中的变化序列,从而为学习连续参数的详细样本器提供了一个基础。这些学习的机制和样本器可以无缝地集成到标准任务和运动规划器中,使它们的组合使用。

**[Paper URL](https://proceedings.mlr.press/v229/mao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mao23a/mao23a.pdf)** 

# Sample-Efficient Preference-based Reinforcement Learning with Dynamics Aware Rewards
**题目:** 基于偏好的实例有效的强化学习,利用动态意识的奖励

**作者:** Katherine Metcalf, Miguel Sarabia, Natalie Mackraz, Barry-John Theobald

**Abstract:** Preference-based reinforcement learning (PbRL) aligns a robot behavior with human preferences via a reward function learned from binary feedback over agent behaviors. We show that encoding environment dynamics in the reward function improves the sample efficiency of PbRL by an order of magnitude. In our experiments we iterate between: (1) encoding environment dynamics in a state-action representation $z^{sa}$ via a self-supervised temporal consistency task, and (2) bootstrapping the preference-based reward function from $z^{sa}$, which results in faster policy learning and better final policy performance. For example, on quadruped-walk, walker-walk, and cheetah-run, with 50 preference labels we achieve the same performance as existing approaches with 500 preference labels, and we recover $83%$ and $66%$ of ground truth reward policy performance versus only $38%$ and $21%$ without environment dynamics. The performance gains demonstrate that explicitly encoding environment dynamics improves preference-learned reward functions.

**摘要:** 基于偏好强化学习(PbRL)通过从二进制反馈行为上学习的奖励函数,将机器人行为与人类偏好匹配。我们证明,在奖励函数中编码环境动态能提高PbRL的样品效率以数量顺序。在我们的实验中,我们重复: (一)通过自我监督的时间一致性任务编码环境动态在状态行动表示$z^{sa}$, (二)从$z^{sa}$ bootstrapping基于偏好奖励函数,这导致更快的政策学习和更好的最终政策性能。性能增益表明,明确的编码环境动态能改善优先学习的奖励函数。

**[Paper URL](https://proceedings.mlr.press/v229/metcalf23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/metcalf23a/metcalf23a.pdf)** 

# Im2Contact: Vision-Based Contact Localization Without Touch or Force Sensing
**题目:** Im2Contact:基于视觉的接触定位,无需触摸或感应力

**作者:** Leon Kim, Yunshuang Li, Michael Posa, Dinesh Jayaraman

**Abstract:** Contacts play a critical role in most manipulation tasks. Robots today mainly use proximal touch/force sensors to sense contacts, but the information they provide must be calibrated and is inherently local, with practical applications relying either on extensive surface coverage or restrictive assumptions to resolve ambiguities. We propose a vision-based extrinsic contact localization task: with only a single RGB-D camera view of a robot workspace, identify when and where an object held by the robot contacts the rest of the environment. We show that careful task-attuned design is critical for a neural network trained in simulation to discover solutions that transfer well to a real robot. Our final approach im2contact demonstrates the promise of versatile general-purpose contact perception from vision alone, performing well for localizing various contact types (point, line, or planar; sticking, sliding, or rolling; single or multiple), and even under occlusions in its camera view. Video results can be found at: https://sites.google.com/view/im2contact/home

**摘要:** 接触在大多数操作任务中扮演关键角色。如今的机器人主要使用近接触摸/强力传感器来感知接触,但它们提供的信息必须校正并具有局域性,而实际应用则依赖于广泛的表面覆盖或限制性假设来解决模糊问题。我们提出了基于视觉的外部接触定位任务:仅使用单个RGB-D摄像机查看机器人工作空间,确定机器人所持有的对象在何时和何处接触到其他环境。我们证明,精心设计的任务设计对于一个模拟训练的神经网络至关重要,以便发现能够向真正的机器人传递的解决方案。视频结果: https://sites.google.com/view/im2contact/home

**[Paper URL](https://proceedings.mlr.press/v229/kim23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf)** 

# DROID: Learning from Offline Heterogeneous Demonstrations via Reward-Policy Distillation
**题目:** DROID:通过奖励政策蒸馏从网上异质示范中学习

**作者:** Sravan Jayanthi, Letian Chen, Nadya Balabanska, Van Duong, Erik Scarlatescu, Ezra Ameperosa, Zulfiqar Haider Zaidi, Daniel Martin, Taylor Keith Del Matto, Masahiro Ono, Matthew Gombolay

**Abstract:** Offline Learning from Demonstrations (OLfD) is valuable in domains where trial-and-error learning is infeasible or specifying a cost function is difficult, such as robotic surgery, autonomous driving, and path-finding for NASA’s Mars rovers. However, two key problems remain challenging in OLfD: 1) heterogeneity: demonstration data can be generated with diverse preferences and strategies, and 2) generalizability: the learned policy and reward must perform well beyond a limited training regime in unseen test settings. To overcome these challenges, we propose Dual Reward and policy Offline Inverse Distillation (DROID), where the key idea is to leverage diversity to improve generalization performance by decomposing common-task and individual-specific strategies and distilling knowledge in both the reward and policy spaces. We ground DROID in a novel and uniquely challenging Mars rover path-planning problem for NASA’s Mars Curiosity Rover. We also curate a novel dataset along 163 Sols (Martian days) and conduct a novel, empirical investigation to characterize heterogeneity in the dataset. We find DROID outperforms prior SOTA OLfD techniques, leading to a $26%$ improvement in modeling expert behaviors and $92%$ closer to the task objective of reaching the final destination. We also benchmark DROID on the OpenAI Gym Cartpole environment and find DROID achieves $55%$ (significantly) better performance modeling heterogeneous demonstrations.

**摘要:** 基于在线演示的在线学习(OLfD)在机器人手术、自主驾驶和NASA火星探测器路径识别等难以实现的领域具有价值。然而,OLfD仍面临着两个关键问题: 1)异质性:演示数据可以以不同的偏好和策略生成; 2)可归纳性:学习的政策和奖励必须在未见的测试环境下远远超出有限的训练制度。为了克服这些挑战,我们提出了双重奖励和政策 Offline Inverse Distillation(DROID),其中关键思想是利用多样性来改进一般化性能,通过分解共同任务和个人特有策略,并在奖励和政策空间中蒸馏知识。我们还管理了163个Sols(马丁日)的全新数据集,并进行了新颖的实证研究,以区分数据集的异质性。我们发现DROID比以前的SOTA OLfD技术高,导致了26%的专家行为模型的改进和92%的接近最终目标的任务目标。

**[Paper URL](https://proceedings.mlr.press/v229/jayanthi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/jayanthi23a/jayanthi23a.pdf)** 

# SA6D: Self-Adaptive Few-Shot 6D Pose Estimator for Novel and Occluded Objects
**题目:** SA6D:新型及隐形物体的自适应短射6D姿势估算器

**作者:** Ning Gao, Vien Anh Ngo, Hanna Ziesche, Gerhard Neumann

**Abstract:** To enable meaningful robotic manipulation of objects in the real-world, 6D pose estimation is one of the critical aspects. Most existing approaches have difficulties to extend predictions to scenarios where novel object instances are continuously introduced, especially with heavy occlusions. In this work, we propose a few-shot pose estimation (FSPE) approach called SA6D, which uses a self-adaptive segmentation module to identify the novel target object and construct a point cloud model of the target object using only a small number of cluttered reference images. Unlike existing methods, SA6D does not require object-centric reference images or any additional object information, making it a more generalizable and scalable solution across categories. We evaluate SA6D on real-world tabletop object datasets and demonstrate that SA6D outperforms existing FSPE methods, particularly in cluttered scenes with occlusions, while requiring fewer reference images.

**摘要:** 为了在现实世界中实现有意义的机器人操作,6D姿态估计是关键方面之一。大多数现有的方法难以将预测扩展到新对象实例不断引入的场景,特别是在重度掩盖的情况下。本研究中,我们提出了一种名为SA6D的少数镜头姿态估计(FSPE)方法,该方法使用自适应分割模块来识别新目标对象,并只使用少量杂乱的参考图像构建目标对象的点云模型。与现有方法不同,SA6D不需要对象中心的参考图像或任何额外的对象信息,使它成为更通用和可扩展的解决方案。

**[Paper URL](https://proceedings.mlr.press/v229/gao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gao23a/gao23a.pdf)** 

# Hierarchical Planning for Rope Manipulation using Knot Theory and a Learned Inverse Model
**题目:** 基于节点理论和学习逆模型的轮船操纵层级规划

**作者:** Matan Sudry, Tom Jurgenson, Aviv Tamar, Erez Karpas

**Abstract:** This work considers planning the manipulation of deformable 1-dimensional objects, such as ropes or cables, specifically to tie knots. We propose TWISTED: Tying With Inverse model and Search in Topological space Excluding Demos, a hierarchical planning approach which, at the high level, uses ideas from knot-theory to plan a sequence of rope configurations, while at the low level uses a neural-network inverse model to move between the configurations in the high-level plan. To train the neural network, we propose a self-supervised approach, where we learn from random movements of the rope. To focus the random movements on interesting configurations, such as knots, we propose a non-uniform sampling method tailored for this domain. In a simulation, we show that our approach can plan significantly faster and more accurately than baselines. We also show that our plans are robust to parameter changes in the physical simulation, suggesting future applications via sim2real.

**摘要:** 本文考虑了规划变形 1- 维 物体,例如绳子或电缆,具体用于绑结节点的操作。我们提出TWISTED: Tying With Inverse Model and Search in Topological Space 不包括Demos,一种层次规划方法,在高层次使用节点理论的观念来规划绳子配置的序列,在低层次使用神经网络反向模型来移动在高层次规划中的配置。为了训练神经网络,我们提出了自监督的方法,我们从绳子随机运动中学习。为了把随机运动集中于有趣的配置,例如节点,我们提出了针对这个领域设计的非统一样本方法。

**[Paper URL](https://proceedings.mlr.press/v229/sudry23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sudry23a/sudry23a.pdf)** 

# OVIR-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data
**题目:** OVIR-3D:无训练的开放语音3D实例检索

**作者:** Shiyang Lu, Haonan Chang, Eric Pu Jing, Abdeslam Boularias, Kostas Bekris

**Abstract:** This work presents OVIR-3D, a straightforward yet effective method for open-vocabulary 3D object instance retrieval without using any 3D data for training. Given a language query, the proposed method is able to return a ranked set of 3D object instance segments based on the feature similarity of the instance and the text query. This is achieved by a multi-view fusion of text-aligned 2D region proposals into 3D space, where the 2D region proposal network could leverage 2D datasets, which are more accessible and typically larger than 3D datasets. The proposed fusion process is efficient as it can be performed in real-time for most indoor 3D scenes and does not require additional training in 3D space. Experiments on public datasets and a real robot show the effectiveness of the method and its potential for applications in robot navigation and manipulation.

**摘要:** 本文介绍OVIR-3D,一种无使用任何3D数据进行训练的开放口语3D对象实例检索的简单而有效的方法。基于语言查询,该方法能够返回基于实例和文本查询的特征相似性而排列的3D对象实例段组。该方法通过多视图融合 text-aligned 2D region proposals into 3D space,使得2D region proposal network能够利用2D datasets,这些数据set比3D datasets更容易访问,而且通常比3D datasets大。该融合过程是有效的,因为它可以在大多数室内3D场景中实时执行,不需要在3D空间进行额外训练。

**[Paper URL](https://proceedings.mlr.press/v229/lu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lu23a/lu23a.pdf)** 

# Efficient Sim-to-real Transfer of Contact-Rich Manipulation Skills with Online Admittance Residual Learning
**题目:** 通过在线汇款剩余学习,有效地模拟到现实的接触式操纵技能转移

**作者:** Xiang Zhang, Changhao Wang, Lingfeng Sun, Zheng Wu, Xinghao Zhu, Masayoshi Tomizuka

**Abstract:** Learning contact-rich manipulation skills is essential. Such skills require the robots to interact with the environment with feasible manipulation trajectories and suitable compliance control parameters to enable safe and stable contact. However, learning these skills is challenging due to data inefficiency in the real world and the sim-to-real gap in simulation. In this paper, we introduce a hybrid offline-online framework to learn robust manipulation skills. We employ model-free reinforcement learning for the offline phase to obtain the robot motion and compliance control parameters in simulation \RV{with domain randomization}. Subsequently, in the online phase, we learn the residual of the compliance control parameters to maximize robot performance-related criteria with force sensor measurements in real-time. To demonstrate the effectiveness and robustness of our approach, we provide comparative results against existing methods for assembly, pivoting, and screwing tasks.

**摘要:** 学习接触丰富的操纵技能是必不可少的,这些技能要求机器人与环境相互作用,具有可行的操纵轨迹和适当的遵守控制参数,以使安全稳定接触。然而,学习这些技能是由于现实世界的数据效率不足和模拟中的模拟模拟模拟中的模拟模拟模拟中的模拟模拟模拟中的模拟模拟间隔所引起的挑战。本文介绍了一种混合的在线在线框架,以学习鲁棒操纵技能。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23e/zhang23e.pdf)** 

# Tell Me Where to Go: A Composable Framework for Context-Aware Embodied Robot Navigation
**题目:** Tell Me Where to Go:一个具有上下文意识的隐身机器人导航构架

**作者:** Harel Biggie, Ajay Narasimha Mopidevi, Dusty Woods, Chris Heckman

**Abstract:** Humans have the remarkable ability to navigate through unfamiliar environments by solely relying on our prior knowledge and descriptions of the environment. For robots to perform the same type of navigation, they need to be able to associate natural language descriptions with their associated physical environment with a limited amount of prior knowledge. Recently, Large Language Models (LLMs) have been able to reason over billions of parameters and utilize them in multi-modal chat-based natural language responses. However, LLMs lack real-world awareness and their outputs are not always predictable. In this work, we develop a low-bandwidth framework that solves this lack of real-world generalization by creating an intermediate layer between an LLM and a robot navigation framework in the form of Python code. Our intermediate shoehorns the vast prior knowledge inherent in an LLM model into a series of input and output API instructions that a mobile robot can understand. We evaluate our method across four different environments and command classes on a mobile robot and highlight our framework’s ability to interpret contextual commands.

**摘要:** 人类具有 navigating 通过 不熟悉的环境 的 显著 能力, 仅 依靠 我们 的 先知 和 环境 的 描述 。 为了 机器人 执行 同样 的 导航, 他们 需要 能够 与 其 关联 的 物理 环境 结合 自然语言 的 描述, 并 具有 有限 的 先知 。 最近, 大型 语言 模型 ( LLM ) 已经 能够 推理 超过 数十亿 参数, 并 利用 它们 在 多 模式 的 聊天 基础 的 自然语言 响应 中 。 然而, LLM 缺乏 现实 世界 的 认识 和 其 产出 并不 总是 可预测 。我们在移动机器人的四个不同环境和命令类中评估了我们的方法,并突出了我们的框架能够解释上下文命令的能力。

**[Paper URL](https://proceedings.mlr.press/v229/biggie23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/biggie23a/biggie23a.pdf)** 

# Dynamic Multi-Team Racing: Competitive Driving on 1/10-th Scale Vehicles via Learning in Simulation
**题目:** 动态多组赛:通过仿真学习在1/10级车辆上进行竞争驾驶

**作者:** Peter Werner, Tim Seyde, Paul Drews, Thomas Matrai Balch, Igor Gilitschenski, Wilko Schwarting, Guy Rosman, Sertac Karaman, Daniela Rus

**Abstract:** Autonomous racing is a challenging task that requires vehicle handling at the dynamic limits of friction. While single-agent scenarios like Time Trials are solved competitively with classical model-based or model-free feedback control, multi-agent wheel-to-wheel racing poses several challenges including planning over unknown opponent intentions as well as negotiating interactions under dynamic constraints. We propose to address these challenges via a learning-based approach that effectively combines model-based techniques, massively parallel simulation, and self-play reinforcement learning to enable zero-shot sim-to-real transfer of highly dynamic policies. We deploy our algorithm in wheel-to-wheel multi-agent races on scale hardware to demonstrate the efficacy of our approach. Further details and videos can be found on the project website: https://sites.google.com/view/dynmutr/home.

**摘要:** 自力赛是一个需要在摩擦力动态限度下处理车辆的挑战性任务。 While single-agent scenarios like Time Trials are solved competitively with classical model-based or model-free feedback control, multi-agent wheel-to-wheel racing poses several challenges including planning over unknown opponent intentions as well as negotiating interactions under dynamic constraints。 我们建议通过一个基于学习的方法来解决这些挑战,它有效地结合了基于模型的技术,大规模的平行仿真,以及自演增强学习,使高动态政策的零射击模拟到实际转移。

**[Paper URL](https://proceedings.mlr.press/v229/werner23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/werner23a/werner23a.pdf)** 

# Stochastic Occupancy Grid Map Prediction in Dynamic Scenes
**题目:** 动态场景中随机占地网格图预测

**作者:** Zhanteng Xie, Philip Dames

**Abstract:** This paper presents two variations of a novel stochastic prediction algorithm that enables mobile robots to accurately and robustly predict the future state of complex dynamic scenes. The proposed algorithm uses a variational autoencoder to predict a range of possible future states of the environment. The algorithm takes full advantage of the motion of the robot itself, the motion of dynamic objects, and the geometry of static objects in the scene to improve prediction accuracy. Three simulated and real-world datasets collected by different robot models are used to demonstrate that the proposed algorithm is able to achieve more accurate and robust prediction performance than other prediction algorithms. Furthermore, a predictive uncertainty-aware planner is proposed to demonstrate the effectiveness of the proposed predictor in simulation and real-world navigation experiments. Implementations are open source at https://github.com/TempleRAIL/SOGMP.

**摘要:** 本文介绍了一种新型随机预测算法的两个变异,使移动机器人能够准确可靠地预测复杂动态场景的未来状态。该算法采用变异自动编码器来预测环境可能的未来状态。该算法充分利用机器人本身的运动、动态对象的运动和场景中的静态对象的几何,以提高预测精度。由不同机器人模型收集的三个模拟和现实数据集被用来证明该算法能够比其他预测算法取得更准确、更强的预测性能。

**[Paper URL](https://proceedings.mlr.press/v229/xie23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xie23a/xie23a.pdf)** 

# A Bayesian approach to breaking things: efficiently predicting and repairing failure modes via sampling
**题目:** 破坏物体的贝叶斯方法:通过采样有效预测和修复故障模式

**作者:** Charles Dawson, Chuchu Fan

**Abstract:** Before autonomous systems can be deployed in safety-critical applications, we must be able to understand and verify the safety of these systems. For cases where the risk or cost of real-world testing is prohibitive, we propose a simulation-based framework for a) predicting ways in which an autonomous system is likely to fail and b) automatically adjusting the system’s design to preemptively mitigate those failures. We frame this problem through the lens of approximate Bayesian inference and use differentiable simulation for efficient failure case prediction and repair. We apply our approach on a range of robotics and control problems, including optimizing search patterns for robot swarms and reducing the severity of outages in power transmission networks. Compared to optimization-based falsification techniques, our method predicts a more diverse, representative set of failure modes, and we also find that our use of differentiable simulation yields solutions that have up to 10x lower cost and requires up to 2x fewer iterations to converge relative to gradient-free techniques.

**摘要:** 在自动系统在安全关键应用中部署之前,我们必须能够理解和验证这些系统的安全。在实际测试风险或成本过高的情况下,我们提出了一种基于仿真的框架:(a)预测自动系统可能发生故障的途径,以及(b)自动调整系统设计,以预防这些故障。我们通过近似贝叶斯推理的视角来构造这一问题,并使用可微分的仿真来有效预测和修复故障情况。我们应用我们的方法在各种机器人和控制问题上,包括优化机器人群组的搜索模式和减少电力传输网络中停电的严重程度。与基于优化的伪造技术相比,我们的方法预测了更多样的、更具代表性的故障模式,而且我们发现,我们使用可分化仿真可以产生最大10倍低成本的解决方案,并且需要最大2倍少的迭代来与无梯度技术相近。

**[Paper URL](https://proceedings.mlr.press/v229/dawson23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/dawson23a/dawson23a.pdf)** 

# BridgeData V2: A Dataset for Robot Learning at Scale
**题目:** BridgeData V2:基于尺度的机器人学习数据集

**作者:** Homer Rich Walke, Kevin Black, Tony Z. Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, Abraham Lee, Kuan Fang, Chelsea Finn, Sergey Levine

**Abstract:** We introduce BridgeData V2, a large and diverse dataset of robotic manipulation behaviors designed to facilitate research in scalable robot learning. BridgeData V2 contains 53,896 trajectories collected across 24 environments on a publicly available low-cost robot. Unlike many existing robotic manipulation datasets, BridgeData V2 provides enough task and environment variability that skills learned from the data generalize across institutions, making the dataset a useful resource for a broad range of researchers. Additionally, the dataset is compatible with a wide variety of open-vocabulary, multi-task learning methods conditioned on goal images or natural language instructions. In our experiments,we apply 6 state-of-the-art imitation learning and offline reinforcement learning methods to the data and find that they succeed on a suite of tasks requiring varying amounts of generalization. We also demonstrate that the performance of these methods improves with more data and higher capacity models. By publicly sharing BridgeData V2 and our pre-trained models, we aim to accelerate research in scalable robot learning methods.

**摘要:** 本文介绍了基于可扩展机器人学习的机器人操纵行为数据集 BridgeData V2 。 BridgeData V2 包含了在24个环境中收集的53,896个低成本机器人轨迹。与许多现有的机器人操纵数据集不同, BridgeData V2 提供足够的任务和环境变异性,使从数据中学习的技能在各个机构中推广,使该数据集成为广泛的研究人员的有用资源。 此外,该数据集与基于目标图像或自然语言指示的开放口语、多任务学习方法的广泛互換。通过公开分享 BridgeData V2和我们的预训练模型,我们旨在加速研究可扩展机器人学习方法。

**[Paper URL](https://proceedings.mlr.press/v229/walke23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/walke23a/walke23a.pdf)** 

# NOIR: Neural Signal Operated Intelligent Robots for Everyday Activities
**题目:** NOIR:用于日常活动的神经信号操作智能机器人

**作者:** Ruohan Zhang, Sharon Lee, Minjune Hwang, Ayano Hiranaka, Chen Wang, Wensi Ai, Jin Jie Ryan Tan, Shreya Gupta, Yilun Hao, Gabrael Levine, Ruohan Gao, Anthony Norcia, Li Fei-Fei, Jiajun Wu

**Abstract:** We present Neural Signal Operated Intelligent Robots (NOIR), a general-purpose, intelligent brain-robot interface system that enables humans to command robots to perform everyday activities through brain signals. Through this interface, humans communicate their intended objects of interest and actions to the robots using electroencephalography (EEG). Our novel system demonstrates success in an expansive array of 20 challenging, everyday household activities, including cooking, cleaning, personal care, and entertainment. The effectiveness of the system is improved by its synergistic integration of robot learning algorithms, allowing for NOIR to adapt to individual users and predict their intentions. Our work enhances the way humans interact with robots, replacing traditional channels of interaction with direct, neural communication.

**摘要:** 神经信号操作智能机器人(英语:Neural Signal Operated Intelligent Robots,NOIR)是一种通用的智能脑-机器人界面系统,允许人类通过脑信号来指挥机器人进行日常活动。通过这种界面,人类通过电脑图谱(英语:EEG)向机器人传达其预期的对象和行为。我们的新系统展示了20项挑战性的日常家庭活动,包括烹饪、清洁、个人护理和娱乐等。该系统通过机器人学习算法的协同集成,提高了系统效率,使NOIR能够适应个人用户并预测他们的意愿。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23f.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23f/zhang23f.pdf)** 

# PolarNet: 3D Point Clouds for Language-Guided Robotic Manipulation
**题目:** PolarNet:基于语言的机器人操纵的3D点云

**作者:** Shizhe Chen, Ricardo Garcia Pinel, Cordelia Schmid, Ivan Laptev

**Abstract:** The ability for robots to comprehend and execute manipulation tasks based on natural language instructions is a long-term goal in robotics. The dominant approaches for language-guided manipulation use 2D image representations, which face difficulties in combining multi-view cameras and inferring precise 3D positions and relationships. To address these limitations, we propose a 3D point cloud based policy called PolarNet for language-guided manipulation. It leverages carefully designed point cloud inputs, efficient point cloud encoders, and multimodal transformers to learn 3D point cloud representations and integrate them with language instructions for action prediction. PolarNet is shown to be effective and data efficient in a variety of experiments conducted on the RLBench benchmark. It outperforms state-of-the-art 2D and 3D approaches in both single-task and multi-task learning. It also achieves promising results on a real robot.

**摘要:** 基于自然语言指令的机器人能够理解和执行操纵任务是机器人的长期目标。语言指导操纵的主导方法使用2D图像表示,在结合多视角摄像机和推导精确的3D位置和关系方面遇到困难。为了解决这些限制,我们提出了一种基于3D点云的政策,即PolarNet,用于语言指导操纵。它利用精心设计的点云输入、高效的点云编码器和多模态变换器学习3D点云表示并将其与语言指令集成为行动预测。

**[Paper URL](https://proceedings.mlr.press/v229/chen23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23b/chen23b.pdf)** 

# Stealthy Terrain-Aware Multi-Agent Active Search
**题目:** 机密地带意识的多代理人主动搜索

**作者:** Nikhil Angad Bakshi, Jeff Schneider

**Abstract:** Stealthy multi-agent active search is the problem of making efficient sequential data-collection decisions to identify an unknown number of sparsely located targets while adapting to new sensing information and concealing the search agents’ location from the targets. This problem is applicable to reconnaissance tasks wherein the safety of the search agents can be compromised as the targets may be adversarial. Prior work usually focuses either on adversarial search, where the risk of revealing the agents’ location to the targets is ignored or evasion strategies where efficient search is ignored. We present the Stealthy Terrain-Aware Reconnaissance (STAR) algorithm, a multi-objective parallelized Thompson sampling-based algorithm that relies on a strong topographical prior to reason over changing visibility risk over the course of the search. The STAR algorithm outperforms existing state-of-the-art multi-agent active search methods on both rate of recovery of targets as well as minimising risk even when subject to noisy observations, communication failures and an unknown number of targets.

**摘要:** 隐形多agent主动搜索(英语:Stealthy multi-agent active search)是识别未知数稀有地点目标并适应新的感知信息和隐藏目标的搜索代理位置的有效连续数据收集决策问题。该问题适用于侦察任务,其中搜索代理的安全可能会受到破坏,因为目标可能是敌对的。以前的工作通常集中于敌对搜索,其中忽略了发现代理位置的危险或忽略了有效搜索的避险策略。我们介绍隐形地带意识侦察(STAR)算法,一种基于汤普森样本的多目标平行算法,它依赖于对搜索过程中可视度变化风险的强烈地层前推理。STAR算法在目标恢复率上优于现有的最先进的多代理主动搜索方法,即使在受到噪音观测、通信故障和未知目标数的情况下,也尽量减少风险。

**[Paper URL](https://proceedings.mlr.press/v229/bakshi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/bakshi23a/bakshi23a.pdf)** 

# A Data-Efficient Visual-Audio Representation with Intuitive Fine-tuning for Voice-Controlled Robots
**题目:** 基于直觉微调的语音控制机器人数据高效的视觉-音频表示

**作者:** Peixin Chang, Shuijing Liu, Tianchen Ji, Neeloy Chakraborty, Kaiwen Hong, Katherine Rose Driggs-Campbell

**Abstract:** A command-following robot that serves people in everyday life must continually improve itself in deployment domains with minimal help from its end users, instead of engineers. Previous methods are either difficult to continuously improve after the deployment or require a large number of new labels during fine-tuning. Motivated by (self-)supervised contrastive learning, we propose a novel representation that generates an intrinsic reward function for command-following robot tasks by associating images with sound commands. After the robot is deployed in a new domain, the representation can be updated intuitively and data-efficiently by non-experts without any hand-crafted reward functions. We demonstrate our approach on various sound types and robotic tasks, including navigation and manipulation with raw sensor inputs. In simulated and real-world experiments, we show that our system can continually self-improve in previously unseen scenarios given fewer new labeled data, while still achieving better performance over previous methods.

**摘要:** 一个在日常工作中服务的命令追踪机器人必须在部署领域内不断改进自己,以最小限度的帮助从其最终用户,而不是工程师。以前的方法要么在部署后难以持续改进,要么在微调过程中需要大量新的标签。由(自我)监督的对比学习动机,我们提出了一种新的表示,它通过将图像与声音命令关联为命令追踪机器人任务产生内在的奖励函数。在机器人部署到一个新的领域后,该表示可以由非专家直观地和数据效率地更新,而无需手工制作的奖励函数。在模拟和现实实验中,我们证明,我们的系统可以在少量新的标签数据下持续自我改进,同时仍能比以前的方法取得更好的性能。

**[Paper URL](https://proceedings.mlr.press/v229/chang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chang23a/chang23a.pdf)** 

# MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations
**题目:** MimicGen:基于人类演示的可扩展机器人学习数据生成系统

**作者:** Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox

**Abstract:** Imitation learning from a large set of human demonstrations has proved to be an effective paradigm for building capable robot agents. However, the demonstrations can be extremely costly and time-consuming to collect. We introduce MimicGen, a system for automatically synthesizing large-scale, rich datasets from only a small number of human demonstrations by adapting them to new contexts. We use MimicGen to generate over 50K demonstrations across 18 tasks with diverse scene configurations, object instances, and robot arms from just  200 human demonstrations. We show that robot agents can be effectively trained on this generated dataset by imitation learning to achieve strong performance in long-horizon and high-precision tasks, such as multi-part assembly and coffee preparation, across broad initial state distributions. We further demonstrate that the effectiveness and utility of MimicGen data compare favorably to collecting additional human demonstrations, making it a powerful and economical approach towards scaling up robot learning. Datasets, simulation environments, videos, and more at https://mimicgen.github.io.

**摘要:** 仿真学习是建立有能力机器人代理的有效范式,但仿真学习的成本非常昂贵,并花费大量时间。我们引入了MimicGen,一种以适应新环境为对象的仿真学习系统,以自动合成仅少数人仿真的大规模、丰富的数据集。我们使用MimicGen,在18个任务中生成超过50K的仿真,包括各种场景配置、对象实例和仅200个人仿真的机器人手臂。我们证明,仿真学习可以有效地训练机器人代理在长视线和高精度任务中取得强性能,例如多部组装和咖啡制备,在广泛的初始状态分布中。我们进一步证明了MimicGen数据的有效性和实用性与收集额外的人类演示相比,使它成为增强机器人学习的强大和经济的方法。

**[Paper URL](https://proceedings.mlr.press/v229/mandlekar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mandlekar23a/mandlekar23a.pdf)** 

# Quantifying Assistive Robustness Via the Natural-Adversarial Frontier
**题目:** 通过自然敌对边界量化辅助鲁棒性

**作者:** Jerry Zhi-Yang He, Daniel S. Brown, Zackory Erickson, Anca Dragan

**Abstract:** Our ultimate goal is to build robust policies for robots that assist people. What makes this hard is that people can behave unexpectedly at test time, potentially interacting with the robot outside its training distribution and leading to failures. Even just measuring robustness is a challenge. Adversarial perturbations are the default, but they can paint the wrong picture: they can correspond to human motions that are unlikely to occur during natural interactions with people. A robot policy might fail under small adversarial perturbations but work under large natural perturbations. We propose that capturing robustness in these interactive settings requires constructing and analyzing the entire natural-adversarial frontier: the Pareto-frontier of human policies that are the best trade-offs between naturalness and low robot performance. We introduce RIGID, a method for constructing this frontier by training adversarial human policies that trade off between minimizing robot reward and acting human-like (as measured by a discriminator). On an Assistive Gym task, we use RIGID to analyze the performance of standard collaborative RL, as well as the performance of existing methods meant to increase robustness. We also compare the frontier RIGID identifies with the failures identified in expert adversarial interaction, and with naturally-occurring failures during user interaction. Overall, we find evidence that RIGID can provide a meaningful measure of robustness predictive of deployment performance, and uncover failure cases that are difficult to find manually.

**摘要:** 我们的最终目标是为人提供帮助的机器人建立强有力的政策。这使得人们在测试时可以不料得地表现出来,可能与机器人在训练分配之外相互作用,导致失败。甚至仅仅测量强力也是一项挑战。敌对扰动是默认的,但它们可以绘制错误的图案:它们可以与与人之间的自然交互中不太可能发生的人类运动相符。我们引入了“RIGID”(英语:RIGID),一种通过训练敌对人类政策来构建这一边界的方法,它在最小限度的机器人奖赏和像人一样的行为之间进行交易。在辅助健身房任务中,我们使用“RIGID”(英语:RIGID)来分析标准协作RL的性能,以及提高鲁棒性的现有方法的性能。我们还比较了“RIGID”(英语:RIGID)的边界与专家敌对交互中发现的故障,以及在用户交互中自然发生的故障。

**[Paper URL](https://proceedings.mlr.press/v229/he23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/he23a/he23a.pdf)** 

# Dynamic Handover: Throw and Catch with Bimanual Hands
**题目:** 动态递交:用双手投掷和抓捕

**作者:** Binghao Huang, Yuanpei Chen, Tianyu Wang, Yuzhe Qin, Yaodong Yang, Nikolay Atanasov, Xiaolong Wang

**Abstract:** Humans throw and catch objects all the time. However, such a seemingly common skill introduces a lot of challenges for robots to achieve: The robots need to operate such dynamic actions at high-speed, collaborate precisely, and interact with diverse objects. In this paper, we design a system with two multi-finger hands attached to robot arms to solve this problem. We train our system using Multi-Agent Reinforcement Learning in simulation and perform Sim2Real transfer to deploy on the real robots. To overcome the Sim2Real gap, we provide multiple novel algorithm designs including learning a trajectory prediction model for the object. Such a model can help the robot catcher has a real-time estimation of where the object will be heading, and then react accordingly. We conduct our experiments with multiple objects in the real-world system, and show significant improvements over multiple baselines. Our project page is available at https://binghao-huang.github.io/dynamic_handover/

**摘要:** 人类经常投掷和捕获物体。然而,这种看似普遍的技能为机器人带来许多挑战:机器人需要在高速操作这些动态动作,准确协作,并与各种物体互动。本文设计了一种与机器人手臂连接的双指手系统,以解决这一问题。我们使用多代理强化学习来训练我们的系统,并在模拟中执行Sim2Real转移,以便在实际机器人上部署。为了克服Sim2Real的差距,我们提供了多种新算法的设计,包括学习对象的轨迹预测模型。

**[Paper URL](https://proceedings.mlr.press/v229/huang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/huang23d/huang23d.pdf)** 

# Cross-Dataset Sensor Alignment: Making Visual 3D Object Detector Generalizable
**题目:** 交叉数据集传感器配置:可将可视3D对象检测器推广

**作者:** Liangtao Zheng, Yicheng Liu, Yue Wang, Hang Zhao

**Abstract:** While camera-based 3D object detection has evolved rapidly, these models are susceptible to overfitting to specific sensor setups. For example, in autonomous driving, most datasets are collected using a single sensor configuration. This paper evaluates the generalization capability of camera-based 3D object detectors, including adapting detectors from one dataset to another and training detectors with multiple datasets. We observe that merely aggregating datasets yields drastic performance drops, contrary to the expected improvements associated with increased training data. To close the gap, we introduce an efficient technique for aligning disparate sensor configurations — a combination of camera intrinsic synchronization, camera extrinsic correction, and ego frame alignment, which collectively enhance cross-dataset performance remarkably. Compared with single dataset baselines, we achieve 42.3 mAP improvement on KITTI, 23.2 mAP improvement on Lyft, 18.5 mAP improvement on nuScenes, 17.3 mAP improvement on KITTI-360, 8.4 mAP improvement on Argoverse2 and 3.9 mAP improvement on Waymo. We hope this comprehensive study can facilitate research on generalizable 3D object detection and associated tasks.

**摘要:** 虽然基于摄像头的3D对象检测已经迅速发展,但这些模型易于适应特定传感器设置。例如,在自主驾驶中,大部分数据集都是通过单个传感器配置收集的。本文评价了基于摄像头的3D对象检测器的一般化能力,包括从一个数据集到另一个数据集的检测器的适应和与多个数据集的训练检测器的训练。与单个数据集基线相比,我们取得了42.3mAP改进 KITTI,23.2mAP改进 Lyft,18.5mAP改进 nuScenes,17.3mAP改进 KITTI-360,8.4mAP改进 Argoverse2和3.9mAP改进 Waymo。我们希望这项综合研究能够促进可推广的3D对象检测和相关任务的研究。

**[Paper URL](https://proceedings.mlr.press/v229/zheng23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zheng23a/zheng23a.pdf)** 

# REBOOT: Reuse Data for Bootstrapping Efficient Real-World Dexterous Manipulation
**题目:** REBOOT:重新利用数据来启动高效的现实世界敏捷操作

**作者:** Zheyuan Hu, Aaron Rovinsky, Jianlan Luo, Vikash Kumar, Abhishek Gupta, Sergey Levine

**Abstract:** Dexterous manipulation tasks involving contact-rich interactions pose a significant challenge for both model-based control systems and imitation learning algorithms. The complexity arises from the need for multi-fingered robotic hands to dynamically establish and break contacts, balance forces on the non-prehensile object, and control a high number of degrees of freedom. Reinforcement learning (RL) offers a promising approach due to its general applicability and capacity to autonomously acquire optimal manipulation strategies. However, its real-world application is often hindered by the necessity to generate a large number of samples, reset the environment, and obtain reward signals. In this work, we introduce an efficient system for learning dexterous manipulation skills with RL to alleviate these challenges. The main idea of our approach is the integration of recent advancements in sample-efficient RL and replay buffer bootstrapping. This unique combination allows us to utilize data from different tasks or objects as a starting point for training new tasks, significantly improving learning efficiency. Additionally, our system completes the real-world training cycle by incorporating learned resets via an imitation-based pickup policy and learned reward functions, to eliminate the need for manual reset and reward engineering. We show the benefits of reusing past data as replay buffer initialization for new tasks, for instance, the fast acquisitions of intricate manipulation skills in the real world on a four-fingered robotic hand. https://sites.google.com/view/reboot-dexterous

**摘要:** 基于模型控制系统和仿真学习算法的复杂性在于需要多指的机器人手动态建立和打破接触,平衡非接触对象的力,并控制大量的自由度。强化学习(RL)由于其通用适用性和自主获取最佳操作策略的能力,提供了一种有前途的方法。然而,其现实应用往往受到大量样品生成、环境重置和奖励信号的必要性所阻碍。这种独特的组合使我们能够利用不同任务或对象的数据作为训练新任务的起点,大大提高学习效率。此外,我们的系统通过模仿式拾取政策和学习奖励功能,通过引入学习重启来完成真实世界训练周期,消除了手动重启和奖励工程的必要性。

**[Paper URL](https://proceedings.mlr.press/v229/hu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/hu23a/hu23a.pdf)** 

# Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs
**题目:** 基于开放口语的3D场景图形的语境意识实体地基

**作者:** Haonan Chang, Kowndinya Boyalakuntla, Shiyang Lu, Siwei Cai, Eric Pu Jing, Shreesh Keskar, Shijie Geng, Adeeb Abbas, Lifeng Zhou, Kostas Bekris, Abdeslam Boularias

**Abstract:** We present an Open-Vocabulary 3D Scene Graph (OVSG), a formal framework for grounding a variety of entities, such as object instances, agents, and regions, with free-form text-based queries. Unlike conventional semantic-based object localization approaches, our system facilitates context-aware entity localization, allowing for queries such as “pick up a cup on a kitchen table" or “navigate to a sofa on which someone is sitting". In contrast to existing research on 3D scene graphs, OVSG supports free-form text input and open-vocabulary querying. Through a series of comparative experiments using the ScanNet dataset and a self-collected dataset, we demonstrate that our proposed approach significantly surpasses the performance of previous semantic-based localization techniques. Moreover, we highlight the practical application of OVSG in real-world robot navigation and manipulation experiments. The code and dataset used for evaluation will be made available upon publication.

**摘要:** 我们提出了一种面向对象实例、代理人和区域的自由文本查询的开放语音场景图(Open-Vocabulary 3D Scene Graph,OVSG)形式框架。与传统的基于语义的对象定位方法不同,我们的系统促进了基于语义的实体定位,允许查询如“把杯子放在厨房桌上”或“向某人坐着的沙发上航行”。与现有的3D场景图研究相比,OVSG支持自由文本输入和开放语音查询。通过一系列使用ScanNet数据集和自收集数据集的比较实验,我们证明了我们提出的方法大大超过了以前基于语义的定位技术的表现。此外,我们突出了OVSG在实际机器人导航和操作实验中的应用。

**[Paper URL](https://proceedings.mlr.press/v229/chang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chang23b/chang23b.pdf)** 

# HomeRobot: Open-Vocabulary Mobile Manipulation
**题目:** HomeRobot:开放语音移动操作

**作者:** Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin S. Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander Clegg, John M. Turner, Zsolt Kira, Manolis Savva, Angel X. Chang, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, Chris Paxton

**Abstract:** HomeRobot (noun): An affordable compliant robot that navigates homes and manipulates a wide range of objects in order to complete everyday tasks. Open-Vocabulary Mobile Manipulation (OVMM) is the problem of picking any object in any unseen environment, and placing it in a commanded location. This is a foundational challenge for robots to be useful assistants in human environments, because it involves tackling sub-problems from across robotics: perception, language understanding, navigation, and manipulation are all essential to OVMM. In addition, integration of the solutions to these sub-problems poses its own substantial challenges. To drive research in this area, we introduce the HomeRobot OVMM benchmark, where an agent navigates household environments to grasp novel objects and place them on target receptacles. HomeRobot has two components: a simulation component, which uses a large and diverse curated object set in new, high-quality multi-room home environments; and a real-world component, providing a software stack for the low-cost Hello Robot Stretch to encourage replication of real-world experiments across labs. We implement both reinforcement learning and heuristic (model-based) baselines and show evidence of sim-to-real transfer. Our baselines achieve a $20%$ success rate in the real world; our experiments identify ways future research work improve performance. See videos on our website: https://home-robot-ovmm.github.io/.

**摘要:** HomeRobot(名词):一种可负担得起的兼容机器人,它可以导航到家园并操纵各种对象,以便完成日常任务。开放式语音移动操纵(OVMM)是选择任何物体在任何无形环境中,并将其置于指挥位置的问题。这是机器人在人类环境中成为有用的助手的一个基本挑战,因为它涉及从整个机器人学中解决次问题:感知、语言理解、导航和操纵都是OVMM的必要条件。HomeRobot有两个组件:一个模拟组件,该组件在新的高质量的多间室内环境中使用大型和多样的托管对象;以及一个真实组件,为低成本的Hello Robot Stretch提供软件堆栈,鼓励在实验室中重复真实世界实验。

**[Paper URL](https://proceedings.mlr.press/v229/yenamandra23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yenamandra23a/yenamandra23a.pdf)** 

# PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play
**题目:** PlayFusion:语言注释游戏通过扩散获取技能

**作者:** Lili Chen, Shikhar Bahl, Deepak Pathak

**Abstract:** Learning from unstructured and uncurated data has become the dominant paradigm for generative approaches in language or vision. Such unstructured and unguided behavior data, commonly known as play, is also easier to collect in robotics but much more difficult to learn from due to its inherently multimodal, noisy, and suboptimal nature. In this paper, we study this problem of learning goal-directed skill policies from unstructured play data which is labeled with language in hindsight. Specifically, we leverage advances in diffusion models to learn a multi-task diffusion model to extract robotic skills from play data. Using a conditional denoising diffusion process in the space of states and actions, we can gracefully handle the complexity and multimodality of play data and generate diverse and interesting robot behaviors. To make diffusion models more useful for skill learning, we encourage robotic agents to acquire a vocabulary of skills by introducing discrete bottlenecks into the conditional behavior generation process. In our experiments, we demonstrate the effectiveness of our approach across a wide variety of environments in both simulation and the real world. Video results available at https://play-fusion.github.io.

**摘要:** 基于不结构和不确定的数据学习已成为语言或视觉生成方法的主导范式。 这种不结构和不引导的行为数据,通常称为游戏,在机器人领域也更容易收集,但由于其固有的多模态、噪声和不优越性,很难从中学习。 本论文研究了从不结构的游戏数据中学习目标导向技能政策这一问题,并将其归纳为后视语言。具体地说,我们利用扩散模型的进步,学习多任务扩散模型,从游戏数据中提取机器人技能。为了使扩散模型更有利于技能学习,我们鼓励机器人代理人通过引入条件行为生成过程的离散瓶颈来获取技能的词汇。在我们的实验中,我们展示了在模拟和现实世界中的广泛环境中我们的方法的有效性。视频结果可于 https://play-fusion.github.io。

**[Paper URL](https://proceedings.mlr.press/v229/chen23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23c/chen23c.pdf)** 

# Shelving, Stacking, Hanging: Relational Pose Diffusion for Multi-modal Rearrangement
**题目:** 架架,架架架,架架架:多模重配关系位扩散

**作者:** Anthony Simeonov, Ankit Goyal, Lucas Manuelli, Yen-Chen Lin, Alina Sarmiento, Alberto Rodriguez Garcia, Pulkit Agrawal, Dieter Fox

**Abstract:** We propose a system for rearranging objects in a scene to achieve a desired object-scene placing relationship, such as a book inserted in an open slot of a bookshelf. The pipeline generalizes to novel geometries, poses, and layouts of both scenes and objects, and is trained from demonstrations to operate directly on 3D point clouds. Our system overcomes challenges associated with the existence of many geometrically-similar rearrangement solutions for a given scene. By leveraging an iterative pose de-noising training procedure, we can fit multi-modal demonstration data and produce multi-modal outputs while remaining precise and accurate. We also show the advantages of conditioning on relevant local geometric features while ignoring irrelevant global structure that harms both generalization and precision. We demonstrate our approach on three distinct rearrangement tasks that require handling multi-modality and generalization over object shape and pose in both simulation and the real world. Project website, code, and videos: https://anthonysimeonov.github.io/rpdiff-multi-modal

**摘要:** 我们提出了一种在场景中重新排列对象的系统,以实现理想的对象-场景配置关系,例如在书架的开槽中插入的书。该管道将对场景和对象的新几何、姿态和布局进行一般化,并由演示训练以直接在3D点云上操作。我们的系统克服了与特定场景存在许多几何相似的重新排列解决方案有关的挑战。通过利用迭代姿态降噪训练程序,我们能够适应多模态演示数据并产生多模态输出,同时保持精确和准确。我们还展示了对相关局部几何特征的条件化的优势,同时忽略了无关紧要的全球结构,从而损害了一般化和精度。我们展示了我们对物体形状的多模态和一般化处理以及在模拟和现实世界中的姿态的三个不同的重新配置任务的方法: https://anthonysimeonov.github.io/rpdiff-multi-modal

**[Paper URL](https://proceedings.mlr.press/v229/simeonov23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/simeonov23a/simeonov23a.pdf)** 

# Learning Efficient Abstract Planning Models that Choose What to Predict
**题目:** 学习有效的抽象规划模型,选择预测什么

**作者:** Nishanth Kumar, Willie McClinton, Rohan Chitnis, Tom Silver, Tomás Lozano-Pérez, Leslie Pack Kaelbling

**Abstract:** An effective approach to solving long-horizon tasks in robotics domains with continuous state and action spaces is bilevel planning, wherein a high-level search over an abstraction of an environment is used to guide low-level decision-making. Recent work has shown how to enable such bilevel planning by learning abstract models in the form of symbolic operators and neural samplers. In this work, we show that existing symbolic operator learning approaches fall short in many robotics domains where a robot’s actions tend to cause a large number of irrelevant changes in the abstract state. This is primarily because they attempt to learn operators that exactly predict all observed changes in the abstract state. To overcome this issue, we propose to learn operators that ‘choose what to predict’ by only modelling changes necessary for abstract planning to achieve specified goals. Experimentally, we show that our approach learns operators that lead to efficient planning across 10 different hybrid robotics domains, including 4 from the challenging BEHAVIOR-100 benchmark, while generalizing to novel initial states, goals, and objects.

**摘要:** 在连续状态和行动空间的机器人领域,解决长期水平任务的有效方法是双层次规划,其中对环境的抽象化进行高层次搜索是指导低层次决策的手段。最近的研究表明,通过学习抽象模型和神经样本器来实现双层次规划。在这个研究中,我们表明,现有的符号操作员学习方法在许多机器人领域中不足,因为机器人的行为往往在抽象状态中造成大量无关紧要的变化。实验表明,我们的方法可以学习操作者,从而在10个不同的混合机器人领域进行有效的规划,包括4个来自挑战性的BEHAVIOR-100基准,同时推广到新的初始状态、目标和对象。

**[Paper URL](https://proceedings.mlr.press/v229/kumar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kumar23a/kumar23a.pdf)** 

# DYNAMO-GRASP: DYNAMics-aware Optimization for GRASP Point Detection in Suction Grippers
**题目:** DYNAMO-GRASP:吸气握手GRASP点检测的动态优化

**作者:** Boling Yang, Soofiyan Atar, Markus Grotz, Byron Boots, Joshua Smith

**Abstract:** In this research, we introduce a novel approach to the challenge of suction grasp point detection. Our method, exploiting the strengths of physics-based simulation and data-driven modeling, accounts for object dynamics during the grasping process, markedly enhancing the robot’s capability to handle previously unseen objects and scenarios in real-world settings. We benchmark DYNAMO-GRASP against established approaches via comprehensive evaluations in both simulated and real-world environments. DYNAMO-GRASP delivers improved grasping performance with greater consistency in both simulated and real-world settings. Remarkably, in real-world tests with challenging scenarios, our method demonstrates a success rate improvement of up to $48%$ over SOTA methods. Demonstrating a strong ability to adapt to complex and unexpected object dynamics, our method offers robust generalization to real-world challenges. The results of this research set the stage for more reliable and resilient robotic manipulation in intricate real-world situations. Experiment videos, dataset, model, and code are available at: https://sites.google.com/view/dynamo-grasp.

**摘要:** 在这一研究中,我们引入了吸取抓取点检测的挑战的新方法。我们的方法,利用基于物理的模拟和数据驱动的建模的优点,在抓取过程中对对象动力学进行分析,明显增强了机器人在现实环境中处理以前未见的对象和场景的能力。我们通过综合评价在模拟环境和现实环境中对已建立的方法进行比较。DYNAMO-GRASP在模拟环境和现实环境中具有更大的一致性,提高了抓取性能。这项研究的成果为复杂现实环境中更可靠、更灵活的机器人操作提供了舞台。实验视频、数据集、模型和代码可于: https://sites.google.com/view/dynamo-grasp。

**[Paper URL](https://proceedings.mlr.press/v229/yang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23a/yang23a.pdf)** 

# HYDRA: Hybrid Robot Actions for Imitation Learning
**题目:** HYDRA:模拟学习的混合机器人动作

**作者:** Suneel Belkhale, Yuchen Cui, Dorsa Sadigh

**Abstract:** Imitation Learning (IL) is a sample efficient paradigm for robot learning using expert demonstrations. However, policies learned through IL suffer from state distribution shift at test time, due to compounding errors in action prediction which lead to previously unseen states. Choosing an action representation for the policy that minimizes this distribution shift is critical in imitation learning. Prior work propose using temporal action abstractions to reduce compounding errors, but they often sacrifice policy dexterity or require domain-specific knowledge. To address these trade-offs, we introduce HYDRA, a method that leverages a hybrid action space with two levels of action abstractions: sparse high-level waypoints and dense low-level actions. HYDRA dynamically switches between action abstractions at test time to enable both coarse and fine-grained control of a robot. In addition, HYDRA employs action relabeling to increase the consistency of actions in the dataset, further reducing distribution shift. HYDRA outperforms prior imitation learning methods by $30-40%$ on seven challenging simulation and real world environments, involving long-horizon tasks in the real world like making coffee and toasting bread. Videos are found on our website: https://tinyurl.com/3mc6793z

**摘要:** 仿真学习(英语:Imitation Learning,缩写为IL)是使用专家演示的机器人学习的有效范式。然而,通过IL学习的策略在测试时受到状态分布变迁的影响,因为在行动预测中出现复合错误,从而导致以前未见的状态。在仿真学习中,选择减少这种分布变迁的政策行动表现是至关重要的。此外,HYDRA还使用动作重播来提高数据集中的动作的一致性,进一步减少分布变化。 HYDRA在7个挑战性模拟和现实世界环境中比以前的仿真学习方法高30-40%,包括做咖啡和烤面包等在现实世界中的长期任务。

**[Paper URL](https://proceedings.mlr.press/v229/belkhale23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/belkhale23a/belkhale23a.pdf)** 

# Embodied Lifelong Learning for Task and Motion Planning
**题目:** 任务及运动规划的内在终身学习

**作者:** Jorge Mendez-Mendez, Leslie Pack Kaelbling, Tomás Lozano-Pérez

**Abstract:** A robot deployed in a home over long stretches of time faces a true lifelong learning problem. As it seeks to provide assistance to its users, the robot should leverage any accumulated experience to improve its own knowledge and proficiency. We formalize this setting with a novel formulation of lifelong learning for task and motion planning (TAMP), which endows our learner with the compositionality of TAMP systems. Exploiting the modularity of TAMP, we develop a mixture of generative models that produces candidate continuous parameters for a planner. Whereas most existing lifelong learning approaches determine a priori how data is shared across various models, our approach learns shared and non-shared models and determines which to use online during planning based on auxiliary tasks that serve as a proxy for each model’s understanding of a state. Our method exhibits substantial improvements (over time and compared to baselines) in planning success on 2D and BEHAVIOR domains.

**摘要:** 由于它寻求向用户提供帮助,机器人应该利用任何积累的经验来提高自己的知识和技能。我们用新的任务和运动规划(TAMP)的终身学习公式形式化这一设置,使我们的学习者具有TAMP系统的组成性。利用TAMP的模块化性,我们开发了一个生成模型的混合物,它为规划者产生候选的连续参数。

**[Paper URL](https://proceedings.mlr.press/v229/mendez-mendez23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mendez-mendez23a/mendez-mendez23a.pdf)** 

# 4D-Former: Multimodal 4D Panoptic Segmentation
**题目:** 4D-Former:多型4D光学分割

**作者:** Ali Athar, Enxu Li, Sergio Casas, Raquel Urtasun

**Abstract:** 4D panoptic segmentation is a challenging but practically useful task that requires every point in a LiDAR point-cloud sequence to be assigned a semantic class label, and individual objects to be segmented and tracked over time. Existing approaches utilize only LiDAR inputs which convey limited information in regions with point sparsity. This problem can, however, be mitigated by utilizing RGB camera images which offer appearance-based information that can reinforce the geometry-based LiDAR features. Motivated by this, we propose 4D-Former: a novel method for 4D panoptic segmentation which leverages both LiDAR and image modalities, and predicts semantic masks as well as temporally consistent object masks for the input point-cloud sequence. We encode semantic classes and objects using a set of concise queries which absorb feature information from both data modalities. Additionally, we propose a learned mechanism to associate object tracks over time which reasons over both appearance and spatial location. We apply 4D-Former to the nuScenes and SemanticKITTI datasets where it achieves state-of-the-art results.

**摘要:** 4D光谱分割是一个挑战性但实用的任务,需要在LiDAR点云序列中每个点分配一个语义类标签,并且在一段时间内分割和跟踪个别对象。现有的方法只使用LiDAR输入,在点稀疏区域传递有限的信息。然而,可以通过使用RGB摄像机图像来缓解这一问题,这些图像提供基于外观的信息,可以增强基于几何的LiDAR特征。此外,我们提出了一种学习的机制,将对象路径随时间关联起来,从而在外观和空间的位置上产生原因。我们应用4D-Former在nuScenes和SemanticKITTI数据集中实现最先进的结果。

**[Paper URL](https://proceedings.mlr.press/v229/athar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/athar23a/athar23a.pdf)** 

# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
**题目:** RT-2:视觉语言行动模型将网络知识转移到机器人控制中

**作者:** Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi, Grecia Salazar, Michael S. Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J. Joshi, Alex Irpan, Brian Ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen, Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, Kehang Han

**Abstract:** We study how vision-language models trained on Internet-scale data can be incorporated directly into end-to-end robotic control to boost generalization and enable emergent semantic reasoning. Our goal is to enable a single end-to-end trained model to both learn to map robot observations to actions and enjoy the benefits of large-scale pretraining on language and vision-language data from the web. To this end, we propose to co-fine-tune state-of-the-art vision-language models on both robotic trajectory data and Internet-scale vision-language tasks, such as visual question answering. In contrast to other approaches, we propose a simple, general recipe to achieve this goal: in order to fit both natural language responses and robotic actions into the same format, we express the actions as text tokens and incorporate them directly into the training set of the model in the same way as natural language tokens. We refer to such category of models as vision-language-action models (VLA) and instantiate an example of such a model, which we call RT-2. Our extensive evaluation (6k evaluation trials) shows that our approach leads to performant robotic policies and enables RT-2 to obtain a range of emergent capabilities from Internet-scale training. This includes significantly improved generalization to novel objects, the ability to interpret commands not present in the robot training data (such as placing an object onto a particular number or icon), and the ability to perform rudimentary reasoning in response to user commands (such as picking up the smallest or largest object, or the one closest to another object). We further show that incorporating chain of thought reasoning allows RT-2 to perform multi-stage semantic reasoning, for example figuring out which object to pick up for use as an improvised hammer (a rock), or which type of drink is best suited for someone who is tired (an energy drink).

**摘要:** 我们研究了如何在网络数据上训练的视觉语言模型能够直接融入到端到端机器人控制中,以促进一般化和启发新的语义推理。我们的目标是使一个单一的端到端训练模型能够学习将机器人观测映射到动作,并从网络上对语言和视觉语言数据进行大规模预训练的好处。为此目的,我们建议在机器人轨迹数据和网络视觉语言任务上共同优化最先进的视觉语言模型,例如视觉问题答复。我们把这种类型的模型称为视觉语言行动模型(VLA)和实例化这种模型,我们称之为RT-2。我们的广泛评估(6k评估试验)表明,我们的方法导致了高效的机器人政策,并使RT-2能够从互联网范围的训练中获得一系列新兴的能力。这包括大幅改善对新对象的一般化,解释机器人训练数据中没有的命令(例如将对象放在特定数目或图标上),以及对用户命令进行初始推理的能力(例如 picking up the smallest or largest object, or the one closest to another object)。我们进一步表明,通过考虑链式思维,RT-2可以进行多阶段语义推理,例如找出哪些对象可以作为即兴锤子(岩石)使用,或者哪些饮料最适合疲劳的人(能量饮料)。

**[Paper URL](https://proceedings.mlr.press/v229/zitkovich23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zitkovich23a/zitkovich23a.pdf)** 

# Seeing-Eye Quadruped Navigation with Force Responsive Locomotion Control
**题目:** 视眼四面航行与力应激运动控制

**作者:** David DeFazio, Eisuke Hirota, Shiqi Zhang

**Abstract:** Seeing-eye robots are very useful tools for guiding visually impaired people, potentially producing a huge societal impact given the low availability and high cost of real guide dogs. Although a few seeing-eye robot systems have already been demonstrated, none considered external tugs from humans, which frequently occur in a real guide dog setting. In this paper, we simultaneously train a locomotion controller that is robust to external tugging forces via Reinforcement Learning (RL), and an external force estimator via supervised learning. The controller ensures stable walking, and the force estimator enables the robot to respond to the external forces from the human. These forces are used to guide the robot to the global goal, which is unknown to the robot, while the robot guides the human around nearby obstacles via a local planner. Experimental results in simulation and on hardware show that our controller is robust to external forces, and our seeing-eye system can accurately detect force direction. We demonstrate our full seeing-eye robot system on a real quadruped robot with a blindfolded human.

**摘要:** 视眼机器人是指导视觉障碍者非常有用的工具,其潜在社会影响是由于真实导狗的低可用性和高成本。虽然已经证明了一些视眼机器人系统,但没有考虑人类外部拖拉,这经常发生在真正的导狗设置中。本文同时训练了通过强化学习(RL)对外部拖拉力强有力的运动控制器和通过监督学习的外部力量估算器。控制器确保了稳定的走路,并且力估算器使机器人能够对来自人类的外部力量作出反应。这些力量被用来引导机器人实现全球目标,而机器人则通过当地规划师引导人绕过附近的障碍。仿真和硬件实验结果表明,我们的控制器对外部力量具有很强的鲁棒性,我们的视觉系统能够准确地检测力向。

**[Paper URL](https://proceedings.mlr.press/v229/defazio23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/defazio23a/defazio23a.pdf)** 

# Waypoint-Based Imitation Learning for Robotic Manipulation
**题目:** 基于方法的仿真学习机器人操纵

**作者:** Lucy Xiaoyang Shi, Archit Sharma, Tony Z. Zhao, Chelsea Finn

**Abstract:** While imitation learning methods have seen a resurgent interest for robotic manipulation, the well-known problem of compounding errors continues to afflict behavioral cloning (BC). Waypoints can help address this problem by reducing the horizon of the learning problem for BC, and thus, the errors compounded over time. However, waypoint labeling is underspecified, and requires additional human supervision. Can we generate waypoints automatically without any additional human supervision? Our key insight is that if a trajectory segment can be approximated by linear motion, the endpoints can be used as waypoints. We propose Automatic Waypoint Extraction (AWE) for imitation learning, a preprocessing module to decompose a demonstration into a minimal set of waypoints which when interpolated linearly can approximate the trajectory up to a specified error threshold. AWE can be combined with any BC algorithm, and we find that AWE can increase the success rate of state-of-the-art algorithms by up to $25%$ in simulation and by $4-28%$ on real-world bimanual manipulation tasks, reducing the decision making horizon by up to a factor of 10. Videos and code are available at https://lucys0.github.io/awe/.

**摘要:** 仿真学习方法对机器人操纵有重新出现的兴趣,但已知的复合误差问题继续影响行为克隆(BC)。方法点可以减少 BC学习问题水平,从而解决这一问题,从而使误差随着时间的推移加重。然而,方法点标签的标识不足,需要额外的人力监督。我们是否能够自动生成方法点,而不需额外的人力监督?我们的关键了解是,如果 trajectory segment can be approximated by linear motion, the endpoints can be used as waypoints。我们建议仿真学习的自动方法点抽取(AWE)是一个预处理模块,将演示分解为最小一套方法点,在线性插值时可以逼近 trajectory to a specified error threshold。AWE可以与任何BC算法结合起来,我们发现AWE可以在模拟中提高最先进的算法的成功率,在真实双人操纵任务中增加到25%$和4-28%$,从而减少决策范围的10倍。

**[Paper URL](https://proceedings.mlr.press/v229/shi23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shi23b/shi23b.pdf)** 

# Multi-Resolution Sensing for Real-Time Control with Vision-Language Models
**题目:** 基于视觉语言模型的实时控制多分辨率感知

**作者:** Saumya Saxena, Mohit Sharma, Oliver Kroemer

**Abstract:** Leveraging sensing modalities across diverse spatial and temporal resolutions can improve performance of robotic manipulation tasks. Multi-spatial resolution sensing provides hierarchical information captured at different spatial scales and enables both coarse and precise motions. Simultaneously multi-temporal resolution sensing enables the agent to exhibit high reactivity and real-time control. In this work, we propose a framework for learning generalizable language-conditioned multi-task policies that utilize sensing at different spatial and temporal resolutions using networks of varying capacities to effectively perform real time control of precise and reactive tasks. We leverage off-the-shelf pretrained vision-language models to operate on low-frequency global features along with small non-pretrained models to adapt to high frequency local feedback. Through extensive experiments in 3 domains (coarse, precise and dynamic manipulation tasks), we show that our approach significantly improves ($2\times$ on average) over recent multi-task baselines. Further, our approach generalizes well to visual and geometric variations in target objects and to varying interaction forces.

**摘要:** 利用不同空间和时间分辨率的感知模式可以提高机器人操纵任务的性能。多空间分辨率感知提供在不同空间尺度的层次信息,并可实现粗糙和精确的运动。同时,多时分辨率感知可使代理显示高反应性和实时控制。本文提出了一种可推广的语言条件下的多任务策略框架,利用不同容量的网络来利用不同空间和时间分辨率的感知来有效执行精确和反应任务的实时控制。通过在3个领域(粗糙、精确和动态操作任务)进行广泛的实验,我们证明,我们的方法在最近的多任务基线上显著改善(平均$2\times$)。此外,我们的方法对目标对象的视觉和几何变化以及相互作用力的变化也进行了一般化。

**[Paper URL](https://proceedings.mlr.press/v229/saxena23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/saxena23a/saxena23a.pdf)** 

# SCALE: Causal Learning and Discovery of Robot Manipulation Skills using Simulation
**题目:** SCALE:基于仿真的机器人操纵技能的诱因学习与发现

**作者:** Tabitha Edith Lee, Shivam Vats, Siddharth Girdhar, Oliver Kroemer

**Abstract:** We propose SCALE, an approach for discovering and learning a diverse set of interpretable robot skills from a limited dataset. Rather than learning a single skill which may fail to capture all the modes in the data, we first identify the different modes via causal reasoning and learn a separate skill for each of them. Our main insight is to associate each mode with a unique set of causally relevant context variables that are discovered by performing causal interventions in simulation. This enables data partitioning based on the causal processes that generated the data, and then compressed skills that ignore the irrelevant variables can be trained. We model each robot skill as a Regional Compressed Option, which extends the options framework by associating a causal process and its relevant variables with the option. Modeled as the skill Data Generating Region, each causal process is local in nature and hence valid over only a subset of the context space. We demonstrate our approach for two representative manipulation tasks: block stacking and peg-in-hole insertion under uncertainty. Our experiments show that our approach yields diverse skills that are compact, robust to domain shifts, and suitable for sim-to-real transfer.

**摘要:** 我们提出了SCALE,一种从有限的数据集中发现和学习多种可解释机器人技能的方法,而不是学习一个单一技能,它可能无法捕捉数据中的所有模式,我们首先通过因果推理识别不同的模式,并学习每个模式的单独技能。我们的主要洞察是将每个模式与由模拟中进行因果干预发现的因果相关上下文变量独特的集合联系起来。这允许基于生成数据的因果过程进行数据分割,然后可以训练忽略不相关的变量的压缩技能。我们展示了我们对两个代表性的操作任务的处理方法:块堆叠和不确定情况下的凹槽插入。我们的实验表明,我们的处理方法能产生各种技能,这些技能 compact, robust to domain shifts, and suitable for sim-to-real transfer。

**[Paper URL](https://proceedings.mlr.press/v229/lee23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lee23b/lee23b.pdf)** 

# Learning Robot Manipulation from Cross-Morphology Demonstration
**题目:** 从交叉形态学演示中学习机器人操纵

**作者:** Gautam Salhotra, I-Chun Arthur Liu, Gaurav S. Sukhatme

**Abstract:** Some Learning from Demonstrations (LfD) methods handle small mismatches in the action spaces of the teacher and student. Here we address the casewhere the teacher’s morphology is substantially different from that of the student. Our framework, Morphological Adaptation in Imitation Learning (MAIL), bridges this gap allowing us to train an agent from demonstrations by other agents with significantly different morphologies. MAIL learns from suboptimal demonstrations, so long as they provide some guidance towards a desired solution. We demonstrate MAIL on manipulation tasks with rigid and deformable objects including 3D cloth manipulation interacting with rigid obstacles. We train a visual control policy for a robot with one end-effector using demonstrations from a simulated agent with two end-effectors. MAIL shows up to $24%$ improvement in a normalized performance metric over LfD and non-LfD baselines. It is deployed to a real Franka Panda robot, handles multiple variations in properties for objects (size, rotation, translation), and cloth-specific properties (color, thickness, size, material).

**摘要:** 一些从演示(LfD)方法学习的方法处理在教师和学生行动空间中的小不一致问题。这里我们讨论了教师的形态与学生形态有很大不同的情况。我们的框架,模仿学习的形态适应(MAIL)将这一差距桥梁,允许我们训练其他具有显著不同形态的代理人演示的代理人。MAIL从亚最佳演示中学习,只要它们提供一些向理想的解决方案方向的指导。我们演示MAIL在操作任务中使用刚形和变形的对象,包括与刚形障碍相互作用的3D布操作。它被部署到一个真正的Franka Panda机器人,处理对象的属性(大小、旋转、翻译)和布的特定属性(颜色、厚度、大小、材料)的多种变异。

**[Paper URL](https://proceedings.mlr.press/v229/salhotra23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/salhotra23a/salhotra23a.pdf)** 

# Synthesizing Navigation Abstractions for Planning with Portable Manipulation Skills
**题目:** 利用可移动操作技能进行规划的导航归纳

**作者:** Eric Rosen, Steven James, Sergio Orozco, Vedant Gupta, Max Merlin, Stefanie Tellex, George Konidaris

**Abstract:** We address the problem of efficiently learning high-level abstractions for task-level robot planning. Existing approaches require large amounts of data and fail to generalize learned abstractions to new environments. To address this, we propose to exploit the independence between spatial and non-spatial state variables in the preconditions of manipulation and navigation skills, mirroring the manipulation-navigation split in robotics research. Given a collection of portable manipulation abstractions (i.e., object-centric manipulation skills paired with matching symbolic representations), we derive an algorithm to automatically generate navigation abstractions that support mobile manipulation planning in a novel environment. We apply our approach to simulated data in AI2Thor and on real robot hardware with a coffee preparation task, efficiently generating plannable representations for mobile manipulators in just a few minutes of robot time, significantly outperforming state-of-the-art baselines.

**摘要:** 针对任务级机器人规划中高效学习高层次抽象问题,现有的方法需要大量数据,并不能将学习抽象推广到新环境。为此,我们提议利用空间和非空间状态变量之间的独立性在操作和导航技能的前提条件中,以反映机器人研究中的操作导航分割。我们应用了AI2Thor的模拟数据和实物机器人硬件的咖啡准备任务,在机器人时间的几分钟内有效地生成移动操作器可预想的显示,大大超过了最先进的基线。

**[Paper URL](https://proceedings.mlr.press/v229/rosen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rosen23a/rosen23a.pdf)** 

# Transforming a Quadruped into a Guide Robot for the Visually Impaired: Formalizing Wayfinding, Interaction Modeling, and Safety Mechanism
**题目:** 将四角形机器人转化为视觉障碍者的指导机器人:形成路径定位、交互模型和安全机制

**作者:** J. Taery Kim, Wenhao Yu, Yash Kothari, Bruce Walker, Jie Tan, Greg Turk, Sehoon Ha

**Abstract:** This paper explores the principles for transforming a quadrupedal robot into a guide robot for individuals with visual impairments. A guide robot has great potential to resolve the limited availability of guide animals that are accessible to only two to three percent of the potential blind or visually impaired (BVI) users. To build a successful guide robot, our paper explores three key topics: (1) formalizing the navigation mechanism of a guide dog and a human, (2) developing a data-driven model of their interaction, and (3) improving user safety. First, we formalize the wayfinding task of the human-guide robot team using Markov Decision Processes based on the literature and interviews. Then we collect real human-robot interaction data from three visually impaired and six sighted people and develop an interaction model called the "Delayed Harness" to effectively simulate the navigation behaviors of the team. Additionally, we introduce an action shielding mechanism to enhance user safety by predicting and filtering out dangerous actions. We evaluate the developed interaction model and the safety mechanism in simulation, which greatly reduce the prediction errors and the number of collisions, respectively. We also demonstrate the integrated system on an AlienGo robot with a rigid harness, by guiding users over 100+ meter trajectories.

**摘要:** 本文探讨了四肢机器人转化为视觉障碍者导游机器人的原理。导游机器人具有解决盲人或视觉障碍者(BVI)使用者只有2%或3%的可访问导游动物的有限可用性潜力。为了构建成功的导游机器人,本文探讨了三个关键问题:(一)正式化导游狗和人类的导航机制;(二)开发一种基于数据的相互作用模型;(三)提高使用者的安全。首先,根据文献和访谈,采用马可夫决策过程形式化导游机器人团队的路径搜索任务。然后,从三个视觉障碍者和六个视觉障碍者中收集了真实人-机器人交互数据,并开发了一个名为“延迟导游工具”的交互模型,以有效模拟团队的导航行为。此外,我们介绍了一种通过预测和筛选危险行为提高用户安全性的动作保护机制。我们评价了开发的交互模型和仿真中的安全机制,分别大大降低了预测误差和碰撞数目。

**[Paper URL](https://proceedings.mlr.press/v229/kim23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kim23c/kim23c.pdf)** 

# A Bayesian Approach to Robust Inverse Reinforcement Learning
**题目:** 贝叶斯主义对鲁棒反向强化学习的方法

**作者:** Ran Wei, Siliang Zeng, Chenliang Li, Alfredo Garcia, Anthony D McDonald, Mingyi Hong

**Abstract:** We consider a Bayesian approach to offline model-based inverse reinforcement learning (IRL). The proposed framework differs from existing offline model-based IRL approaches by performing simultaneous estimation of the expert’s reward function and subjective model of environment dynamics. We make use of a class of prior distributions which parameterizes how accurate the expert’s model of the environment is to develop efficient algorithms to estimate the expert’s reward and subjective dynamics in high-dimensional settings. Our analysis reveals a novel insight that the estimated policy exhibits robust performance when the expert is believed (a priori) to have a highly accurate model of the environment. We verify this observation in the MuJoCo environments and show that our algorithms outperform state-of-the-art offline IRL algorithms.

**摘要:** 我们考虑了基于非线性模型的逆强化学习(IRL)的贝叶斯方法。该方法与现有基于非线性模型的IRL方法不同,通过对专家的奖励函数和环境动力学的主体模型进行同时估计。我们利用一个参数化了专家环境模型的准确性,以开发高维环境中评估专家的奖励和主体动力学的高效算法。我们的分析揭示了估计政策在专家被认为具有高精度的环境模型时表现出强有力的性能。

**[Paper URL](https://proceedings.mlr.press/v229/wei23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wei23a/wei23a.pdf)** 

# ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation
**题目:** ChainedDiffuser:统一机器人操纵的推理扩散和关键定位预测

**作者:** Zhou Xian, Nikolaos Gkanatsios, Theophile Gervet, Tsung-Wei Ke, Katerina Fragkiadaki

**Abstract:** We present ChainedDiffuser, a policy architecture that unifies action keypose prediction and trajectory diffusion generation for learning robot manipulation from demonstrations. Our main innovation is to use a global transformer-based action predictor to predict actions at keyframes, a task that requires multi- modal semantic scene understanding, and to use a local trajectory diffuser to predict trajectory segments that connect predicted macro-actions. ChainedDiffuser sets a new record on established manipulation benchmarks, and outperforms both state-of-the-art keypose (macro-action) prediction models that use motion plan- ners for trajectory prediction, and trajectory diffusion policies that do not predict keyframe macro-actions. We conduct experiments in both simulated and real-world environments and demonstrate ChainedDiffuser’s ability to solve a wide range of manipulation tasks involving interactions with diverse objects.

**摘要:** 我们介绍ChainedDiffuser,一种基于变换器的全球行动预测器用于预测关键帧的动作,该任务需要多模态语义场景的理解,并使用局部轨迹扩散器来预测与预测的宏观动作相连的轨迹段。ChainedDiffuser在已建立的操纵基准上建立了新纪录,并且超越了使用运动规划者来预测轨迹的最先进的关键位置(宏观动作)预测模型,以及没有预测关键帧的宏观动作的轨迹扩散政策。我们在模拟和现实环境中进行了实验,并展示了ChainedDiffuser解决与不同对象交互的广泛操纵任务的能力。

**[Paper URL](https://proceedings.mlr.press/v229/xian23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xian23a/xian23a.pdf)** 

# IIFL: Implicit Interactive Fleet Learning from Heterogeneous Human Supervisors
**题目:** IIFL:隐性互动舰队从异种人类监督者学习

**作者:** Gaurav Datta, Ryan Hoque, Anrui Gu, Eugen Solowjow, Ken Goldberg

**Abstract:** Imitation learning has been applied to a range of robotic tasks, but can struggle when robots encounter edge cases that are not represented in the training data (i.e., distribution shift). Interactive fleet learning (IFL) mitigates distribution shift by allowing robots to access remote human supervisors during task execution and learn from them over time, but different supervisors may demonstrate the task in different ways. Recent work proposes Implicit Behavior Cloning (IBC), which is able to represent multimodal demonstrations using energy-based models (EBMs). In this work, we propose Implicit Interactive Fleet Learning (IIFL), an algorithm that builds on IBC for interactive imitation learning from multiple heterogeneous human supervisors. A key insight in IIFL is a novel approach for uncertainty quantification in EBMs using Jeffreys divergence. While IIFL is more computationally expensive than explicit methods, results suggest that IIFL achieves a 2.8x higher success rate in simulation experiments and a 4.5x higher return on human effort in a physical block pushing task over (Explicit) IFL, IBC, and other baselines.

**摘要:** 仿真学习已经应用于多种机器人任务,但当机器人遇到在训练数据中没有显示的边缘事件时(即分布变迁)时,它会遇到困难。交互式舰队学习(IFL)通过允许机器人在任务执行过程中访问远程人类监督者,并从他们学习,可以缓解分布变迁,但不同的监督者可能以不同的方式显示任务。最近的工作提出了隐形行为克隆(IBC),它能够代表使用能量模型(EBM)的多模态演示。尽管IIFL比显式方法的计算成本高,但结果表明,IIFL在模拟实验中获得2.8倍的成功率,并在物理块推力任务中获得4.5倍的人力投入回报。

**[Paper URL](https://proceedings.mlr.press/v229/datta23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/datta23a/datta23a.pdf)** 

# CAT: Closed-loop Adversarial Training for Safe End-to-End Driving
**题目:** CAT:安全终点驾驶的闭环敌对训练

**作者:** Linrui Zhang, Zhenghao Peng, Quanyi Li, Bolei Zhou

**Abstract:** Driving safety is a top priority for autonomous vehicles. Orthogonal to prior work handling accident-prone traffic events by algorithm designs at the policy level, we investigate a Closed-loop Adversarial Training (CAT) framework for safe end-to-end driving in this paper through the lens of environment augmentation. CAT aims to continuously improve the safety of driving agents by training the agent on safety-critical scenarios that are dynamically generated over time. A novel resampling technique is developed to turn log-replay real-world driving scenarios into safety-critical ones via probabilistic factorization, where the adversarial traffic generation is modeled as the multiplication of standard motion prediction sub-problems. Consequently, CAT can launch more efficient physical attacks compared to existing safety-critical scenario generation methods and yields a significantly less computational cost in the iterative learning pipeline. We incorporate CAT into the MetaDrive simulator and validate our approach on hundreds of driving scenarios imported from real-world driving datasets. Experimental results demonstrate that CAT can effectively generate adversarial scenarios countering the agent being trained. After training, the agent can achieve superior driving safety in both log-replay and safety-critical traffic scenarios on the held-out test set. Code and data are available at: https://metadriverse.github.io/cat

**摘要:** 本文通过环境增强的视角,对安全端到端驾驶的闭环敌对训练框架进行了研究。该框架旨在通过对随时间动态生成的安全关键场景进行训练,持续提高驾驶人员的安全。一种新型的重新模拟技术被开发,通过概率因素化将日志重演的现实驾驶场景转化为安全关键场景,使得敌对的交通生成被建模为标准运动预测子问题的乘法。因此,CTC可以比现有的安全关键场景生成方法进行更有效的物理攻击,并在迭代学习管道中产生更低的计算成本。我们将CAT集成到MetaDrive模拟器中,并验证了从真实世界驾驶数据集进口的数百个驾驶场景中的方法。实验结果表明,CAT能够有效地产生对抗代理进行训练的敌对场景。经过训练后,代理可以在记录重演和安全关键的交通场景中达到较高的驾驶安全性。代码和数据可于: https://metadriverse.github.io/cat

**[Paper URL](https://proceedings.mlr.press/v229/zhang23g.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23g/zhang23g.pdf)** 

# Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control
**题目:** 神经图控制屏障功能导引分布式碰撞-避免多剂控制

**作者:** Songyuan Zhang, Kunal Garg, Chuchu Fan

**Abstract:** We consider the problem of designing distributed collision-avoidance multi-agent control in large-scale environments with potentially moving obstacles, where a large number of agents are required to maintain safety using only local information and reach their goals. This paper addresses the problem of collision avoidance, scalability, and generalizability by introducing graph control barrier functions (GCBFs) for distributed control. The newly introduced GCBF is based on the well-established CBF theory for safety guarantees but utilizes a graph structure for scalable and generalizable decentralized control. We use graph neural networks to learn both neural a GCBF certificate and distributed control. We also extend the framework from handling state-based models to directly taking point clouds from LiDAR for more practical robotics settings. We demonstrated the efficacy of GCBF in a variety of numerical experiments, where the number, density, and traveling distance of agents, as well as the number of unseen and uncontrolled obstacles increase. Empirical results show that GCBF outperforms leading methods such as MAPPO and multi-agent distributed CBF (MDCBF). Trained with only $16$ agents, GCBF can achieve up to $3$ times improvement of success rate (agents reach goals and never encountered in any collisions) on $<500$ agents, and still maintain more than $50%$ success rates for $>\!1000$ agents when other methods completely fail.

**摘要:** 本文通过引入分布式控制的图控屏障函数(GCBF)来解决分布式控制的碰撞、可扩展性和可推广性问题。新引入的GCBF基于安全保证的已建立的CBF理论,但利用图形结构来实现可扩展和可推广的分散控制。我们使用图形神经网络学习图形神经证书和分布式控制,并从处理状态模型到直接从LiDAR获取点云来实现更实用的机器人设置。通过数值实验,验证了GCBF的有效性,其中代理的数量、密度、 traveling distance以及无视和无控障碍的数量增加。实验结果表明,GCBF比MAPPO和多代理分发CBF(MDCBF)等领先方法高出。经训练,只有$16的代理,GCBF可以在$<500的代理中提高成功率达$3倍(代理达到目标并从未遇到任何碰撞)、在其他方法完全失败时仍保持50%以上成功率的$>\!1000代理。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23h.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23h/zhang23h.pdf)** 

# STERLING: Self-Supervised Terrain Representation Learning from Unconstrained Robot Experience
**题目:** STERLING:自控地表表现从无约束机器人经验中学习

**作者:** Haresh Karnan, Elvin Yang, Daniel Farkash, Garrett Warnell, Joydeep Biswas, Peter Stone

**Abstract:** Terrain awareness, i.e., the ability to identify and distinguish different types of terrain, is a critical ability that robots must have to succeed at autonomous off-road navigation. Current approaches that provide robots with this awareness either rely on labeled data which is expensive to collect, engineered features and cost functions that may not generalize, or expert human demonstrations which may not be available. Towards endowing robots with terrain awareness without these limitations, we introduce Self-supervised TErrain Representation LearnING (STERLING), a novel approach for learning terrain representations that relies solely on easy-to-collect, unconstrained (e.g., non-expert), and unlabelled robot experience, with no additional constraints on data collection. STERLING employs a novel multi-modal self-supervision objective through non-contrastive representation learning to learn relevant terrain representations for terrain-aware navigation. Through physical robot experiments in off-road environments, we evaluate STERLING features on the task of preference-aligned visual navigation and find that STERLING features perform on par with fully-supervised approaches and outperform other state-of-the-art methods with respect to preference alignment. Additionally, we perform a large-scale experiment of autonomously hiking a 3-mile long trail which STERLING completes successfully with only two manual interventions, demonstrating its robustness to real-world off-road conditions.

**摘要:** 地形认知,即识别和区分不同类型的地形的能力,是机器人必须在自主外路导航中取得成功的关键能力。目前提供这种认知的机器人的方法要么依靠收集昂贵的标签数据,要么采用工程特征和成本函数,可能不会推广,或者可能没有专家的人类演示。为了使机器人无这些限制地具有地形认知,我们引入了自我监督的特雷恩表示学习(STERLING),一种全新的学习地形表示的方法,仅依靠易于收集、无约束(例如非专家)和无标签的机器人经验,无数据收集的额外约束。通过在野外环境中进行物理机器人实验,我们对STERLING特征在偏好定位视觉导航任务中进行评价,发现STERLING特征在偏好定位方面与完全监督的方法相等,并且在偏好定位方面比其他最先进的方法高。此外,我们还进行了一次大规模的自走3英里长的行道实验,该行道由STERLING仅通过两项手动干预完成,证明了它在野外环境中具有很强的鲁棒性。

**[Paper URL](https://proceedings.mlr.press/v229/karnan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/karnan23a/karnan23a.pdf)** 

# Towards General Single-Utensil Food Acquisition with Human-Informed Actions
**题目:** 以人为本的全面单用食品采购

**作者:** Ethan Kroll Gordon, Amal Nanavati, Ramya Challa, Bernie Hao Zhu, Taylor Annette Kessler Faulkner, Siddhartha Srinivasa

**Abstract:** Food acquisition with common general-purpose utensils is a necessary component of robot applications like in-home assistive feeding. Learning acquisition policies in this space is difficult in part because any model will need to contend with extensive state and actions spaces. Food is extremely diverse and generally difficult to simulate, and acquisition actions like skewers, scoops, wiggles, and twirls can be parameterized in myriad ways. However, food’s visual diversity can belie a degree of physical homogeneity, and many foods allow flexibility in how they are acquired. Due to these facts, our key insight is that a small subset of actions is sufficient to acquire a wide variety of food items. In this work, we present a methodology for identifying such a subset from limited human trajectory data. We first develop an over-parameterized action space of robot acquisition trajectories that capture the variety of human food acquisition technique. By mapping human trajectories into this space and clustering, we construct a discrete set of 11 actions. We demonstrate that this set is capable of acquiring a variety of food items with $\geq80%$ success rate, a rate that users have said is sufficient for in-home robot-assisted feeding. Furthermore, since this set is so small, we also show that we can use online learning to determine a sufficiently optimal action for a previously-unseen food item over the course of a single meal.

**摘要:** 使用通用工具获取食物是机器人应用中必不可少的组成部分,如家庭辅助喂食。在这一领域学习获取政策是困难的,因为任何模型都需要与广泛的状态和行动空间作斗争。食物是极其多样的,一般很难模拟的,获取行动如 skewers, scoops, wiggles, and twirls可以以多种方式参数化。然而,食品的视觉多样性可能误导某种程度的物理均匀性,许多食品允许它们获得的灵活性。通过将人类轨迹映射到这个空间和聚类中,我们构建了一个11个行动的离散集合。我们证明,这个集合能够获得各种食品物品的成功率$\geq80%$,这个比率已经被用户认为足够用于家庭机器人辅助喂食。

**[Paper URL](https://proceedings.mlr.press/v229/gordon23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gordon23a/gordon23a.pdf)** 

# ScalableMap: Scalable Map Learning for Online Long-Range Vectorized HD Map Construction
**题目:** ScalableMap:在线长距离向量化HD地图建构的可扩展地图学习

**作者:** Jingyi Yu, Zizhao Zhang, Shengfu Xia, Jizhang Sang

**Abstract:** We propose a novel end-to-end pipeline for online long-range vectorized high-definition (HD) map construction using on-board camera sensors. The vectorized representation of HD maps, employing polylines and polygons to represent map elements, is widely used by downstream tasks. However, previous schemes designed with reference to dynamic object detection overlook the structural constraints within linear map elements, resulting in performance degradation in long-range scenarios. In this paper, we exploit the properties of map elements to improve the performance of map construction. We extract more accurate bird’s eye view (BEV) features guided by their linear structure, and then propose a hierarchical sparse map representation to further leverage the scalability of vectorized map elements, and design a progressive decoding mechanism and a supervision strategy based on this representation. Our approach, ScalableMap, demonstrates superior performance on the nuScenes dataset, especially in long-range scenarios, surpassing previous state-of-the-art model by 6.5 mAP while achieving 18.3 FPS.

**摘要:** 本文提出了一种基于内置摄像机传感器的在线长距离矢量化高清晰度(HD)地图建构的新型端到端管道。采用多边形和多边形来表示地图元素的HD地图矢量化建构,广泛应用于下游任务。然而,基于动态对象检测设计的以往方案忽略了线性地图元素内部的结构约束,导致长距离场景中的性能下降。本文利用地图元素的特性来提高地图建构性能。我们的方法,ScaleableMap,证明了nuScenes数据集的优越性能,特别是在远距离场景中,超过了先前的最先进的模型6.5mAP,同时达到18.3 FPS。

**[Paper URL](https://proceedings.mlr.press/v229/yu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yu23b/yu23b.pdf)** 

# Tuning Legged Locomotion Controllers via Safe Bayesian Optimization
**题目:** 通过安全贝伊西亚优化调整 Legged Locomotion控制器

**作者:** Daniel Widmer, Dongho Kang, Bhavya Sukhija, Jonas Hübotter, Andreas Krause, Stelian Coros

**Abstract:** This paper presents a data-driven strategy to streamline the deployment of model-based controllers in legged robotic hardware platforms. Our approach leverages a model-free safe learning algorithm to automate the tuning of control gains, addressing the mismatch between the simplified model used in the control formulation and the real system. This method substantially mitigates the risk of hazardous interactions with the robot by sample-efficiently optimizing parameters within a probably safe region. Additionally, we extend the applicability of our approach to incorporate the different gait parameters as contexts, leading to a safe, sample-efficient exploration algorithm capable of tuning a motion controller for diverse gait patterns. We validate our method through simulation and hardware experiments, where we demonstrate that the algorithm obtains superior performance on tuning a model-based motion controller for multiple gaits safely.

**摘要:** 本文提出了一种基于数据的策略,使基于模型的控制器在腿部机器人硬件平台的部署合理化。该方法利用无模型安全学习算法实现控制增益的自动化,解决控制公式和实际系统中使用的简化模型之间的不匹配问题。该方法通过在可能安全的区域内有效优化参数,大大缓解了与机器人的危险交互风险。此外,我们扩展了该方法的适用性,将不同的运动参数纳入上下文,从而实现一种安全、高效的实验算法,能够对多种运动模式的运动控制器进行调谐。我们通过仿真和硬件实验验证了该方法,证明该算法在调谐多个运动模式的运动控制器时具有较高的性能。

**[Paper URL](https://proceedings.mlr.press/v229/widmer23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/widmer23a/widmer23a.pdf)** 

# TraCo: Learning Virtual Traffic Coordinator for Cooperation with Multi-Agent Reinforcement Learning
**题目:** 特拉科:与多机构加强学习合作学习的虚拟交通协调员

**作者:** Weiwei Liu, Wei Jing, lingping Gao, Ke Guo, Gang Xu, Yong Liu

**Abstract:** Multi-agent reinforcement learning (MARL) has emerged as a popular technique in diverse domains due to its ability to automate system controller design and facilitate continuous intelligence learning. For instance, traffic flow is often trained with MARL to enable intelligent simulations for autonomous driving. However, The existing MARL algorithm only characterizes the relative degree of each agent’s contribution to the team, and cannot express the contribution that the team needs from the agent. Especially in the field of autonomous driving, the team changes over time, and the agent needs to act directly according to the needs of the team. To address these limitations, we propose an innovative method inspired by realistic traffic coordinators called the Traffic Coordinator Network (TraCo). Our approach leverages a combination of cross-attention and counterfactual advantage function, allowing us to extract distinctive characteristics of domain agents and accurately quantify the contribution that a team needs from an agent. Through experiments conducted on four traffic tasks, we demonstrate that our method outperforms existing approaches, yielding superior performance. Furthermore, our approach enables the emergence of rich and diverse social behaviors among vehicles within the traffic flow.

**摘要:** 多代理强化学习(英语:Multi-Agent Reinforcement Learning,简称MARL)由于其能够自动化系统控制器设计和促进持续智能学习的能力,在不同领域成为一种流行的技术。例如,交通流量经常用MARL进行训练,以便实现自主驾驶的智能仿真。然而,现有的MARL算法只描述了每个代理人对团队的贡献的相对程度,不能表达团队需要的代理人贡献。特别是在自主驾驶领域,团队随着时间的推移,代理人需要根据团队需要采取直接行动。通过对四个交通任务进行的实验,我们证明了我们的方法比现有方法优越,具有较高的性能。此外,我们的方法可使车辆在交通流量中产生丰富多样的社会行为。

**[Paper URL](https://proceedings.mlr.press/v229/liu23f.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23f/liu23f.pdf)** 

# Enabling Efficient, Reliable Real-World Reinforcement Learning with Approximate Physics-Based Models
**题目:** 利用近似基于物理模型的高效可靠的实世界强化学习

**作者:** Tyler Westenbroek, Jacob Levy, David Fridovich-Keil

**Abstract:** We focus on developing efficient and reliable policy optimization strategies for robot learning with real-world data.  In recent years, policy gradient methods have emerged as a promising paradigm for training control policies in simulation.  However, these approaches often remain too data inefficient or unreliable to train on real robotic hardware. In this paper we introduce a novel policy gradient-based policy optimization framework which systematically leverages a (possibly highly simplified) first-principles model and enables learning precise control policies with limited amounts of real-world data. Our approach $1)$ uses the derivatives of the model to produce sample-efficient estimates of the policy gradient and $2)$ uses the model to design a low-level tracking controller, which is embedded in the policy class. Theoretical analysis provides insight into how the presence of this feedback controller addresses overcomes key limitations of stand-alone policy gradient methods, while hardware experiments with a small car and quadruped demonstrate that our approach can learn precise control strategies reliably and with only minutes of real-world data.

**摘要:** 本文主要介绍一种基于实际数据的新型策略梯度优化框架,系统地利用一个(可能高度简化)初始原理模型,并利用有限的实物数据学习精确的控制策略。我们的方法 $1)$使用模型的衍生物来生成模型梯度的样本效率估计和 $2)$使用模型设计一个低级跟踪控制器,该模型被嵌入到政策类中。理论分析提供了对该反馈控制器的存在如何解决独立政策梯度方法的关键限制的洞察力,而硬件实验在小型车和四轮车上证明,我们的方法可以可靠地学习精确的控制策略,并且只有几分钟的实际数据。

**[Paper URL](https://proceedings.mlr.press/v229/westenbroek23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/westenbroek23a/westenbroek23a.pdf)** 

# Large Language Models as General Pattern Machines
**题目:** 大型语言模型作为通用模式机器

**作者:** Suvir Mirchandani, Fei Xia, Pete Florence, Brian Ichter, Danny Driess, Montserrat Gonzalez Arenas, Kanishka Rao, Dorsa Sadigh, Andy Zeng

**Abstract:** We observe that pre-trained large language models (LLMs) are capable of autoregressively completing complex token sequences–from arbitrary ones procedurally generated by probabilistic context-free grammars (PCFG), to more rich spatial patterns found in the Abstraction and Reasoning Corpus (ARC), a general AI benchmark, prompted in the style of ASCII art. Surprisingly, pattern completion proficiency can be partially retained even when the sequences are expressed using tokens randomly sampled from the vocabulary. These results suggest that without any additional training, LLMs can serve as general sequence modelers, driven by in-context learning. In this work, we investigate how these zero-shot capabilities may be applied to problems in robotics–from extrapolating sequences of numbers that represent states over time to complete simple motions, to least-to-most prompting of reward-conditioned trajectories that can discover and represent closed-loop policies (e.g., a stabilizing controller for CartPole). While difficult to deploy today for real systems due to latency, context size limitations, and compute costs, the approach of using LLMs to drive low-level control may provide an exciting glimpse into how the patterns among words could be transferred to actions.

**摘要:** 我们观察到预训练的大型语言模型(LLMs)能够自动完成复杂的符号序列——从随机过程生成的概率语文自由语法(PCFG)到基于ASCII艺术风格的抽象和推理 Corpus(ARC)中发现的更丰富的空间模式,即使序列被使用随机从词汇中抽取的符号表达时,也能部分地保持模式完成能力。在这项研究中,我们研究了这些零射击能力如何应用于机器人领域的问题 — — 从对代表时间状态的数序推导到完成简单的运动,到能够发现和代表闭环政策(例如,卡特波尔的稳定控制器)的最小到最小的奖励条件轨迹的提示。

**[Paper URL](https://proceedings.mlr.press/v229/mirchandani23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mirchandani23a/mirchandani23a.pdf)** 

# One-shot Imitation Learning via Interaction Warping
**题目:** 基于交互变换的单击仿真学习

**作者:** Ondrej Biza, Skye Thompson, Kishore Reddy Pagidi, Abhinav Kumar, Elise van der Pol, Robin Walters, Thomas Kipf, Jan-Willem van de Meent, Lawson L. S. Wong, Robert Platt

**Abstract:** Learning robot policies from few demonstrations is crucial in open-ended applications. We propose a new method, Interaction Warping, for one-shot learning SE(3) robotic manipulation policies. We infer the 3D mesh of each object in the environment using shape warping, a technique for aligning point clouds across object instances. Then, we represent manipulation actions as keypoints on objects, which can be warped with the shape of the object. We show successful one-shot imitation learning on three simulated and real-world object re-arrangement tasks. We also demonstrate the ability of our method to predict object meshes and robot grasps in the wild. Webpage: https://shapewarping.github.io.

**摘要:** 在开放应用中,从少数演示中学习机器人政策是至关重要的。我们提出了一种新的方法,即交互变形,用于一击学习SE(3)机器人操纵政策。我们利用形状变形来推导在环境中每个对象的3D网格,这是在对象实例中调整点云的技术。然后,我们代表操纵行动为对象的关键点,可以与对象的形状变形。我们在三个模拟和现实对象重新配置任务上展示了成功的一击仿真学习。

**[Paper URL](https://proceedings.mlr.press/v229/biza23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/biza23a/biza23a.pdf)** 

# Learning to See Physical Properties with Active Sensing Motor Policies
**题目:** 学习使用主动感应运动策略查看物理特性

**作者:** Gabriel B. Margolis, Xiang Fu, Yandong Ji, Pulkit Agrawal

**Abstract:** To plan efficient robot locomotion, we must use the information about a terrain’s physics that can be inferred from color images. To this end, we train a visual perception module that predicts terrain properties using labels from a small amount of real-world proprioceptive locomotion. To ensure label precision, we introduce Active Sensing Motor Policies (ASMP). These policies are trained to prefer motor skills that facilitate accurately estimating the environment’s physics, like swiping a foot to observe friction. The estimated labels supervise a vision model that infers physical properties directly from color images and can be reused for different tasks. Leveraging a pretrained vision backbone, we demonstrate robust generalization in image space, enabling path planning from overhead imagery despite using only ground camera images for training.

**摘要:** 为了规划高效的机器人运动,我们必须利用从颜色图像中推导的地形物理信息。为此目的,我们训练了一个视觉感知模块,它利用少量真实感知运动的标签来预测地形的特性。为了确保标签的准确性,我们引入了主动感知运动策略(ASMP)。这些策略被训练以偏好运动技能,有助于准确估计环境的物理,如旋转脚来观察摩擦。估计的标签监督视觉模型,直接推导颜色图像的物理特性,并可用于不同的任务。

**[Paper URL](https://proceedings.mlr.press/v229/margolis23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/margolis23a/margolis23a.pdf)** 

# General In-hand Object Rotation with Vision and Touch
**题目:** 视觉和触摸的一般手动物体转动

**作者:** Haozhi Qi, Brent Yi, Sudharshan Suresh, Mike Lambeta, Yi Ma, Roberto Calandra, Jitendra Malik

**Abstract:** We introduce Rotateit, a system that enables fingertip-based object rotation along multiple axes by leveraging multimodal sensory inputs. Our system is trained in simulation, where it has access to ground-truth object shapes and physical properties. Then we distill it to operate on realistic yet noisy simulated visuotactile and proprioceptive sensory inputs. These multimodal inputs are fused via a visuotactile transformer, enabling online inference of object shapes and physical properties during deployment. We show significant performance improvements over prior methods and highlight the importance of visual and tactile sensing.

**摘要:** 介绍了一种基于指尖的多模式感官输入的多轴旋转系统Rotateit。该系统采用模拟技术,可获得地面真实物体形状和物理特性。然后,我们将该系统蒸馏成现实而又吵闹的仿真视觉感官输入和亲感官输入。这些多模式感官输入通过视觉感官变换器结合起来,在部署过程中可进行物体形状和物理特性的在线推导。

**[Paper URL](https://proceedings.mlr.press/v229/qi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/qi23a/qi23a.pdf)** 

# Imitating Task and Motion Planning with Visuomotor Transformers
**题目:** 与视觉运动变换器模拟任务和运动规划

**作者:** Murtaza Dalal, Ajay Mandlekar, Caelan Reed Garrett, Ankur Handa, Ruslan Salakhutdinov, Dieter Fox

**Abstract:** Imitation learning is a powerful tool for training robot manipulation policies, allowing them to learn from expert demonstrations without manual programming or trial-and-error. However, common methods of data collection, such as human supervision, scale poorly, as they are time-consuming and labor-intensive. In contrast, Task and Motion Planning (TAMP) can autonomously generate large-scale datasets of diverse demonstrations. In this work, we show that the combination of large-scale datasets generated by TAMP supervisors and flexible Transformer models to fit them is a powerful paradigm for robot manipulation. We present a novel imitation learning system called OPTIMUS that trains large-scale visuomotor Transformer policies by imitating a TAMP agent. We conduct a thorough study of the design decisions required to imitate TAMP and demonstrate that OPTIMUS can solve a wide variety of challenging vision-based manipulation tasks with over 70 different objects, ranging from long-horizon pick-and-place tasks, to shelf and articulated object manipulation, achieving $70$ to $80%$ success rates. Video results and code at https://mihdalal.github.io/optimus/

**摘要:** 仿真学习是训练机器人操纵政策的强有力工具,允许他们从专家的演示中学习,而不需要手工编程或试错。然而,数据收集的一般方法,如人监督,规模较差,因为它们耗费时间和劳动 intensive。相反,任务和运动规划(TAMP)可以自主生成各种演示的大规模数据集。我们对仿真TAMP所需的设计决策进行了深入的研究,并证明OPTIMUS能够解决70多个不同对象的复杂视觉操作任务,从远景选择和定位任务到架子和弹性对象操作,达到70至80%的成功率。

**[Paper URL](https://proceedings.mlr.press/v229/dalal23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/dalal23a/dalal23a.pdf)** 

# Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks
**题目:** 好奇心驱动的共同运动和操纵任务学习

**作者:** Clemens Schwarke, Victor Klemm, Matthijs van der Boon, Marko Bjelonic, Marco Hutter

**Abstract:** Learning complex locomotion and manipulation tasks presents significant challenges, often requiring extensive engineering of, e.g., reward functions or curricula to provide meaningful feedback to the Reinforcement Learning (RL) algorithm. This paper proposes an intrinsically motivated RL approach to reduce task-specific engineering. The desired task is encoded in a single sparse reward, i.e., a reward of “+1" is given if the task is achieved. Intrinsic motivation enables learning by guiding exploration toward the sparse reward signal. Specifically, we adapt the idea of Random Network Distillation (RND) to the robotics domain to learn holistic motion control policies involving simultaneous locomotion and manipulation. We investigate opening doors as an exemplary task for robotic ap- plications. A second task involving package manipulation from a table to a bin highlights the generalization capabilities of the presented approach. Finally, the resulting RL policies are executed in real-world experiments on a wheeled-legged robot in biped mode. We experienced no failure in our experiments, which consisted of opening push doors (over 15 times in a row) and manipulating packages (over 5 times in a row).

**摘要:** 研究复杂运动和操作任务具有重大的挑战,往往需要广泛的工程,例如奖励函数或课程,以便对增强学习(RL)算法提供有意义的反馈。本文提出了一种具有内在动机的RL方法,以减少特定任务的工程。希望的任务被编码成单一的稀有奖励,即在完成任务时给予“+1”的奖励。内在动机通过引导探索向稀有奖励信号进行学习。具体地说,我们将随机网络蒸馏(RND)的概念适应到机器人领域,以学习包括同时运动和操作的整体运动控制政策。最后,结果的RL策略在双脚机器人的实际实验中执行。我们实验中没有遇到任何失败,包括打开推门(连续15次)和操作包装(连续5次)。

**[Paper URL](https://proceedings.mlr.press/v229/schwarke23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/schwarke23a/schwarke23a.pdf)** 

# Towards Scalable Coverage-Based Testing of Autonomous Vehicles
**题目:** 基于可扩展覆盖范围的自主车辆测试

**作者:** James Tu, Simon Suo, Chris Zhang, Kelvin Wong, Raquel Urtasun

**Abstract:** To deploy autonomous vehicles(AVs) in the real world, developers must understand the conditions in which the system can operate safely. To do this in a scalable manner, AVs are often tested in simulation on parameterized scenarios. In this context, it’s important to build a testing framework that partitions the scenario parameter space into safe, unsafe, and unknown regions. Existing approaches rely on discretizing continuous parameter spaces into bins, which scales poorly to high-dimensional spaces and cannot describe regions with arbitrary shape. In this work, we introduce a problem formulation which avoids discretization – by modeling the probability of meeting safety requirements everywhere, the parameter space can be paritioned using a probability threshold. Based on our formulation, we propose GUARD as a testing framework which leverages Gaussian Processes to model probability and levelset algorithms to efficiently generate tests. Moreover, we introduce a set of novel evaluation metrics for coverage-based testing frameworks to capture the key objectives of testing. In our evaluation suite of diverse high-dimensional scenarios, GUARD significantly outperforms existing approaches. By proposing an efficient, accurate, and scalable testing framework, our work is a step towards safely deploying autonomous vehicles at scale.

**摘要:** 为了在现实世界中部署自主车辆(AV),开发者必须了解系统安全运行的条件。以可扩展的方式,AV经常在参数化场景的仿真中测试。在这方面,重要的是建立一个测试框架,将场景参数空间分割为安全、不安全和未知区域。现有的方法依赖于将连续参数空间分割为垃圾箱,这种方法对高维空间的规模很差,无法描述任意形状的区域。此外,我们引入了一套新的评估指标,用于基于覆盖范围的测试框架,以捕捉测试的关键目标。在我们的多种高次元场景的评估套件中,GUARD significantly outperforms existing approaches。通过提出有效的、准确的和可扩展的测试框架,我们的工作是安全地部署大规模的自主车辆的一步。

**[Paper URL](https://proceedings.mlr.press/v229/tu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tu23a/tu23a.pdf)** 

# PLEX: Making the Most of the Available Data for Robotic Manipulation Pretraining
**题目:** PLEX:为机器人操纵预训练提供最大的数据

**作者:** Garrett Thomas, Ching-An Cheng, Ricky Loynd, Felipe Vieira Frujeri, Vibhav Vineet, Mihai Jalobeanu, Andrey Kolobov

**Abstract:** A rich representation is key to general robotic manipulation, but existing approaches to representation learning require large amounts of multimodal demonstrations. In this work we propose PLEX, a transformer-based architecture that learns from a small amount of task-agnostic visuomotor trajectories and a much larger amount of task-conditioned object manipulation videos – a type of data available in quantity. PLEX uses visuomotor trajectories to induce a latent feature space and to learn task-agnostic manipulation routines, while diverse video-only demonstrations teach PLEX how to plan in the induced latent feature space for a wide variety of tasks. Experiments showcase PLEX’s generalization on Meta-World and SOTA performance in challenging Robosuite environments. In particular, using relative positional encoding in PLEX’s transformers greatly helps in low-data regimes of learning from human-collected demonstrations.

**摘要:** 丰富的描述是通用机器人操作的关键,但现有的表示学习方法需要大量多模态的演示。在这个工作中,我们提出了PLEX,一种基于变换器的架构,它从少量的任务感知视觉运动轨迹和大量任务条件的对象操作视频中学习 — — 一种数量上可用的数据类型。PLEX使用视觉运动轨迹来诱导潜在特征空间和学习任务感知操作程序,而不同的视频性演示只教PLEX如何在诱导潜在特征空间规划广泛的任务。

**[Paper URL](https://proceedings.mlr.press/v229/thomas23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/thomas23a/thomas23a.pdf)** 

# Learning Lyapunov-Stable Polynomial Dynamical Systems Through Imitation
**题目:** 通过模仿学习拉普诺夫-稳定多项式动态系统

**作者:** Amin Abyaneh, Hsiu-Chin Lin

**Abstract:** Imitation learning is a paradigm to address complex motion planning problems by learning a policy to imitate an expert’s behavior. However, relying solely on the expert’s data might lead to unsafe actions when the robot deviates from the demonstrated trajectories. Stability guarantees have previously been provided utilizing nonlinear dynamical systems, acting as high-level motion planners, in conjunction with the Lyapunov stability theorem. Yet, these methods are prone to inaccurate policies, high computational cost, sample inefficiency, or quasi stability when replicating complex and highly nonlinear trajectories. To mitigate this problem, we present an approach for learning a globally stable nonlinear dynamical system as a motion planning policy. We model the nonlinear dynamical system as a parametric polynomial and learn the polynomial’s coefficients jointly with a Lyapunov candidate. To showcase its success, we compare our method against the state of the art in simulation and conduct real-world experiments with the Kinova Gen3 Lite manipulator arm. Our experiments demonstrate the sample efficiency and reproduction accuracy of our method for various expert trajectories, while remaining stable in the face of perturbations.

**摘要:** 仿真学习是解决复杂运动规划问题的一种范式,通过学习一种政策来模仿专家的行为。然而,仅依靠专家的数据可能导致机器人偏离所演示的轨迹时的不安全行为。稳定性保证以前是通过使用非线性动力系统,作为高层次运动规划者,结合拉普诺夫稳定性定理提供。然而,这些方法在复制复杂和高非线性轨迹时容易产生不准确的政策、高计算成本、样品效率低、或准稳定性。为了展示其成功,我们将我们的方法与仿真技术相比较,并与Kinova Gen3 Lite操纵器臂进行实时实验。我们的实验显示了我们的方法在不同专家轨迹的样品效率和复制精度,同时在面对扰动时保持稳定。

**[Paper URL](https://proceedings.mlr.press/v229/abyaneh23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/abyaneh23a/abyaneh23a.pdf)** 

# MUTEX: Learning Unified Policies from Multimodal Task Specifications
**题目:** MUTEX:从多型任务规范中学习统一政策

**作者:** Rutav Shah, Roberto Martín-Martín, Yuke Zhu

**Abstract:** Humans use different modalities, such as speech, text, images, videos, etc., to communicate their intent and goals with teammates. For robots to become better assistants, we aim to endow them with the ability to follow instructions and understand tasks specified by their human partners. Most robotic policy learning methods have focused on one single modality of task specification while ignoring the rich cross-modal information. We present MUTEX, a unified approach to policy learning from multimodal task specifications. It trains a transformer-based architecture to facilitate cross-modal reasoning, combining masked modeling and cross-modal matching objectives in a two-stage training procedure. After training, MUTEX can follow a task specification in any of the six learned modalities (video demonstrations, goal images, text goal descriptions, text instructions, speech goal descriptions, and speech instructions) or a combination of them. We systematically evaluate the benefits of MUTEX in a newly designed dataset with 100 tasks in simulation and 50 tasks in the real world, annotated with multiple instances of task specifications in different modalities, and observe improved performance over methods trained specifically for any single modality. More information at https://ut-austin-rpl.github.io/MUTEX/

**摘要:** 为了使机器人成为更好的助手,我们的目标是赋予他们遵循指令和理解由人类伙伴指定的任务的能力。大多数机器人政策学习方法都集中在单一的任务规范模式,同时忽略了丰富的跨模态信息。我们介绍MUTEX,一种统一的政策学习方法,从多模态任务规范中学习。它训练了一个基于变换器的架构,以促进跨模态推理,并结合伪装模型和跨模态匹配目标在两个阶段的训练程序中。我们系统地评估了MUTEX在新的设计数据集中的作用,其中有100个模拟任务和50个实时任务,并附有多个不同模式的任务规范实例,并观察了针对任何单一模式的训练方法的改进性能。

**[Paper URL](https://proceedings.mlr.press/v229/shah23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shah23b/shah23b.pdf)** 

# Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning
**题目:** 大型语言模型导航:语义推测作为规划的理论

**作者:** Dhruv Shah, Michael Robert Equi, Błażej Osiński, Fei Xia, Brian Ichter, Sergey Levine

**Abstract:** Navigation in unfamiliar environments presents a major challenge for robots: while mapping and planning techniques can be used to build up a representation of the world, quickly discovering a path to a desired goal in unfamiliar settings with such methods often requires lengthy mapping and exploration. Humans can rapidly navigate new environments, particularly indoor environments that are laid out logically, by leveraging semantics — e.g., a kitchen often adjoins a living room, an exit sign indicates the way out, and so forth. Language models can provide robots with such knowledge, but directly using language models to instruct a robot how to reach some destination can also be impractical: while language models might produce a narrative about how to reach some goal, because they are not grounded in real-world observations, this narrative might be arbitrarily wrong. Therefore, in this paper we study how the “semantic guesswork” produced by language models can be utilized as a guiding heuristic for planning algorithms. Our method, Language Frontier Guide (LFG), uses the language model to bias exploration of novel real-world environments by incorporating the semantic knowledge stored in language models as a search heuristic for planning with either topological or metric maps. We evaluate LFG in challenging real-world environments and simulated benchmarks, outperforming uninformed exploration and other ways of using language models.

**摘要:** 在不熟悉的环境中导航是机器人面临的一个重大挑战:虽然映射和规划技术可以用于建立世界形象,但在不熟悉的环境中快速找到目标的路径往往需要长时间的映射和探索。人类能够快速地导航新的环境,特别是在逻辑上确定的室内环境,通过利用语义 — — 例如,厨房经常毗邻客厅,出口标志指示出路,等等。因此,本文研究了语言模型所产生的“语义推测”如何作为规划算法的指导性推测。我们的方法,语言边界指南(英语:Language Frontier Guide)(LFG)利用语言模型,通过将语言模型中存储的语义知识纳入拓扑或度量地图的规划搜索推测。

**[Paper URL](https://proceedings.mlr.press/v229/shah23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shah23c/shah23c.pdf)** 

# A Data-efficient Neural ODE Framework for Optimal Control of Soft Manipulators
**题目:** 一种数据高效的神经ODE框架,用于优化软操作器的控制

**作者:** Mohammadreza Kasaei, Keyhan Kouhkiloui Babarahmati, Zhibin Li, Mohsen Khadem

**Abstract:** This paper introduces a novel approach for modeling continuous forward kinematic models of soft continuum robots by employing Augmented Neural ODE (ANODE), a cutting-edge family of deep neural network models. To the best of our knowledge, this is the first application of ANODE in modeling soft continuum robots. This formulation introduces auxiliary dimensions, allowing the system’s states to evolve in the augmented space which provides a richer set of dynamics that the model can learn, increasing the flexibility and accuracy of the model. Our methodology achieves exceptional sample efficiency, training the continuous forward kinematic model using only 25 scattered data points. Additionally, we design and implement a fully parallel Model Predictive Path Integral (MPPI)-based controller running on a GPU, which efficiently manages a non-convex objective function. Through a set of experiments, we showed that the proposed framework (ANODE+MPPI) significantly outperforms state-of-the-art learning-based methods such as FNN and RNN in unseen-before scenarios and marginally outperforms them in seen-before scenarios.

**摘要:** 本文介绍了一种应用深神经网络模型的最先进的增强神经ODE(ANODE)方法对软连续机器人连续运动模型的建模,为软连续机器人连续运动模型的建模提供了一个新颖的途径。该方法为软连续机器人建模提供了辅助维度,允许系统状态在增强空间中演化,从而提供模型学习的更丰富的动力学,提高了模型的灵活性和准确性。通过一系列实验,我们表明,该框架(ANODE+MPPI)在未见前场景中大大超过了最先进的基于学习方法,如FNN和RNN,并在未见前场景中略微超过了它们。

**[Paper URL](https://proceedings.mlr.press/v229/kasaei23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kasaei23a/kasaei23a.pdf)** 

# Language Conditioned Traffic Generation
**题目:** 语言条件下的交通生成

**作者:** Shuhan Tan, Boris Ivanovic, Xinshuo Weng, Marco Pavone, Philipp Kraehenbuehl

**Abstract:** Simulation forms the backbone of modern self-driving development. Simulators help develop, test, and improve driving systems without putting humans, vehicles, or their environment at risk. However, simulators face a major challenge: They rely on realistic, scalable, yet interesting content. While recent advances in rendering and scene reconstruction make great strides in creating static scene assets, modeling their layout, dynamics, and behaviors remains challenging. In this work, we turn to language as a source of supervision for dynamic traffic scene generation. Our model, LCTGen, combines a large language model with a transformer-based decoder architecture that selects likely map locations from a dataset of maps, and produces an initial traffic distribution, as well as the dynamics of each vehicle. LCTGen outperforms prior work in both unconditional and conditional traffic scene generation in terms of realism and fidelity.

**摘要:** 仿真是现代自驾驶发展的支柱。仿真器可以帮助开发、测试和改进驾驶系统,而不让人类、车辆或环境受到威胁。然而,仿真器面临一个重大挑战:它们依赖于现实的、可扩展的、但有趣的内容。尽管最近的渲染和场景重建在创建静态场景资产方面取得了巨大进展,但其布局、动态和行为的建模仍然是挑战性的。在这个工作中,我们转向语言作为动态交通场景生成的监督源。我们的模型,LCTGen,结合了一个基于变换器的解码器架构,它从地图数据集中选择可能的地图位置,并产生初始的交通分布,以及每个车辆的动态。

**[Paper URL](https://proceedings.mlr.press/v229/tan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tan23a/tan23a.pdf)** 

# CALAMARI: Contact-Aware and Language conditioned spatial Action MApping for contact-RIch manipulation
**题目:** CALAMARI:接触感知和语言条件的空间行动映射用于接触-RIch操作

**作者:** Youngsun Wi, Mark Van der Merwe, Pete Florence, Andy Zeng, Nima Fazeli

**Abstract:** Making contact with purpose is a central part of robot manipulation and remains essential for many household tasks – from sweeping dust into a dustpan, to wiping tables; from erasing whiteboards, to applying paint. In this work, we investigate learning language-conditioned, vision-based manipulation policies wherein the action representation is in fact, contact itself – predicting contact formations at which tools grasped by the robot should meet an observable surface. Our approach, Contact-Aware and Language conditioned spatial Action MApping for contact-RIch manipulation (CALAMARI), exhibits several advantages including (i) benefiting from existing visual-language models for pretrained spatial features, grounding instructions to behaviors, and for sim2real transfer; and (ii) factorizing perception and control over a natural boundary (i.e. contact) into two modules that synergize with each other, whereby action predictions can be aligned per pixel with image observations, and low-level controllers can optimize motion trajectories that maintain contact while avoiding penetration. Experiments show that CALAMARI outperforms existing state-of-the-art model architectures for a broad range of contact-rich tasks, and pushes new ground on embodiment-agnostic generalization to unseen objects with varying elasticity, geometry, and colors in both simulated and real-world settings.

**摘要:** 目的的接触是机器人操纵的核心部分,也是许多家庭作业中必不可少的,从扫尘到擦桌子、擦白板、涂漆等。我们研究学习语言条件的视觉操纵政策,其中动作表现实际上是接触本身 — — 预测机器人掌握的工具应达到可观察的表面的接触形成。实验表明,CALAMARI比现有的最先进的模型架构更适合广泛的接触式任务,并且在模拟和现实环境中将隐形性、几何和颜色变异的隐形普遍化应用于无形对象。

**[Paper URL](https://proceedings.mlr.press/v229/wi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wi23a/wi23a.pdf)** 

# Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities
**题目:** 通过认识和沟通能力对异种多机器人政策的推广

**作者:** Pierce Howell, Max Rudolph, Reza Joseph Torbati, Kevin Fu, Harish Ravichandar

**Abstract:** Recent advances in multi-agent reinforcement learning (MARL) are enabling impressive coordination in heterogeneous multi-robot teams. However, existing approaches often overlook the challenge of generalizing learned policies to teams of new compositions, sizes, and robots. While such generalization might not be important in teams of virtual agents that can retrain policies on-demand, it is pivotal in multi-robot systems that are deployed in the real-world and must readily adapt to inevitable changes. As such, multi-robot policies must remain robust to team changes – an ability we call adaptive teaming. In this work, we investigate if awareness and communication of robot capabilities can provide such generalization by conducting detailed experiments involving an established multi-robot test bed. We demonstrate that shared decentralized policies, that enable robots to be both aware of and communicate their capabilities, can achieve adaptive teaming by implicitly capturing the fundamental relationship between collective capabilities and effective coordination. Videos of trained policies can be viewed at https://sites.google.com/view/cap-comm .

**摘要:** 近年来,多代理强化学习(MARL)的进展使得异种多机器人团队能够进行令人印象深刻的协调。然而,现有的方法往往忽略了将学习政策推广到新组成、大小和机器人的团队的挑战。虽然这种推广在可以要求重新培训政策的虚拟代理团队中可能并不重要,但在实际应用的多机器人系统中却至关重要,并且必须随时适应不可避免的变化。因此,多机器人政策必须保持对团队变化的鲁棒性——我们称之为适应性团队化的能力。我们证明,共享分散政策,使机器人既能意识到自己的能力和沟通,也可以通过隐含地捕捉集体能力和有效的协调之间的基本关系实现适应性团队合作。

**[Paper URL](https://proceedings.mlr.press/v229/howell23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/howell23a/howell23a.pdf)** 

# CAJun: Continuous Adaptive Jumping using a Learned Centroidal Controller
**题目:** CAJun:使用学习中心控制器进行持续适应跳跃

**作者:** Yuxiang Yang, Guanya Shi, Xiangyun Meng, Wenhao Yu, Tingnan Zhang, Jie Tan, Byron Boots

**Abstract:** We present CAJun, a novel hierarchical learning and control framework that enables legged robots to jump continuously with adaptive jumping distances. CAJun consists of a high-level centroidal policy and a low-level leg controller. In particular, we use reinforcement learning (RL) to train the centroidal policy, which specifies the gait timing, base velocity, and swing foot position for the leg controller. The leg controller optimizes motor commands for the swing and stance legs according to the gait timing to track the swing foot target and base velocity commands. Additionally, we reformulate the stance leg optimizer in the leg controller to speed up policy training by an order of magnitude. Our system combines the versatility of learning with the robustness of optimal control. We show that after 20 minutes of training on a single GPU, CAJun can achieve continuous, long jumps with adaptive distances on a Go1 robot with small sim-to-real gaps. Moreover, the robot can jump across gaps with a maximum width of 70cm, which is over $40%$ wider than existing methods.

**摘要:** 提出了一种新型的层次学习和控制框架,使腿部机器人能够连续跳跃适应性跳跃距离。CAJun由高层次的中心政策和低层次的腿部控制器组成。特别是,我们使用强化学习(RL)来训练中心政策,该政策指定脚部控制器的步调时间、基速和摇脚位置。脚部控制器根据步调时间对摇脚目标和基速命令跟踪摇脚和站脚的运动命令进行优化。此外,我们将脚部控制器中的站脚优化器改编成大小顺序来加快政策训练。我们的系统结合了学习的多样性与优化控制的鲁棒性。此外,机器人可以跳过空隙,最大宽度为70厘米,比现有的方法更宽40美元。

**[Paper URL](https://proceedings.mlr.press/v229/yang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23b/yang23b.pdf)** 

# Multi-Predictor Fusion: Combining Learning-based and Rule-based Trajectory Predictors
**题目:** 多预测融合:基于学习和基于规则的预测

**作者:** Sushant Veer, Apoorva Sharma, Marco Pavone

**Abstract:** Trajectory prediction modules are key enablers for safe and efficient planning of autonomous vehicles (AVs), particularly in highly interactive traffic scenarios. Recently, learning-based trajectory predictors have experienced considerable success in providing state-of-the-art performance due to their ability to learn multimodal behaviors of other agents from data. In this paper, we present an algorithm called multi-predictor fusion (MPF) that augments the performance of learning-based predictors by imbuing them with motion planners that are tasked with satisfying logic-based rules. MPF probabilistically combines learning- and rule-based predictors by mixing trajectories from both standalone predictors in accordance with a belief distribution that reflects the online performance of each predictor. In our results, we show that MPF outperforms the two standalone predictors on various metrics and delivers the most consistent performance.

**摘要:** 基于学习的轨迹预测器最近在提供最先进的性能方面取得了巨大的成功,因为它们能够从数据中学习其他代理人的多模态行为。本文提出了一种叫做多预测器融合(MPF)的算法,以满足逻辑规则的运动规划器来增强基于学习的预测器的性能。 MPF概率性地结合了基于学习的和基于规则的预测器,通过混合两个独立的预测器的轨迹,以反映每个预测器的在线性能的信念分布。

**[Paper URL](https://proceedings.mlr.press/v229/veer23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/veer23a/veer23a.pdf)** 

# Neural Field Dynamics Model for Granular Object Piles Manipulation
**题目:** 粒状物体桩操作神经场动力学模型

**作者:** Shangjie Xue, Shuo Cheng, Pujith Kachana, Danfei Xu

**Abstract:** We present a learning-based dynamics model for granular material manipulation. Drawing inspiration from computer graphics’ Eulerian approach, our method adopts a fully convolutional neural network that operates on a density field-based representation of object piles, allowing it to exploit the spatial locality of inter-object interactions through the convolution operations. This approach greatly improves the learning and computation efficiency compared to existing latent or particle-based methods and sidesteps the need for state estimation, making it directly applicable to real-world settings. Furthermore, our differentiable action rendering module makes the model fully differentiable and can be directly integrated with a gradient-based algorithm for curvilinear trajectory optimization. We evaluate our model with a wide array of piles manipulation tasks both in simulation and real-world experiments and demonstrate that it significantly exceeds existing methods in both accuracy and computation efficiency. More details can be found at https://sites.google.com/view/nfd-corl23/

**摘要:** 我们提出了一种基于学习的微粒材料操作动力学模型,从计算机图形学的欧勒主义方法中汲取灵感,采用了一种基于密度场的对象堆表的完全卷积神经网络,通过卷积操作利用对象间相互作用的空间局部性,大大提高了学习和计算效率,与现有的隐形或粒子based方法相比,减少了状态估计的必要性,从而直接适用于现实环境。我们以大量的桩操作任务来评估我们的模型,在仿真和现实实验中,并证明它大大超过了现有的精度和计算效率的方法。

**[Paper URL](https://proceedings.mlr.press/v229/xue23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xue23a/xue23a.pdf)** 

# AR2-D2: Training a Robot Without a Robot
**题目:** AR2-D2:训练无机器人的机器人

**作者:** Jiafei Duan, Yi Ru Wang, Mohit Shridhar, Dieter Fox, Ranjay Krishna

**Abstract:** Diligently gathered human demonstrations serve as the unsung heroes empowering the progression of robot learning. Today, demonstrations are collected by training people to use specialized controllers, which (tele-)operate robots to manipulate a small number of objects. By contrast, we introduce AR2-D2: a system for collecting demonstrations which (1) does not require people with specialized training, (2) does not require any real robots during data collection, and therefore, (3) enables manipulation of diverse objects with a real robot. AR2-D2 is a framework in the form of an iOS app that people can use to record a video of themselves manipulating any object while simultaneously capturing essential data modalities for training a real robot. We show that data collected via our system enables the training of behavior cloning agents in manipulating real objects. Our experiments further show that training with our AR data is as effective as training with real-world robot demonstrations. Moreover, our user study indicates that users find AR2-D2 intuitive to use and require no training in contrast to four other frequently employed methods for collecting robot demonstrations.

**摘要:** 通过训练人们使用专门控制器(电话)操作机器人来操纵小数量的对象,我们收集了人体的演示。 相反,我们引入了AR2-D2:一种收集演示的系统,它(一)不需要有专门训练的人,(二)不需要在数据收集过程中任何真正的机器人,因此(三)允许用真正的机器人操纵各种对象。 AR2-D2是一个iOS应用程序的形式的框架,人们可以使用它来记录自己操纵任何对象,同时捕捉一个真正的机器人训练的基本数据模式。此外,我们的用户研究表明,用户发现AR2-D2易于使用,不需要任何训练,与其他四种经常使用的收集机器人演示方法相比。

**[Paper URL](https://proceedings.mlr.press/v229/duan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/duan23a/duan23a.pdf)** 

# Affordance-Driven Next-Best-View Planning for Robotic Grasping
**题目:** 基于成本的机器人耕作下一个最佳视图规划

**作者:** Xuechao Zhang, Dong Wang, Sun Han, Weichuang Li, Bin Zhao, Zhigang Wang, Xiaoming Duan, Chongrong Fang, Xuelong Li, Jianping He

**Abstract:** Grasping occluded objects in cluttered environments is an essential component in complex robotic manipulation tasks. In this paper, we introduce an AffordanCE-driven Next-Best-View planning policy (ACE-NBV) that tries to find a feasible grasp for target object via continuously observing scenes from new viewpoints. This policy is motivated by the observation that the grasp affordances of an occluded object can be better-measured under the view when the view-direction are the same as the grasp view. Specifically, our method leverages the paradigm of novel view imagery to predict the grasps affordances under previously unobserved view, and select next observation view based on the highest imagined grasp quality of the target object. The experimental results in simulation and on a real robot demonstrate the effectiveness of the proposed affordance-driven next-best-view planning policy. Project page: https://sszxc.net/ace-nbv/.

**摘要:** 在复杂机器人操纵任务中,在杂乱的环境中捕捉隐形对象是必不可少的组成部分。本文介绍了基于AffordanCE的“下一步最佳视图”规划策略(ACE-NBV),该策略旨在通过不断观察从新视角的场景来寻找目标对象的可行把握。该策略的动机是观察到隐形对象的把握权限在视图下可以更好地测量,视图方向与把握视图相同。具体而言,我们的方法利用新视图图像的范式来预测先前未观察视图下的把握权限,并根据目标对象的最高想象的把握质量选择下观察视图。仿真和实际机器人实验结果表明了基于 affordance的拟议的下一步最佳视图规划策略的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23i.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23i/zhang23i.pdf)** 

# PairwiseNet: Pairwise Collision Distance Learning for High-dof Robot Systems
**题目:** PairwiseNet:高性能机器人系统双向碰撞距离学习

**作者:** Jihwan Kim, Frank C. Park

**Abstract:** Motion planning for robot manipulation systems operating in complex environments remains a challenging problem. It requires the evaluation of both the collision distance and its derivative. Owing to its computational complexity, recent studies have attempted to utilize data-driven approaches to learn the collision distance. However, their performance degrades significantly for complicated high-dof systems, such as multi-arm robots. Additionally, the model must be retrained every time the environment undergoes even slight changes. In this paper, we propose PairwiseNet, a model that estimates the minimum distance between two geometric shapes and overcomes many of the limitations of current models. By dividing the problem of global collision distance learning into smaller pairwise sub-problems, PairwiseNet can be used to efficiently calculate the global collision distance. PairwiseNet can be deployed without further modifications or training for any system comprised of the same shape elements (as those in the training dataset). Experiments with multi-arm manipulation systems of various dof indicate that our model achieves significant performance improvements concerning several performance metrics, especially the false positive rate with the collision-free guaranteed threshold. Results further demonstrate that our single trained PairwiseNet model is applicable to all multi-arm systems used in the evaluation. The code is available at https://github.com/kjh6526/PairwiseNet.

**摘要:** 在复杂环境下运行的机器人操纵系统运动规划仍然是一个挑战性问题,它需要对碰撞距离及其衍生物进行评估。由于其计算复杂性,最近的研究试图利用数据驱动的方法来学习碰撞距离。然而,它们的性能对于复杂高频系统,如多臂机器人,有显著的降低。此外,每当环境发生轻微的变化时,必须重新训练该模型。PairwiseNet可以在任何由相同形状元素组成的系统(如训练数据集中的系统)中部署,无需进一步修改或训练。对不同DOF的多臂操作系统进行实验表明,我们的模型在几个性能指标方面取得了显著的性能改善,特别是与无碰撞的保证阈值的错误正率。结果进一步表明,我们的单一训练的PairwiseNet模型适用于评估中使用的所有多臂系统。该代码可于 https://github.com/kjh6526/PairwiseNet 下载。

**[Paper URL](https://proceedings.mlr.press/v229/kim23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kim23d/kim23d.pdf)** 

# Fighting Uncertainty with Gradients: Offline Reinforcement Learning via Diffusion Score Matching
**题目:** 与梯度对付不确定性:通过扩散分数匹配的在线强化学习

**作者:** H.J. Terry Suh, Glen Chou, Hongkai Dai, Lujie Yang, Abhishek Gupta, Russ Tedrake

**Abstract:** Gradient-based methods enable efficient search capabilities in high dimensions. However, in order to apply them effectively in offline optimization paradigms such as offline Reinforcement Learning (RL) or Imitation Learning (IL), we require a more careful consideration of how uncertainty estimation interplays with first-order methods that attempt to minimize them. We study smoothed distance to data as an uncertainty metric, and claim that it has two beneficial properties: (i) it allows gradient-based methods that attempt to minimize uncertainty to drive iterates to data as smoothing is annealed, and (ii) it facilitates analysis of model bias with Lipschitz constants. As distance to data can be expensive to compute online, we consider settings where we need amortize this computation. Instead of learning the distance however, we propose to learn its gradients directly as an oracle for first-order optimizers. We show these gradients can be efficiently learned with score-matching techniques by leveraging the equivalence between distance to data and data likelihood. Using this insight, we propose Score-Guided Planning (SGP), a planning algorithm for offline RL that utilizes score-matching to enable first-order planning in high-dimensional problems, where zeroth-order methods were unable to scale, and ensembles were unable to overcome local minima. Website: https://sites.google.com/view/score-guided-planning/home

**摘要:** 基于梯度的方法使高维的搜索能力有效率。然而,为了有效地应用这些方法在 Offline优化模式中,例如 offline Reinforcement Learning (RL) 或 Imitation Learning (IL),我们需要更仔细地考虑如何不确定估计与试图最小化它们的初级方法相互作用。我们研究数据的平滑距离作为不确定度量,并声称它有两个有益的特性: (i)它允许在平滑时驱动迭代数据以减少不确定度的基于梯度的方法,以及 (ii)它帮助分析与利普希茨常数的模型偏差。我们表明,这些梯度可以通过利用数据距离和数据概率之间的等价性来有效地学习。利用这一洞察力,我们提出了Score-Guided Planning(SGP),一个用于非线性RL的规划算法,它利用Score-matching,使高维问题中的初级规划能够实现,其中零级方法无法进行规模化,和组合无法克服局部最小值。

**[Paper URL](https://proceedings.mlr.press/v229/suh23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/suh23a/suh23a.pdf)** 

# Generative Skill Chaining: Long-Horizon Skill Planning with Diffusion Models
**题目:** 生成技能链:扩散模型的长期技能规划

**作者:** Utkarsh Aashu Mishra, Shangjie Xue, Yongxin Chen, Danfei Xu

**Abstract:** Long-horizon tasks, usually characterized by complex subtask dependencies, present a significant challenge in manipulation planning. Skill chaining is a practical approach to solving unseen tasks by combining learned skill priors. However, such methods are myopic if sequenced greedily and face scalability issues with search-based planning strategy. To address these challenges, we introduce Generative Skill Chaining (GSC), a probabilistic framework that learns skill-centric diffusion models and composes their learned distributions to generate long-horizon plans during inference. GSC samples from all skill models in parallel to efficiently solve unseen tasks while enforcing geometric constraints. We evaluate the method on various long-horizon tasks and demonstrate its capability in reasoning about action dependencies, constraint handling, and generalization, along with its ability to replan in the face of perturbations. We show results in simulation and on real robot to validate the efficiency and scalability of GSC, highlighting its potential for advancing long-horizon task planning. More details are available at: https://generative-skill-chaining.github.io/

**摘要:** 长期任务通常以复杂次级任务依赖性为特征,在操作规划中存在着重大的挑战。技能追踪是通过结合学问技能 priors来解决未知任务的实用方法。然而,如果被贪婪地序列,并与基于搜索的规划策略面对可扩展性问题,那么这种方法是不可分割的。为了解决这些挑战,我们引入了 Generative Skill Chaining(GSC),一种学习技能中心扩散模型的概率框架,并将其学问分布组成,在推导过程中生成长期任务计划。GSC从所有技能模型中提取样本,以有效解决未知任务,同时执行几何约束。我们在仿真和实物机器人上验证了GSC的效率和可扩展性,突出了其可促进长期任务规划的潜力。详情可浏览: https://generative-skill-chaining.github.io/

**[Paper URL](https://proceedings.mlr.press/v229/mishra23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mishra23a/mishra23a.pdf)** 

# Online Learning for Obstacle Avoidance
**题目:** 避免障碍的网上学习

**作者:** David Snyder, Meghan Booker, Nathaniel Simon, Wenhan Xia, Daniel Suo, Elad Hazan, Anirudha Majumdar

**Abstract:** We approach the fundamental problem of obstacle avoidance for robotic systems via the lens of online learning. In contrast to prior work that either assumes worst-case realizations of uncertainty in the environment or a stationary stochastic model of uncertainty, we propose a method that is efficient to implement and provably grants instance-optimality with respect to perturbations of trajectories generated from an open-loop planner (in the sense of minimizing worst-case regret). The resulting policy adapts online to realizations of uncertainty and provably compares well with the best obstacle avoidance policy in hindsight from a rich class of policies. The method is validated in simulation on a dynamical system environment and compared to baseline open-loop planning and robust Hamilton-Jacobi reachability techniques. Further, it is implemented on a hardware example where a quadruped robot traverses a dense obstacle field and encounters input disturbances due to time delays, model uncertainty, and dynamics nonlinearities.

**摘要:** 通过在线学习的视角,我们对机器人系统避免障碍的基本问题进行了探讨。与以往的工作相比,我们提出了一种有效的实现方法,并可证明为从开环规划师产生轨迹的扰动提供实例优化(以减少最坏情况的遗憾)的方法。结果的政策在网络上适应了不确定性的实现,并可证明与来自丰富的政策类别的最优的障碍回避政策相比较。该方法在动态系统环境的仿真中得到了验证,与基准开环规划和强有力的汉密尔顿-雅科比可达技术相比。此外,它在硬件实例中实现,一个四倍机器人通过一个密集的障碍场,由于时间延迟、模型不确定性和动力学非线性而遇到输入干扰。

**[Paper URL](https://proceedings.mlr.press/v229/snyder23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/snyder23a/snyder23a.pdf)** 

# Polybot: Training One Policy Across Robots While Embracing Variability
**题目:** 波利博特:在容纳变量的同时,培训一个机器人政策

**作者:** Jonathan Heewon Yang, Dorsa Sadigh, Chelsea Finn

**Abstract:** Reusing large datasets is crucial to scale vision-based robotic manipulators to everyday scenarios due to the high cost of collecting robotic datasets. However, robotic platforms possess varying control schemes, camera viewpoints, kinematic configurations, and end-effector morphologies, posing significant challenges when transferring manipulation skills from one platform to another. To tackle this problem, we propose a set of key design decisions to train a single policy for deployment on multiple robotic platforms. Our framework first aligns the observation and action spaces of our policy across embodiments via utilizing wrist cameras and a unified, but modular codebase. To bridge the remaining domain shift, we align our policy’s internal representations across embodiments via contrastive learning. We evaluate our method on a dataset collected over 60 hours spanning 6 tasks and 3 robots with varying joint configurations and sizes: the WidowX 250S, Franka Emika Panda, and Sawyer. Our results demonstrate significant improvements in success rate and sample efficiency for our policy when using new task data collected on a different robot, validating our proposed design decisions. More details and videos can be found on our project website: https://sites.google.com/view/cradle-multirobot

**摘要:** 由于收集机器人数据集的成本高,重用大型数据集对于将基于视觉的机器人操纵者 scale到日常场景至关重要。然而,机器人平台拥有不同的控制方案、摄像机视角、运动配置和最终效果器形态,在将操作技能从一个平台转移到另一个平台时会产生重大挑战。为了解决这个问题,我们提出了一套关键的设计决策,以训练多个机器人平台部署单一的政策。我们的框架首先通过使用手腕摄像机和统一的,但模块化代码库来调整我们的政策的观察和行动空间。我们的研究结果表明,在不同机器人上收集的新任务数据,验证了我们提出的设计决策,从而大大提高了我们政策的成功率和样品效率。更多细节和视频可浏览我们的项目网站: https://sites.google.com/view/cradle-multirobot

**[Paper URL](https://proceedings.mlr.press/v229/yang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23c/yang23c.pdf)** 

# RoboPianist: Dexterous Piano Playing with Deep Reinforcement Learning
**题目:** RoboPianist:精巧的钢琴演奏与深层次强化学习

**作者:** Kevin Zakka, Philipp Wu, Laura Smith, Nimrod Gileadi, Taylor Howell, Xue Bin Peng, Sumeet Singh, Yuval Tassa, Pete Florence, Andy Zeng, Pieter Abbeel

**Abstract:** Replicating human-like dexterity in robot hands represents one of the largest open problems in robotics. Reinforcement learning is a promising approach that has achieved impressive progress in the last few years; however, the class of problems it has typically addressed corresponds to a rather narrow definition of dexterity as compared to human capabilities. To address this gap, we investigate piano-playing, a skill that challenges even the human limits of dexterity, as a means to test high-dimensional control, and which requires high spatial and temporal precision, and complex finger coordination and planning. We introduce RoboPianist, a system that enables simulated anthropomorphic hands to learn an extensive repertoire of 150 piano pieces where traditional model-based optimization struggles. We additionally introduce an open-sourced environment, benchmark of tasks, interpretable evaluation metrics, and open challenges for future study. Our website featuring videos, code, and datasets is available at https://kzakka.com/robopianist/

**摘要:** 仿真人手的敏捷性是机器人领域最大的开放问题之一。强化学习是近年来取得令人印象深刻进展的有前途的途径,但它所处理的问题类别与人类能力相比相当狭窄的敏捷性定义。为了解决这一差距,我们研究钢琴演奏,一种挑战甚至人类的敏捷性限度的技能,作为测试高维控制的手段,需要高空间和时间精度,以及复杂的手指协调和规划。我们提供视频、代码和数据集的网站可浏览 https://kzakka.com/robopianist/

**[Paper URL](https://proceedings.mlr.press/v229/zakka23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zakka23a/zakka23a.pdf)** 

# Revisiting Depth-guided Methods for Monocular 3D Object Detection by Hierarchical Balanced Depth
**题目:** 基于层次平衡深度对单眼3D对象检测的深度指导方法的检讨

**作者:** Yi-Rong Chen, Ching-Yu Tseng, Yi-Syuan Liou, Tsung-Han Wu, Winston H. Hsu

**Abstract:** Monocular 3D object detection has seen significant advancements with the incorporation of depth information. However, there remains a considerable performance gap compared to LiDAR-based methods, largely due to inaccurate depth estimation. We argue that this issue stems from the commonly used pixel-wise depth map loss, which inherently creates the imbalance of loss weighting between near and distant objects. To address these challenges, we propose MonoHBD (Monocular Hierarchical Balanced Depth), a comprehensive solution with the hierarchical mechanism. We introduce the Hierarchical Depth Map (HDM) structure that incorporates depth bins and depth offsets to enhance the localization accuracy for objects. Leveraging RoIAlign, our Balanced Depth Extractor (BDE) module captures both scene-level depth relationships and object-specific depth characteristics while considering the geometry properties through the inclusion of camera calibration parameters. Furthermore, we propose a novel depth map loss that regularizes object-level depth features to mitigate imbalanced loss propagation. Our model reaches state-of-the-art results on the KITTI 3D object detection benchmark while supporting real-time detection. Excessive ablation studies are also conducted to prove the efficacy of our proposed modules.

**摘要:** 光学3D物体检测在包含深度信息方面取得了重大进展,但与基于LiDAR的方法相比,仍然存在着相当大的性能差距,主要是由于深度估计不准确。我们认为,这一问题源于常用的像素方向深度地图损失,这本质上会造成近距离物体和远距离物体之间的损失权衡不平衡。为了解决这些问题,我们提出了一种包括层次机制的综合解决方案, MonoHBD(Monocular Hierarchical Balanced Depth)。我们引入了层次性深度地图(HDM)结构,它包含深度箱和深度补偿,以提高物体的定位精度。此外,我们提出了一种新颖的深度图损耗,使物体水平深度特征正常化,以缓解失衡的损耗传播。我们的模型在 KITTI 3D物体检测基准上达到最先进的结果,同时支持实时检测。

**[Paper URL](https://proceedings.mlr.press/v229/chen23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23d/chen23d.pdf)** 

# Heteroscedastic Gaussian Processes and Random Features: Scalable Motion Primitives with Guarantees
**题目:** 异代数高斯过程与随机特性:具有保证的可 skalable Motion Primitives

**作者:** Edoardo Caldarelli, Antoine Chatalic, Adrià Colomé, Lorenzo Rosasco, Carme Torras

**Abstract:** Heteroscedastic Gaussian processes (HGPs) are kernel-based, non-parametric models that can be used to infer nonlinear functions with time-varying noise. In robotics, they can be employed for learning from demonstration as motion primitives, i.e. as a model of the trajectories to be executed by the robot. HGPs provide variance estimates around the reference signal modeling the trajectory, capturing both the predictive uncertainty and the motion variability. However, similarly to standard Gaussian processes they suffer from a cubic complexity in the number of training points, due to the inversion of the kernel matrix. The uncertainty can be leveraged for more complex learning tasks, such as inferring the variable impedance profile required from a robotic manipulator. However, suitable approximations are needed to make HGPs scalable, at the price of potentially worsening the posterior mean and variance profiles. Motivated by these observations, we study the combination of HGPs and random features, which are a popular, data-independent approximation strategy of kernel functions. In a theoretical analysis, we provide novel guarantees on the approximation error of the HGP posterior due to random features. Moreover, we validate this scalable motion primitive on real robot data, related to the problem of variable impedance learning. In this way, we show that random features offer a viable and theoretically sound alternative for speeding up the trajectory processing, without sacrificing accuracy.

**摘要:** HGP(英语:Heroscedastic Gaussian processes,缩写为HGP)是基于内核的非参数模型,可用于推导与时间变化的噪声的非线性函数。在机器人学中,它们可用于从演示中学习运动原型,即由机器人执行的轨迹的模型。HGP提供在参考信号模型周围的变量估计,捕捉预测的不确定性和运动变量。然而,与标准高斯过程类似,它们由于内核矩阵的反演而导致训练点数的立方复杂性。基于这些观察,我们研究了HGP和随机特性的结合,它们是核函数的流行数据独立的近似策略。在理论分析中,我们提供了随机特性导致HGP后部近似误差的新保证。此外,我们对实际机器人数据上的可扩展运动原始性进行了验证,与变阻学习问题有关。

**[Paper URL](https://proceedings.mlr.press/v229/caldarelli23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/caldarelli23a/caldarelli23a.pdf)** 

# Human-in-the-Loop Task and Motion Planning for Imitation Learning
**题目:** 模仿学习的人体跳跃任务和运动规划

**作者:** Ajay Mandlekar, Caelan Reed Garrett, Danfei Xu, Dieter Fox

**Abstract:** Imitation learning from human demonstrations can teach robots complex manipulation skills, but is time-consuming and labor intensive. In contrast, Task and Motion Planning (TAMP) systems are automated and excel at solving long-horizon tasks, but they are difficult to apply to contact-rich tasks. In this paper, we present Human-in-the-Loop Task and Motion Planning (HITL-TAMP), a novel system that leverages the benefits of both approaches. The system employs a TAMP-gated control mechanism, which selectively gives and takes control to and from a human teleoperator. This enables the human teleoperator to manage a fleet of robots, maximizing data collection efficiency. The collected human data is then combined with an imitation learning framework to train a TAMP-gated policy, leading to superior performance compared to training on full task demonstrations. We compared HITL-TAMP to a conventional teleoperation system — users gathered more than 3x the number of demos given the same time budget. Furthermore, proficient agents ($75%$+ success) could be trained from just 10 minutes of non-expert teleoperation data. Finally, we collected 2.1K demos with HITL-TAMP across 12 contact-rich, long-horizon tasks and show that the system often produces near-perfect agents. Videos and additional results at https://hitltamp.github.io .

**摘要:** 仿真学习从人类演示中可以教导机器人复杂的操纵技能,但它耗费时间和劳动密集。与此相反,任务和运动规划(TAMP)系统是自动化的,在解决长期任务方面具有突出的优点,但很难应用于接触丰富的任务。本论文中,我们介绍了人类在跳跃任务和运动规划(HITL-TAMP),一种新颖的系统,它利用了两种方法的优点。该系统采用了TAMP-gated控制机制,它选择性地给予和从人类远程操作者手中控制。我们比较了HTL-TAMP与传统的远程通信系统 — 用户收集了3倍多于同一时间预算的演示数目. 此外,熟练的代理人(75%$+成功)可以从仅10分钟的非专家远程通信数据中获得训练.最后,我们收集了2.1K的HTL-TAMP在12个接触丰富、远景任务中进行的演示,并显示该系统经常产生接近完美的代理人. 视频和额外的结果在 https://hitltamp.github.io.

**[Paper URL](https://proceedings.mlr.press/v229/mandlekar23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mandlekar23b/mandlekar23b.pdf)** 

# Gesture-Informed Robot Assistance via Foundation Models
**题目:** 基于基础模型的 gesture-informed robot assistance

**作者:** Li-Heng Lin, Yuchen Cui, Yilun Hao, Fei Xia, Dorsa Sadigh

**Abstract:** Gestures serve as a fundamental and significant mode of non-verbal communication among humans. Deictic gestures (such as pointing towards an object), in particular, offer valuable means of efficiently expressing intent in situations where language is inaccessible, restricted, or highly specialized. As a result, it is essential for robots to comprehend gestures in order to infer human intentions and establish more effective coordination with them. Prior work often rely on a rigid hand-coded library of gestures along with their meanings. However, interpretation of gestures is often context-dependent, requiring more flexibility and common-sense reasoning. In this work, we propose a framework, GIRAF, for more flexibly interpreting gesture and language instructions by leveraging the power of large language models. Our framework is able to accurately infer human intent and contextualize the meaning of their gestures for more effective human-robot collaboration. We instantiate the framework for three table-top manipulation tasks and demonstrate that it is both effective and preferred by users. We further demonstrate GIRAF’s ability on reasoning about diverse types of gestures by curating a GestureInstruct dataset consisting of 36 different task scenarios. GIRAF achieved $81%$ success rate on finding the correct plan for tasks in GestureInstruct. Videos and datasets can be found on our project website: https://tinyurl.com/giraf23

**摘要:** 举动作为人类间非语言交流的基本和重要的方式。举动(例如指向对象)尤其在语言无法访问、限制或高度专业的情况下提供有效的表达意图的宝贵手段。因此,机器人必须理解举动,以便推断人类的意图并建立更有效的协调。以前的工作经常依靠一种严格的手编码的举动库及其意义。然而,举动的解释往往取决于上下文,需要更多的灵活性和公理推理。我们对三个表顶操作任务的框架进行了实例化,并证明它既有效,又被用户偏好。我们还通过对36个不同的任务场景组成的“GestureInstruct”数据集进行管理,展示了GIRAF在不同类型的动作的推理能力。 GIRAF在“GestureInstruct”任务中找到正确的计划方面取得了81%的成功率。

**[Paper URL](https://proceedings.mlr.press/v229/lin23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lin23a/lin23a.pdf)** 

# TactileVAD: Geometric Aliasing-Aware Dynamics for High-Resolution Tactile Control
**题目:** TactileVAD:高分辨率触觉控制的几何隐形意识动力学

**作者:** Miquel Oller, Dmitry Berenson, Nima Fazeli

**Abstract:** Touch-based control is a promising approach to dexterous manipulation. However, existing tactile control methods often overlook tactile geometric aliasing which can compromise control performance and reliability. This type of aliasing occurs when different contact locations yield similar tactile signatures. To address this, we propose TactileVAD, a generative decoder-only linear latent dynamics formulation compatible with standard control methods that is capable of resolving geometric aliasing. We evaluate TactileVAD on two mechanically-distinct tactile sensors, SoftBubbles (pointcloud data) and Gelslim 3.0 (RGB data), showcasing its effectiveness in handling different sensing modalities. Additionally, we introduce the tactile cartpole, a novel benchmarking setup to evaluate the ability of a control method to respond to disturbances based on tactile input. Evaluations comparing TactileVAD to baselines suggest that our method is better able to achieve goal tactile configurations and hand poses.

**摘要:** 触控控制是一种灵活的操作方法,但现有触控控制方法往往忽略了触控几何变形,这可能影响控制性能和可靠性。这种变形发生在不同接触地点产生相似的触控签名时。为了解决这个问题,我们提出了触控VAD,一种基于触控输入的控制方法可解决几何变形的线性隐形动力学公式。我们评估触控VAD在两个机械性差异的触控传感器, SoftBubbles(点云数据)和Gelslim 3.0(RGB数据)上,显示其在处理不同感知模式方面的有效性。对 TactileVAD与基线的评价表明,我们的方法能够更好地实现目标触觉配置和手姿态。

**[Paper URL](https://proceedings.mlr.press/v229/oller23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/oller23a/oller23a.pdf)** 

# FastRLAP: A System for Learning High-Speed Driving via Deep RL and Autonomous Practicing
**题目:** FastRLAP:通过 Deep RL和自主练习学习高速驾驶系统

**作者:** Kyle Stachowicz, Dhruv Shah, Arjun Bhorkar, Ilya Kostrikov, Sergey Levine

**Abstract:** We present a system that enables an autonomous small-scale RC car to drive aggressively from visual observations using reinforcement learning (RL). Our system, FastRLAP, trains autonomously in the real world, without human interventions, and without requiring any simulation or expert demonstrations. Our system integrates a number of important components to make this possible: we initialize the representations for the RL policy and value function from a large prior dataset of other robots navigating in other environments (at low speed), which provides a navigation-relevant representation. From here, a sample-efficient online RL method uses a single low-speed user-provided demonstration to determine the desired driving course, extracts a set of navigational checkpoints, and autonomously practices driving through these checkpoints, resetting automatically on collision or failure. Perhaps surprisingly, we find that with appropriate initialization and choice of algorithm, our system can learn to drive over a variety of racing courses with less than 20 minutes of online training. The resulting policies exhibit emergent aggressive driving skills, such as timing braking and acceleration around turns and avoiding areas which impede the robot’s motion, approaching the performance of a human driver using a similar first-person interface over the course of training.

**摘要:** 我们提出了一种能够由增强学习(RL)的视觉观测驱动自控小型RC车的系统。我们的系统,快速RLAP,在现实世界中进行自控训练,无需人干预,并不需要任何模拟或专家的演示。我们的系统集成了若干重要组成部分,使这成为可能:我们从其他环境(低速)导航的其他机器人的大型预先数据集中初始化RL政策和值函数的表示,从而提供导航相关的表示。也许令人惊奇的是,我们发现,通过适当的初始化和算法选择,我们的系统能够学习在不到20分钟的在线训练中驾驶各种赛车课程。

**[Paper URL](https://proceedings.mlr.press/v229/stachowicz23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/stachowicz23a/stachowicz23a.pdf)** 

# Energy-based Potential Games for Joint Motion Forecasting and Control
**题目:** 基于能源的联合运动预测和控制潜在游戏

**作者:** Christopher Diehl, Tobias Klosek, Martin Krueger, Nils Murzyn, Timo Osterburg, Torsten Bertram

**Abstract:** This work uses game theory as a mathematical framework to address interaction modeling in multi-agent motion forecasting and control. Despite its interpretability, applying game theory to real-world robotics, like automated driving, faces challenges such as unknown game parameters. To tackle these, we establish a connection between differential games, optimal control, and energy-based models, demonstrating how existing approaches can be unified under our proposed Energy-based Potential Game formulation. Building upon this, we introduce a new end-to-end learning application that combines neural networks for game-parameter inference with a differentiable game-theoretic optimization layer, acting as an inductive bias. The analysis provides empirical evidence that the game-theoretic layer adds interpretability and improves the predictive performance of various neural network backbones using two simulations and two real-world driving datasets.

**摘要:** 本研究采用游戏理论作为数学框架,以解决多代理运动预测和控制中的交互模型问题。尽管游戏理论具有可解释性,但应用于现实机器人领域,如自动驾驶,面临未知的游戏参数等挑战。为了解决这些问题,我们建立了微分游戏、优化控制和基于能量的模型之间的联系,以示我们提出的基于能量的潜在游戏公式下,现有的方法是如何统一的。在此基础上,我们引入了一种新的end-to-end学习应用,它将神经网络用于游戏参数推导与可微分的游戏理论优化层结合起来,作为诱导偏差。

**[Paper URL](https://proceedings.mlr.press/v229/diehl23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/diehl23a/diehl23a.pdf)** 

# Dexterity from Touch: Self-Supervised Pre-Training of Tactile Representations with Robotic Play
**题目:** 触摸的敏捷性:自控预训练机器人游戏的触觉表现

**作者:** Irmak Guzey, Ben Evans, Soumith Chintala, Lerrel Pinto

**Abstract:** Teaching dexterity to multi-fingered robots has been a longstanding challenge in robotics. Most prominent work in this area focuses on learning controllers or policies that either operate on visual observations or state estimates derived from vision. However, such methods perform poorly on fine-grained manipulation tasks that require reasoning about contact forces or about objects occluded by the hand itself. In this work, we present T-Dex, a new approach for tactile-based dexterity, that operates in two phases. In the first phase, we collect 2.5 hours of play data, which is used to train self-supervised tactile encoders. This is necessary to bring high-dimensional tactile readings to a lower-dimensional embedding. In the second phase, given a handful of demonstrations for a dexterous task, we learn non-parametric policies that combine the tactile observations with visual ones. Across five challenging dexterous tasks, we show that our tactile-based dexterity models outperform purely vision and torque-based models by an average of 1.7X. Finally, we provide a detailed analysis on factors critical to T-Dex including the importance of play data, architectures, and representation learning.

**摘要:** 对多指机器人的敏捷教学一直是机器人学界的长期挑战。在这一领域,最突出的工作集中在学习控制器或操作视觉观察或视力推测的政策上。然而,这些方法在需要关于接触力或手本身隐藏的对象的推理的细微操作任务中 perform poorly。在这个工作中,我们介绍了T-Dex,一种基于触觉敏捷的新方法,它在两个阶段运行。第一阶段,我们收集了2.5小时的游戏数据,用于训练自监督触觉编码器。在5个挑战性敏捷任务中,我们证明了基于触觉的敏捷模型比纯视觉和基于扭矩的模型平均高1.7倍。最后,我们提供了对T-Dex关键因素的详细分析,包括游戏数据、架构和表示学习的重要性。

**[Paper URL](https://proceedings.mlr.press/v229/guzey23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/guzey23a/guzey23a.pdf)** 

# ADU-Depth: Attention-based Distillation with Uncertainty Modeling for Depth Estimation
**题目:** ADU-深度:基于注意的蒸馏与深度估计的不确定性建模

**作者:** ZiZhang Wu, Zhuozheng Li, Zhi-Gang Fan, Yunzhe Wu, Xiaoquan Wang, Rui Tang, Jian Pu

**Abstract:** Monocular depth estimation is challenging due to its inherent ambiguity and ill-posed nature, yet it is quite important to many applications. While recent works achieve limited accuracy by designing increasingly complicated networks to extract features with limited spatial geometric cues from a single RGB image, we intend to introduce spatial cues by training a teacher network that leverages left-right image pairs as inputs and transferring the learned 3D geometry-aware knowledge to the monocular student network. Specifically, we present a novel knowledge distillation framework, named ADU-Depth, with the goal of leveraging the well-trained teacher network to guide the learning of the student network, thus boosting the precise depth estimation with the help of extra spatial scene information. To enable domain adaptation and ensure effective and smooth knowledge transfer from teacher to student, we apply both attention-adapted feature distillation and focal-depth-adapted response distillation in the training stage. In addition, we explicitly model the uncertainty of depth estimation to guide distillation in both feature space and result space to better produce 3D-aware knowledge from monocular observations and thus enhance the learning for hard-to-predict image regions. Our extensive experiments on the real depth estimation datasets KITTI and DrivingStereo demonstrate the effectiveness of the proposed method, which ranked 1st on the challenging KITTI online benchmark.

**摘要:** 单眼深度估计是由于其固有含糊性和不定性的性质而具有挑战性,但对于许多应用而言,它是十分重要的。虽然最近的工作通过设计日益复杂的网络来从单一的RGB图像中提取具有有限空间几何标识的特征,但我们打算通过训练一个教师网络来引入空间标识,将左-右两对图像作为输入,并将学习的3D几何知识转移到单眼学生网络。为了使领域适应并确保从教师到学生有效和顺利地传递知识,我们在训练阶段应用了注意力适应的特征蒸馏和焦点深度适应的响应蒸馏。此外,我们明确地模拟了深度估计的不确定性,指导了特征空间和结果空间的蒸馏,以更好地从单眼观察中产生3D认知知识,从而提高难以预测图像区域的学习。

**[Paper URL](https://proceedings.mlr.press/v229/wu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wu23c/wu23c.pdf)** 

# Structural Concept Learning via Graph Attention for Multi-Level Rearrangement Planning
**题目:** 基于图形的结构概念学习,用于多层次重组规划

**作者:** Manav Kulshrestha, Ahmed H. Qureshi

**Abstract:** Robotic manipulation tasks, such as object rearrangement, play a crucial role in enabling robots to interact with complex and arbitrary environments. Existing work focuses primarily on single-level rearrangement planning and, even if multiple levels exist, dependency relations among substructures are geometrically simpler, like tower stacking. We propose Structural Concept Learning (SCL), a deep learning approach that leverages graph attention networks to perform multi-level object rearrangement planning for scenes with structural dependency hierarchies. It is trained on a self-generated simulation data set with intuitive structures, works for unseen scenes with an arbitrary number of objects and higher complexity of structures, infers independent substructures to allow for task parallelization over multiple manipulators, and generalizes to the real world. We compare our method with a range of classical and model-based baselines to show that our method leverages its scene understanding to achieve better performance, flexibility, and efficiency. The dataset, demonstration videos, supplementary details, and code implementation are available at: https://manavkulshrestha.github.io/scl

**摘要:** 机器人操作任务,如对象重配置,在使机器人与复杂和任意环境相互作用方面起着关键作用。现有的工作主要集中在单级重配置规划上,即使存在多个层次,子结构间的依赖关系是几何上更简单,如塔堆叠。我们提出了结构概念学习(SCL),一种深度学习方法,利用图形注意力网络来执行具有结构依赖等级的场景的多级对象重配置规划。我们与一系列经典和基于模型的基线进行比较,以证明我们的方法利用场景理解来实现更好的性能、灵活性和效率。数据集、示范视频、补充细节和代码实现可于: https://manavkulshrestha.github.io/scl

**[Paper URL](https://proceedings.mlr.press/v229/kulshrestha23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kulshrestha23a/kulshrestha23a.pdf)** 

# Few-Shot In-Context Imitation Learning via Implicit Graph Alignment
**题目:** 通过隐形图形排列的少数实例仿真学习

**作者:** Vitalis Vosylius, Edward Johns

**Abstract:** Consider the following problem: given a few demonstrations of a task across a few different objects, how can a robot learn to perform that same task on new, previously unseen objects? This is challenging because the large variety of objects within a class makes it difficult to infer the task-relevant relationship between the new objects and the objects in the demonstrations. We address this by formulating imitation learning as a conditional alignment problem between graph representations of objects. Consequently, we show that this conditioning allows for in-context learning, where a robot can perform a task on a set of new objects immediately after the demonstrations, without any prior knowledge about the object class or any further training. In our experiments, we explore and validate our design choices, and we show that our method is highly effective for few-shot learning of several real-world, everyday tasks, whilst outperforming baselines. Videos are available on our project webpage at https://www.robot-learning.uk/implicit-graph-alignment.

**摘要:** 考虑以下问题: 针对多个不同对象的任务,机器人如何学习在新、未见对象上执行相同的任务? 这是一个挑战性的问题,因为类内对象的多样性使得推断新对象与实例中的对象之间的任务相关关系变得困难。我们通过拟定仿真学习作为对象的图形表示之间的条件匹配问题来解决这个问题。因此,我们证明这种条件化允许在实例中学习,即机器人可以在实例后立即完成一系列新对象上的任务,而无需对对象类或任何进一步训练。视频可以在我们的项目网页 https://www.robot-learning.uk/implicit-graph-alignment.

**[Paper URL](https://proceedings.mlr.press/v229/vosylius23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/vosylius23a/vosylius23a.pdf)** 

# Topology-Matching Normalizing Flows for Out-of-Distribution Detection in Robot Learning
**题目:** 机器人学习中非分布检测的拓扑匹配正常化流

**作者:** Jianxiang Feng, Jongseok Lee, Simon Geisler, Stephan Günnemann, Rudolph Triebel

**Abstract:** To facilitate reliable deployments of autonomous robots in the real world, Out-of-Distribution (OOD) detection capabilities are often required. A powerful approach for OOD detection is based on density estimation with Normalizing Flows (NFs). However, we find that prior work with NFs attempts to match the complex target distribution topologically with naïve base distributions leading to adverse implications. In this work, we circumvent this topological mismatch using an expressive class-conditional base distribution trained with an information-theoretic objective to match the required topology. The proposed method enjoys the merits of wide compatibility with existing learned models without any performance degradation and minimum computation overhead while enhancing OOD detection capabilities. We demonstrate superior results in density estimation and 2D object detection benchmarks in comparison with extensive baselines. Moreover, we showcase the applicability of the method with a real-robot deployment.

**摘要:** 为了在现实世界中实现自主机器人的可靠部署,我们经常需要外分布检测能力。外分布检测的强有力方法是基于规范流(NFs)的密度估计。然而,我们发现,NFs的前期工作试图将复杂的目标分布topologically与虚构的基准分布相匹配,从而产生不良影响。在这个工作中,我们利用具有信息理论目标的表达式类-条件基准分布来绕过这种拓扑不匹配,以满足需要的拓扑。该方法具有与现有学习模型的广泛兼容性,无性能降级和最小计算费用,同时提高了外分布检测能力。此外,我们展示了该方法在实际的机器人部署中的应用。

**[Paper URL](https://proceedings.mlr.press/v229/feng23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/feng23b/feng23b.pdf)** 

# Compositional Diffusion-Based Continuous Constraint Solvers
**题目:** 基于复合扩散的连续约束处理器

**作者:** Zhutian Yang, Jiayuan Mao, Yilun Du, Jiajun Wu, Joshua B. Tenenbaum, Tomás Lozano-Pérez, Leslie Pack Kaelbling

**Abstract:** This paper introduces an approach for learning to solve continuous constraint satisfaction problems (CCSP) in robotic reasoning and planning. Previous methods primarily rely on hand-engineering or learning generators for specific constraint types and then rejecting the value assignments when other constraints are violated. By contrast, our model, the compositional diffusion continuous constraint solver (Diffusion-CCSP) derives global solutions to CCSPs by representing them as factor graphs and combining the energies of diffusion models trained to sample for individual constraint types. Diffusion-CCSP exhibits strong generalization to novel combinations of known constraints, and it can be integrated into a task and motion planner to devise long-horizon plans that include actions with both discrete and continuous parameters.

**摘要:** 本文介绍了一种在机器人推理和规划中学习解决连续约束满意问题的方法。以往的方法主要依赖于特定约束类型的手工工程或学习生成器,然后在其他约束被侵犯时拒绝分配值。相反,我们模型的复合扩散连续约束解决器(Diffusion-CCSP)通过将它们代表为因子图和结合用于个别约束类型的样本训练的扩散模型的能量来导出全球解决方案。

**[Paper URL](https://proceedings.mlr.press/v229/yang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23d/yang23d.pdf)** 

# Precise Robotic Needle-Threading with Tactile Perception and Reinforcement Learning
**题目:** 敏锐感知和增强学习的精密机器人针刺

**作者:** Zhenjun Yu, Wenqiang Xu, Siqiong Yao, Jieji Ren, Tutian Tang, Yutong Li, Guoying Gu, Cewu Lu

**Abstract:** This work presents a novel tactile perception-based method, named T-NT, for performing the needle-threading task, an application of deformable linear object (DLO) manipulation. This task is divided into two main stages: Tail-end Finding and Tail-end Insertion. In the first stage, the agent traces the contour of the thread twice using vision-based tactile sensors mounted on the gripper fingers. The two-run tracing is to locate the tail-end of the thread. In the second stage, it employs a tactile-guided reinforcement learning (RL) model to drive the robot to insert the thread into the target needle eyelet. The RL model is trained in a Unity-based simulated environment. The simulation environment supports tactile rendering which can produce realistic tactile images and thread modeling. During insertion, the position of the poke point and the center of the eyelet are obtained through a pre-trained segmentation model, Grounded-SAM, which predicts the masks for both the needle eye and thread imprints. These positions are then fed into the reinforcement learning model, aiding in a smoother transition to real-world applications. Extensive experiments on real robots are conducted to demonstrate the efficacy of our method. More experiments and videos can be found in the supplementary materials and on the website: https://sites.google.com/view/tac-needlethreading.

**摘要:** 该工作提出了一种基于触觉的新方法,名称为T-NT,用于执行针刺任务,一种可变线性对象(DLO)操作的应用程序。该任务分为两个主要阶段:针刺端检测和针刺端插入。在第一阶段,代理使用在握手手指上安装的视觉触觉传感器跟踪针刺的轮廓。两个运行的跟踪是定位针刺的尾端。在第二阶段,它使用触觉导向增强学习(RL)模型驱动机器人将针刺插入目标针眼球。RL模型在统一的仿真环境中进行训练。仿真环境支持触觉渲染,可以产生真实触觉图像和针刺模型。在插入过程中,波克点和眼球中心的位置通过预训练的分割模型Grounded-SAM获得,该模型预测针眼和线纹纹的面具。这些位置然后被输入增强学习模型,帮助更顺利地转变到现实世界中的应用。

**[Paper URL](https://proceedings.mlr.press/v229/yu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yu23c/yu23c.pdf)** 

# Cold Diffusion on the Replay Buffer: Learning to Plan from Known Good States
**题目:** 冷扩散在反弹缓冲器上:从既知好的国家学习规划

**作者:** Zidan Wang, Takeru Oba, Takuma Yoneda, Rui Shen, Matthew Walter, Bradly C. Stadie

**Abstract:** Learning from demonstrations (LfD) has successfully trained robots to exhibit remarkable generalization capabilities. However, many powerful imitation techniques do not prioritize the feasibility of the robot behaviors they generate. In this work, we explore the feasibility of plans produced by LfD. As in prior work, we employ a temporal diffusion model with fixed start and goal states to facilitate imitation through in-painting. Unlike previous studies, we apply cold diffusion to ensure the optimization process is directed through the agent’s replay buffer of previously visited states. This routing approach increases the likelihood that the final trajectories will predominantly occupy the feasible region of the robot’s state space. We test this method in simulated robotic environments with obstacles and observe a significant improvement in the agent’s ability to avoid these obstacles during planning.

**摘要:** LfD的仿真技术使机器人具有显著的一般化能力。然而,许多强有力的仿真技术并没有优先考虑机器人行为的可行性。在这项工作中,我们探讨了由LfD制作的计划的可行性。正如以前的工作一样,我们采用固定启动和目标状态的时空扩散模型,以通过绘制来促进仿真。与以往的研究不同,我们应用冷扩散来确保优化过程通过前所访问的状态的代理再播放缓冲器进行。这种路由方法增加了最终路径的可能占用率,占有机器人状态空间的可行的区域。

**[Paper URL](https://proceedings.mlr.press/v229/wang23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23e/wang23e.pdf)** 

# Self-Improving Robots: End-to-End Autonomous Visuomotor Reinforcement Learning
**题目:** 自修机器人:终到终的自主视觉动力增强学习

**作者:** Archit Sharma, Ahmed M. Ahmed, Rehaan Ahmad, Chelsea Finn

**Abstract:** In imitation and reinforcement learning (RL), the cost of human supervision limits the amount of data that the robots can be trained on. While RL offers a framework for building self-improving robots that can learn via trial-and-error autonomously, practical realizations end up requiring extensive human supervision for reward function design and repeated resetting of the environment between episodes of interactions. In this work, we propose MEDAL++, a novel design for self-improving robotic systems: given a small set of expert demonstrations at the start, the robot autonomously practices the task by learning to both do and undo the task, simultaneously inferring the reward function from the demonstrations. The policy and reward function are learned end-to-end from high-dimensional visual inputs, bypassing the need for explicit state estimation or task-specific pre-training for visual encoders used in prior work. We first evaluate our proposed system on a simulated non-episodic benchmark EARL, finding that MEDAL++ is both more data efficient and gets up to $30%$ better final performance compared to state-of-the-art vision-based methods. Our real-robot experiments show that MEDAL++ can be applied to manipulation problems in larger environments than those considered in prior work, and autonomous self-improvement can improve the success rate by $30%$ to $70%$ over behavioral cloning on just the expert data.

**摘要:** 在仿真和增强学习(RL)中,人力监督的成本限制了机器人可以进行训练的数据量。 RL提供了通过试验和误差自动学习的自适应机器人的构架,而实际实现最终需要广泛的人力监督来设计奖励函数和在交互事件之间重复重新设置环境。 本文提出了自适应机器人系统的新设计 MEDAL++:首先给出了一个小组专家演示,机器人通过学习做和取消任务,同时从演示中推导奖励函数来自适应任务。我们首先在仿真的非 episodic benchmark EARL 上评估了我们提出的系统,发现 MEDAL++ 既具有较好的数据效率,而且在最先进的视力based 方法上达到30%的最终性能。 我们的实物机器人实验表明, MEDAL++ 可以应用于比以往工作所考虑的更大型环境的操纵问题,而自主自我改进则能够提高成功率30%到70%,比只凭专家数据进行行为克隆。

**[Paper URL](https://proceedings.mlr.press/v229/sharma23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sharma23b/sharma23b.pdf)** 

# Equivariant Reinforcement Learning under Partial Observability
**题目:** 部分可观察性下的等价增强学习

**作者:** Hai Huu Nguyen, Andrea Baisero, David Klee, Dian Wang, Robert Platt, Christopher Amato

**Abstract:** Incorporating inductive biases is a promising approach for tackling challenging robot learning domains with sample-efficient solutions. This paper identifies partially observable domains where symmetries can be a useful inductive bias for efficient learning. Specifically, by encoding the equivariance regarding specific group symmetries into the neural networks, our actor-critic reinforcement learning agents can reuse solutions in the past for related scenarios. Consequently, our equivariant agents outperform non-equivariant approaches significantly in terms of sample efficiency and final performance, demonstrated through experiments on a range of robotic tasks in simulation and real hardware.

**摘要:** 引入诱导性偏见是解决具有样本效率的机器人学习领域问题的一种有前途的途径。本文指出了部分可观察的领域,其中对称可以成为有效的诱导性偏见。具体地说,通过将有关特定群体对称的等效性编码到神经网络中,我们的参与者-关键的强化学习代理可以重用过去对相关场景的解决方案。因此,我们的等效性代理在样本效率和最终性能方面明显优于非等效性方法。

**[Paper URL](https://proceedings.mlr.press/v229/nguyen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/nguyen23a/nguyen23a.pdf)** 

# UniFolding: Towards Sample-efficient, Scalable, and Generalizable Robotic Garment Folding
**题目:** 单发:面向高效、可扩展和通用的机器人服装折叠

**作者:** Han Xue, Yutong Li, Wenqiang Xu, Huanyu Li, Dongzhe Zheng, Cewu Lu

**Abstract:** This paper explores the development of UniFolding, a sample-efficient, scalable, and generalizable robotic system for unfolding and folding various garments. UniFolding employs the proposed UFONet neural network to integrate unfolding and folding decisions into a single policy model that is adaptable to different garment types and states. The design of UniFolding is based on a garment’s partial point cloud, which aids in generalization and reduces sensitivity to variations in texture and shape. The training pipeline prioritizes low-cost, sample-efficient data collection. Training data is collected via a human-centric process with offline and online stages. The offline stage involves human unfolding and folding actions via Virtual Reality, while the online stage utilizes human-in-the-loop learning to fine-tune the model in a real-world setting. The system is tested on two garment types: long-sleeve and short-sleeve shirts. Performance is evaluated on 20 shirts with significant variations in textures, shapes, and materials. More experiments and videos can be found in the supplementary materials and on the website: https://unifolding.robotflow.ai.

**摘要:** 本文研究了一种用于展开和折叠各种服装的样品高效、可扩展和通用的机器人系统,即UniFolding。UniFolding采用了拟议的UFONet神经网络,将展开和折叠决策集成到一个可适应不同服装类型和状态的单一政策模型中。UniFolding的设计基于服装的局部点云,有助于推广和减少对纹理和形状变化的敏感性。培训管道优先考虑低成本、高效的样品数据收集。对20件衬衫的性能进行了评估,其纹理、形状和材料有明显的差异。更多的实验和视频可以在补充材料和网站上找到: https://unifolding.robotflow.ai。

**[Paper URL](https://proceedings.mlr.press/v229/xue23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xue23b/xue23b.pdf)** 

# A Universal Semantic-Geometric Representation for Robotic Manipulation
**题目:** 机器人操纵的通用语义-几何表示

**作者:** Tong Zhang, Yingdong Hu, Hanchen Cui, Hang Zhao, Yang Gao

**Abstract:** Robots rely heavily on sensors, especially RGB and depth cameras, to perceive and interact with the world. RGB cameras record 2D images with rich semantic information while missing precise spatial information. On the other side, depth cameras offer critical 3D geometry data but capture limited semantics. Therefore, integrating both modalities is crucial for learning representations for robotic perception and control. However, current research predominantly focuses on only one of these modalities, neglecting the benefits of incorporating both. To this end, we present Semantic-Geometric Representation (SGR), a universal perception module for robotics that leverages the rich semantic information of large-scale pre-trained 2D models and inherits the merits of 3D spatial reasoning. Our experiments demonstrate that SGR empowers the agent to successfully complete a diverse range of simulated and real-world robotic manipulation tasks, outperforming state-of-the-art methods significantly in both single-task and multi-task settings. Furthermore, SGR possesses the capability to generalize to novel semantic attributes, setting it apart from the other methods. Project website: https://semantic-geometric-representation.github.io.

**摘要:** RGB摄像机记录有丰富的语义信息的2D图像,同时缺少精确的空间信息。另一方面,深层摄像机提供关键的3D几何数据,但捕获有限的语义。因此,整合这两个模式对于学习机器人的感知和控制是至关重要的。然而,目前的研究主要集中在其中一个模式上,忽略了两个模式的好处。为此,我们提出了语义-几何表示(SGR),一种用于机器人的普遍感知模块,它利用大规模预训练的2D模型的丰富语义信息,并继承了3D空间推理的优点。我们的实验证明,SGR能使代理人成功完成多种模拟和现实机器人操作任务,在单任务和多任务设置中大大超过了最先进的方法。此外,SGR具有将新语义属性推广到新语义属性的能力,使其与其他方法不同。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23j.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23j/zhang23j.pdf)** 

# LabelFormer: Object Trajectory Refinement for Offboard Perception from LiDAR Point Clouds
**题目:** LabelFormer:LiDAR点云的离岸感知对象推理精细

**作者:** Anqi Joyce Yang, Sergio Casas, Nikita Dvornik, Sean Segal, Yuwen Xiong, Jordan Sir Kwang Hu, Carter Fang, Raquel Urtasun

**Abstract:** A major bottleneck to scaling-up training of self-driving perception systems are the human annotations required for supervision. A promising alternative is to leverage “auto-labelling" offboard perception models that are trained to automatically generate annotations from raw LiDAR point clouds at a fraction of the cost. Auto-labels are most commonly generated via a two-stage approach – first objects are detected and tracked over time, and then each object trajectory is passed to a learned refinement model to improve accuracy. Since existing refinement models are overly complex and lack advanced temporal reasoning capabilities, in this work we propose LabelFormer, a simple, efficient, and effective trajectory-level refinement approach. Our approach first encodes each frame’s observations separately, then exploits self-attention to reason about the trajectory with full temporal context, and finally decodes the refined object size and per-frame poses. Evaluation on both urban and highway datasets demonstrates that LabelFormer outperforms existing works by a large margin. Finally, we show that training on a dataset augmented with auto-labels generated by our method leads to improved downstream detection performance compared to existing methods. Please visit the project website for details https://waabi.ai/labelformer/.

**摘要:** 自我驱动感知系统扩充训练的一个主要瓶颈是需要进行监督的人类注释。一个有前途的替代方案是利用“自动标记”的离岸感知模型,这些模型被训练以自动生成来自原始LiDAR点云的注释,费用仅小于成本。自动标记最常见的方法是通过两个阶段的方法生成 – 首先对象被检测和追踪,然后每个对象的轨迹被传递到一个学习的改进模型来提高精度。对城市和公路数据集的评估表明,LabelFormer比现有的工作更优于现有的工作。最后,我们证明,通过我们方法生成的自动标签增强的数据集的训练,与现有方法相比,可以提高下游检测性能。详情请浏览项目网站 https://waabi.ai/labelformer/。

**[Paper URL](https://proceedings.mlr.press/v229/yang23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23e/yang23e.pdf)** 

# Language-Conditioned Path Planning
**题目:** 语言条件路径规划

**作者:** Amber Xie, Youngwoon Lee, Pieter Abbeel, Stephen James

**Abstract:** Contact is at the core of robotic manipulation. At times, it is desired (e.g. manipulation and grasping), and at times, it is harmful (e.g. when avoiding obstacles). However, traditional path planning algorithms focus solely on collision-free paths, limiting their applicability in contact-rich tasks. To address this limitation, we propose the domain of Language-Conditioned Path Planning, where contact-awareness is incorporated into the path planning problem. As a first step in this domain, we propose Language-Conditioned Collision Functions (LACO), a novel approach that learns a collision function using only a single-view image, language prompt, and robot configuration. LACO predicts collisions between the robot and the environment, enabling flexible, conditional path planning without the need for manual object annotations, point cloud data, or ground-truth object meshes. In both simulation and the real world, we demonstrate that LACO can facilitate complex, nuanced path plans that allow for interaction with objects that are safe to collide, rather than prohibiting any collision.

**摘要:** 接触是机器人操纵的核心,有时是需要的(例如操纵和把握),有时是有害的(例如避免障碍)。然而,传统的路径规划算法只关注无碰撞路径,限制了它们在接触丰富的任务中的应用。为了解决这一限制,我们提出了语言约束路径规划的领域,其中接触意识被纳入路径规划问题。作为这一领域的第一步,我们提出了语言约束碰撞函数(LACO),一种新颖的方法,它只使用单视图图像、语言提示和机器人配置学习碰撞函数。LACO预测机器人与环境之间的碰撞,允许灵活、条件的路径规划,不需要手动对象注释、点云数据或地面实物对象网格。在模拟和现实世界中,我们证明了LACO能够促进复杂、细微的路径规划,允许与安全碰撞的对象进行交互,而不是禁止任何碰撞。

**[Paper URL](https://proceedings.mlr.press/v229/xie23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xie23b/xie23b.pdf)** 

# Open-World Object Manipulation using Pre-Trained Vision-Language Models
**题目:** 使用预训练视觉语言模型进行开放世界对象操作

**作者:** Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, Chelsea Finn, Karol Hausman

**Abstract:** For robots to follow instructions from people, they must be able to connect the rich semantic information in human vocabulary, e.g. “can you get me the pink stuffed whale?” to their sensory observations and actions. This brings up a notably difficult challenge for robots: while robot learning approaches allow robots to learn many different behaviors from first-hand experience, it is impractical for robots to have first-hand experiences that span all of this semantic information. We would like a robot’s policy to be able to perceive and pick up the pink stuffed whale, even if it has never seen any data interacting with a stuffed whale before. Fortunately, static data on the internet has vast semantic information, and this information is captured in pre-trained vision-language models. In this paper, we study whether we can interface robot policies with these pre-trained models, with the aim of allowing robots to complete instructions involving object categories that the robot has never seen first-hand. We develop a simple approach, which we call Manipulation of Open-World Objects (MOO), which leverages a pre-trained vision-language model to extract object-identifying information from the language command and image, and conditions the robot policy on the current image, the instruction, and the extracted object information. In a variety of experiments on a real mobile manipulator, we find that MOO generalizes zero-shot to a wide range of novel object categories and environments. In addition, we show how MOO generalizes to other, non-language-based input modalities to specify the object of interest such as finger pointing, and how it can be further extended to enable open-world navigation and manipulation. The project’s website and evaluation videos can be found at https://robot-moo.github.io/.

**摘要:** 为了机器人遵循人们的指导,他们必须能够将人类词汇中丰富的语义信息,例如“你能帮我找到粉红色的填海鲸吗?”连接到他们的感官观察和行动。这给机器人带来了一个特别困难的挑战:虽然机器人学习方法允许机器人从亲身经验中学习许多不同的行为,但对于机器人来说,拥有所有这些语义信息的亲身经验是不实际的。本文研究了如何将机器人政策与这些预训练模型结合起来,以使机器人能够完成机器人从未亲眼见过的对象类别的指令。我们开发了一种简单的方法,叫做“开放世界对象操作”(MOO),它利用预训练的视觉语言模型,从语言命令和图像中提取对象识别信息,并条件下对当前图像、指令和提取对象信息的机器人政策。该项目网站和评估视频可以在 https://robot-moo.github.io/找到。

**[Paper URL](https://proceedings.mlr.press/v229/stone23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/stone23a/stone23a.pdf)** 

# Learning Generalizable Manipulation Policies with Object-Centric 3D Representations
**题目:** 基于对象中心的3D表示的通用操作策略学习

**作者:** Yifeng Zhu, Zhenyu Jiang, Peter Stone, Yuke Zhu

**Abstract:** We introduce GROOT, an imitation learning method for learning robust policies with object-centric and 3D priors. GROOT builds policies that generalize beyond their initial training conditions for vision-based manipulation. It constructs object-centric 3D representations that are robust toward background changes and camera views and reason over these representations using a transformer-based policy. Furthermore, we introduce a segmentation correspondence model that allows policies to generalize to new objects at test time. Through comprehensive experiments, we validate the robustness of GROOT policies against perceptual variations in simulated and real-world environments. GROOT’s performance excels in generalization over background changes, camera viewpoint shifts, and the presence of new object instances, whereas both state-of-the-art end-to-end learning methods and object proposal-based approaches fall short. We also extensively evaluate GROOT policies on real robots, where we demonstrate the efficacy under very wild changes in setup. More videos and model details can be found in the appendix and the project website https://ut-austin-rpl.github.io/GROOT.

**摘要:** 介绍了一种基于对象和3D预先的仿真学习方法,用于学习基于视觉操作的强有力政策。该方法构建基于变换器的强有力的面向背景变化和摄像机视图的对象3D表示,并利用这些表示的推理。此外,我们引入了一种基于变换器的分区匹配模型,允许政策在测试时对新的对象进行一般化。通过综合实验,我们验证了 GROOT政策在模拟和现实环境中的感知变异的强有力性。 GROOT的性能优于基于背景变化、摄像机视角转移和新对象实例的一般化,而最先进的端到端学习方法和基于对象提案的方法都不足之处。我们还对真正的机器人的GROOT政策进行了广泛的评估,在设置的非常野蛮变化下,我们证明了 GROOT的有效性。更多视频和模型细节可以在附件和项目网站 https://ut-austin-rpl.github.io/GROOT中找到。

**[Paper URL](https://proceedings.mlr.press/v229/zhu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhu23b/zhu23b.pdf)** 

# AdaptSim: Task-Driven Simulation Adaptation for Sim-to-Real Transfer
**题目:** AdaptSim:基于任务的模拟 adaptation for Sim-to-Real Transfer

**作者:** Allen Z. Ren, Hongkai Dai, Benjamin Burchfiel, Anirudha Majumdar

**Abstract:** Simulation parameter settings such as contact models and object geometry approximations are critical to training robust manipulation policies capable of transferring from simulation to real-world deployment. There is often an irreducible gap between simulation and reality: attempting to match the dynamics between simulation and reality may be infeasible and may not lead to policies that perform well in reality for a specific task. We propose AdaptSim, a new task-driven adaptation framework for sim-to-real transfer that aims to optimize task performance in target (real) environments. First, we meta-learn an adaptation policy in simulation using reinforcement learning for adjusting the simulation parameter distribution based on the current policy’s performance in a target environment. We then perform iterative real-world adaptation by inferring new simulation parameter distributions for policy training. Our extensive simulation and hardware experiments demonstrate AdaptSim achieving 1-3x asymptotic performance and 2x real data efficiency when adapting to different environments, compared to methods based on Sys-ID and directly training the task policy in target environments.

**摘要:** 仿真参数设置,例如接触模型和对象几何近似,对于训练能够从仿真到现实部署的鲁棒操纵政策至关重要:仿真与现实之间往往存在不可缩减的差距:试图匹配仿真与现实之间的动力学可能是不可能的,可能不会导致在特定任务的现实中表现良好的政策。我们提出了AdaptSim,一种旨在优化目标(现实)环境中的任务性能的新任务驱动的模拟变换框架。首先,我们利用增强学习来在仿真中学习一种适应策略,根据当前政策在目标环境中的表现,调整仿真参数分布。然后,我们通过推导新的仿真参数分布来进行迭代现实适应。我们的广泛的仿真和硬件实验证明了AdapSim在适应不同环境时达到1-3倍的渐近性能和2倍的实际数据效率,与基于Sys-ID的方法和直接培训目标环境的任务政策相比。

**[Paper URL](https://proceedings.mlr.press/v229/ren23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ren23b/ren23b.pdf)** 

# Dexterous Functional Grasping
**题目:** 精巧的函数测井

**作者:** Ananye Agarwal, Shagun Uppal, Kenneth Shaw, Deepak Pathak

**Abstract:** While there have been significant strides in dexterous manipulation, most of it is limited to benchmark tasks like in-hand reorientation which are of limited utility in the real world. The main benefit of dexterous hands over two-fingered ones is their ability to pickup tools and other objects (including thin ones) and grasp them firmly in order to apply force. However, this task requires both a complex understanding of functional affordances as well as precise low-level control. While prior work obtains affordances from human data this approach doesn’t scale to low-level control. Similarly, simulation training cannot give the robot an understanding of real-world semantics. In this paper, we aim to combine the best of both worlds to accomplish functional grasping for in-the-wild objects. We use a modular approach. First, affordances are obtained by matching corresponding regions of different objects and then a low-level policy trained in sim is run to grasp it. We propose a novel application of eigengrasps to reduce the search space of RL using a small amount of human data and find that it leads to more stable and physically realistic motion. We find that eigengrasp action space beats baselines in simulation and outperforms hardcoded grasping in real and matches or outperforms a trained human teleoperator. Videos at https://dexfunc.github.io/.

**摘要:** 尽管在精巧操作中取得了重大进展,但其中大部分仅限于在实际世界中具有有限实用价值的手柄重定位等基准任务。精巧手柄在两指上的主要好处是它们能够抓取工具和其他对象(包括薄型对象)并牢牢抓住它们,以便运用力。然而,这项任务既要求复杂的功能 affordances的理解,也要求精确的低级控制。我们提出了一种新颖的 Eigengrasps应用,以减少RL的搜索空间,使用少量的人类数据,并发现它会导致更稳定的物理现实运动。

**[Paper URL](https://proceedings.mlr.press/v229/agarwal23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/agarwal23a/agarwal23a.pdf)** 

# REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction
**题目:** 选择:总结机器人故障解释和纠正经验

**作者:** Zeyi Liu, Arpit Bahety, Shuran Song

**Abstract:** The ability to detect and analyze failed executions automatically is crucial for an explainable and robust robotic system. Recently, Large Language Models (LLMs) have demonstrated strong reasoning abilities on textual inputs. To leverage the power of LLMs for robot failure explanation, we introduce REFLECT, a framework which queries LLM for failure reasoning based on a hierarchical summary of robot past experiences generated from multisensory observations. The failure explanation can further guide a language-based planner to correct the failure and complete the task. To systematically evaluate the framework, we create the RoboFail dataset with a variety of tasks and failure scenarios. We demonstrate that the LLM-based framework is able to generate informative failure explanations that assist successful correction planning.

**摘要:** 自动检测和分析失败执行的能力对于一个可解释和强有力的机器人系统至关重要。最近,大型语言模型(LLMs)已经在文本输入上显示出强大的推理能力。为了利用LLMs对机器人失败解释的能力,我们引入了REFLECT,一种基于多感知观测的机器人过去的经验的层次性总结来查询LLM的失败推理框架。该框架可以进一步指导语言规划者纠正失败并完成任务。为了系统地评估该框架,我们创建了RoboFail数据集,包含各种任务和失败场景。我们证明,基于LLM的框架能够产生有益的失败解释,以协助成功纠正规划。

**[Paper URL](https://proceedings.mlr.press/v229/liu23g.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23g/liu23g.pdf)** 

# Task Generalization with Stability Guarantees via Elastic Dynamical System Motion Policies
**题目:** 通过弹性动态系统运动政策实现任务通用化与稳定性保证

**作者:** Tianyu Li, Nadia Figueroa

**Abstract:** Dynamical System (DS) based Learning from Demonstration (LfD) allows learning of reactive motion policies with stability and convergence guarantees from a few trajectories. Yet, current DS learning techniques lack the flexibility to generalize to new task instances as they overlook explicit task parameters that inherently change the underlying demonstrated trajectories. In this work, we propose Elastic-DS, a novel DS learning and generalization approach that embeds task parameters into the Gaussian Mixture Model (GMM) based Linear Parameter Varying (LPV) DS formulation. Central to our approach is the Elastic-GMM, a GMM constrained to SE(3) task-relevant frames. Given a new task instance/context, the Elastic-GMM is transformed with Laplacian Editing and used to re-estimate the LPV-DS policy. Elastic-DS is compositional in nature and can be used to construct flexible multi-step tasks. We showcase its strength on a myriad of simulated and real-robot experiments while preserving desirable control-theoretic guarantees.

**摘要:** 动态系统(DS)基于演示学习(LfD)允许从少数轨迹中学习有稳定性和收敛性保证的反动运动政策。然而,当前的DS学习技术缺乏对新任务实例的一般化灵活性,因为它们忽略了明确的任务参数,这些参数本身改变了潜在的演示轨迹。在这个工作中,我们提出了一种新的DS学习和一般化方法,它将任务参数嵌入到基于高斯混合模型(GMM)的线性参数变异(LPV)DS公式中。我们展示了它在无数模拟和实际机器人实验中的优势,同时保留了理想的控制理论保证。

**[Paper URL](https://proceedings.mlr.press/v229/li23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/li23b/li23b.pdf)** 

# Push Past Green: Learning to Look Behind Plant Foliage by Moving It
**题目:** 推动过去绿色:通过移动植物叶子的背后学习

**作者:** Xiaoyu Zhang, Saurabh Gupta

**Abstract:** Autonomous agriculture applications (e.g., inspection, phenotyping, plucking fruits) require manipulating the plant foliage to look behind the leaves and the branches. Partial visibility, extreme clutter, thin structures, and unknown geometry and dynamics for plants make such manipulation challenging. We tackle these challenges through data-driven methods. We use self-supervision to train SRPNet, a neural network that predicts what space is revealed on execution of a candidate action on a given plant. We use SRPNet with the cross-entropy method to predict actions that are effective at revealing space beneath plant foliage. Furthermore, as SRPNet does not just predict how much space is revealed but also where it is revealed, we can execute a sequence of actions that incrementally reveal more and more space beneath the plant foliage. We experiment with a synthetic (vines) and a real plant (Dracaena) on a physical test-bed across 5 settings including 2 settings that test generalization to novel plant configurations. Our experiments reveal the effectiveness of our overall method, PPG, over a competitive hand-crafted exploration method, and the effectiveness of SRPNet over a hand-crafted dynamics model and relevant ablations. Project website with execution videos, code, data, and models: https://sites.google.com/view/pushingfoliage/.

**摘要:** 自主农业应用(例如检查、标记、摘果等)需要操纵植物叶子,以观察叶子和树枝的背后。部分可视性、极端杂乱、薄结构和未知的植物几何和动力学使得这种操纵具有挑战性。我们通过数据驱动的方法来解决这些挑战。我们使用自我监督来训练SRPNet,一种神经网络,它预测在某一植物的候选行动执行时所发现的空间。我们使用跨热带方法来预测在植物叶子下的空间的有效性。我们用合成(葡萄)和实物植物(Dracaena)在5个设置中进行物理测试,包括2个设置,测试对新植物配置的一般化。我们的实验揭示了我们的整体方法,PPG,在竞争性手工制作的勘探方法上,以及在手工制作的动力学模型上SRPNet的有效性和相关调试。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23k.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23k/zhang23k.pdf)** 

# XSkill: Cross Embodiment Skill Discovery
**题目:** XSkill:跨体积技能探索

**作者:** Mengda Xu, Zhenjia Xu, Cheng Chi, Manuela Veloso, Shuran Song

**Abstract:** Human demonstration videos are a widely available data source for robot learning and an intuitive user interface for expressing desired behavior. However, directly extracting reusable robot manipulation skills from unstructured human videos is challenging due to the big embodiment difference and unobserved action parameters. To bridge this embodiment gap, this paper introduces XSkill, an imitation learning framework that 1) discovers a cross-embodiment representation called skill prototypes purely from unlabeled human and robot manipulation videos, 2) transfers the skill representation to robot actions using conditional diffusion policy, and finally, 3) composes the learned skill to accomplish unseen tasks specified by a human prompt video. Our experiments in simulation and real-world environments show that the discovered skill prototypes facilitate both skill transfer and composition for unseen tasks, resulting in a more general and scalable imitation learning framework.

**摘要:** 人类演示视频是机器人学习的广泛可用数据源,也是表达行为的直观用户界面。然而,直接从没有结构的人类视频中提取可再利用的机器人操纵技能是由于体积差异和未观察的动作参数所引起的挑战。为了弥补这种体积缺口,本文介绍了一种模仿学习框架 XSkill,该框架 1) 从未标记的人类和机器人操纵视频中发现一种称为“技能原型”的交叉体积表示, 2) 通过条件扩散政策将技能表示转移到机器人动作中,最后, 3) 将学习技能编译成以完成由人类快速视频指定的未观察任务。

**[Paper URL](https://proceedings.mlr.press/v229/xu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xu23a/xu23a.pdf)** 

# SayTap: Language to Quadrupedal Locomotion
**题目:** SayTap: 四面体运动语言

**作者:** Yujin Tang, Wenhao Yu, Jie Tan, Heiga Zen, Aleksandra Faust, Tatsuya Harada

**Abstract:** Large language models (LLMs) have demonstrated the potential to perform high-level planning. Yet, it remains a challenge for LLMs to comprehend low-level commands, such as joint angle targets or motor torques. This paper proposes an approach to use foot contact patterns as an interface that bridges human commands in natural language and a locomotion controller that outputs these low-level commands. This results in an interactive system for quadrupedal robots that allows the users to craft diverse locomotion behaviors flexibly. We contribute an LLM prompt design, a reward function, and a method to expose the controller to the feasible distribution of contact patterns. The results are a controller capable of achieving diverse locomotion patterns that can be transferred to real robot hardware. Compared with other design choices, the proposed approach enjoys more than $50%$ success rate in predicting the correct contact patterns and can solve 10 more tasks out of a total of 30 tasks. (https://saytap.github.io)

**摘要:** 大型语言模型(英语:Large language models,LLMs)已经展示了执行高层次规划的潜力。然而,对于LLMs来说,理解低层次命令仍然是一个挑战,例如联合角度目标或电动转矩。本文提出了一种使用脚接触模式作为自然语言中的人类命令桥梁和输出这些低层次命令的运动控制器的方法。这导致了四节机器人的交互系统,允许用户灵活地创造各种运动行为。我们为LLM快速设计、奖励函数和控制器暴露接触模式的可行分配方法作出贡献。结果是能够实现多种运动模式的控制器,可以转移到真正的机器人硬件。(https://saytap.github.io)

**[Paper URL](https://proceedings.mlr.press/v229/tang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tang23a/tang23a.pdf)** 

# SLAP: Spatial-Language Attention Policies
**题目:** SLAP: Spatial-Language Attention Policies

**作者:** Priyam Parashar, Vidhi Jain, Xiaohan Zhang, Jay Vakil, Sam Powers, Yonatan Bisk, Chris Paxton

**Abstract:** Despite great strides in language-guided manipulation, existing work has been constrained to table-top settings. Table-tops allow for perfect and consistent camera angles, properties are that do not hold in mobile manipulation. Task plans that involve moving around the environment must be robust to egocentric views and changes in the plane and angle of grasp. A further challenge is ensuring this is all true while still being able to learn skills efficiently from limited data. We propose Spatial-Language Attention Policies (SLAP) as a solution. SLAP uses three-dimensional tokens as the input representation to train a single multi-task, language-conditioned action prediction policy. Our method shows an $80%$ success rate in the real world across eight tasks with a single model, and a $47.5%$ success rate when unseen clutter and unseen object configurations are introduced, even with only a handful of examples per task. This represents an improvement of $30%$ over prior work ($20%$ given unseen distractors and configurations). We see a 4x improvement over baseline in mobile manipulation setting. In addition, we show how SLAPs robustness allows us to execute Task Plans from open-vocabulary instructions using a large language model for multi-step mobile manipulation. For videos, see the website: https://robotslap.github.io

**摘要:** 尽管语言指导的操作取得了巨大的进步,但现有的工作已经限制在表顶设置上。表顶允许完美的和一致的摄像机角度,属性是不能在移动操作中保持的。环境周围移动的任务计划必须对自我中心的视角和把握平面和角度的变化有强力。另一个挑战是确保所有这些都是正确的,同时仍然能够从有限的数据有效学习技能。我们提出了空间语言注意政策(SLAP)作为解决方案。SLAP使用三维图标作为输入表示,训练一个单个多任务,语言条件下的行动预测政策。这代表了在以前的工作上30%的改进($20%是未见的分散器和配置)。我们看到移动操作设置的基线上的4倍的改进。此外,我们还展示了SLAPs的鲁棒性让我们能够从开放语音指令中执行多步移动操作的大型语言模型。

**[Paper URL](https://proceedings.mlr.press/v229/parashar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/parashar23a/parashar23a.pdf)** 

# Learning Human Contribution Preferences in Collaborative Human-Robot Tasks
**题目:** 人类-机器人协作任务中学习人类贡献的偏好

**作者:** Michelle D Zhao, Reid Simmons, Henny Admoni

**Abstract:** In human-robot collaboration, both human and robotic agents must work together to achieve a set of shared objectives. However, each team member may have individual preferences, or constraints, for how they would like to contribute to the task. Effective teams align their actions to optimize task performance while satisfying each team member’s constraints to the greatest extent possible. We propose a framework for representing human and robot contribution constraints in collaborative human-robot tasks. Additionally, we present an approach for learning a human partner’s contribution constraint online during a collaborative interaction. We evaluate our approach using a variety of simulated human partners in a collaborative decluttering task. Our results demonstrate that our method improves team performance over baselines with some, but not all, simulated human partners. Furthermore, we conducted a pilot user study to gather preliminary insights into the effectiveness of our approach on task performance and collaborative fluency. Preliminary results suggest that pilot users performed fluently with our method, motivating further investigation into considering preferences that emerge from collaborative interactions.

**摘要:** 在人与机器人协作中,人与机器人的代理人必须共同工作,以达成共同的目标。然而,每个团队成员可能有各自的偏好或限制,以决定他们如何参与任务。有效的团队会协调行动,以优化任务表现,同时尽可能满足每个团队成员的限制。我们提出了一种框架,以代表人与机器人在合作人与机器人任务中所承担的贡献限制。此外,我们还提出了一种方法,以在线学习合作互动中人与机器人的贡献限制。此外,我们进行了一次试验用户研究,以收集初步了解我们方法对任务性能和协作流畅的有效性。初步结果表明,试验用户使用我们的方法流畅,从而促使进一步研究考虑协作互动产生的偏好。

**[Paper URL](https://proceedings.mlr.press/v229/zhao23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhao23b/zhao23b.pdf)** 

# M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place
**题目:** M2T2:面向对象选择和定位的多任务面具变换器

**作者:** Wentao Yuan, Adithyavairavan Murali, Arsalan Mousavian, Dieter Fox

**Abstract:** With the advent of large language models and large-scale robotic datasets, there has been tremendous progress in high-level decision-making for object manipulation. These generic models are able to interpret complex tasks using language commands, but they often have difficulties generalizing to out-of-distribution objects due to the inability of low-level action primitives. In contrast, existing task-specific models excel in low-level manipulation of unknown objects, but only work for a single type of action. To bridge this gap, we present M2T2, a single model that supplies different types of low-level actions that work robustly on arbitrary objects in cluttered scenes. M2T2 is a transformer model which reasons about contact points and predicts valid gripper poses for different action modes given a raw point cloud of the scene. Trained on a large-scale synthetic dataset with 128K scenes, M2T2 achieves zero-shot sim2real transfer on the real robot, outperforming the baseline system with state-of-the-art task-specific models by about $19%$ in overall performance and $37.5%$ in challenging scenes were the object needs to be re-oriented for collision-free placement. M2T2 also achieves state-of-the-art results on a subset of language conditioned tasks in RLBench. Videos of robot experiments on unseen objects in both real world and simulation are available at m2-t2.github.io.

**摘要:** 随着大型语言模型和大规模机器人数据集的出现,对对象操作的高级决策取得了巨大的进展。这些通用模型能够使用语言命令解释复杂的任务,但由于低级行动原型无法将它们推广到非分布对象时,它们经常有困难。相反,现有的任务特有模型在未知对象的低级操作中优越,但只适用于单一类型的操作。M2T2在128K场景的大规模合成数据集上训练,实现了实物机器人的零射 sim2real传输,超过了最先进的任务特定模型的基线系统,总性能约为$19%和挑战场景的$37.5%,这些对象需要重新定位,以避免碰撞。 M2T2还实现了在RLBench的语言条件下任务的最先进的结果。

**[Paper URL](https://proceedings.mlr.press/v229/yuan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yuan23a/yuan23a.pdf)** 

# Learning to Drive Anywhere
**题目:** 学习开车任何地方

**作者:** Ruizhao Zhu, Peng Huang, Eshed Ohn-Bar, Venkatesh Saligrama

**Abstract:** Human drivers can seamlessly adapt their driving decisions across geographical locations with diverse conditions and rules of the road, e.g., left vs. right-hand traffic. In contrast, existing models for autonomous driving have been thus far only deployed within restricted operational domains, i.e., without accounting for varying driving behaviors across locations or model scalability. In this work, we propose GeCo, a single geographically-aware conditional imitation learning (CIL) model that can efficiently learn from heterogeneous and globally distributed data with dynamic environmental, traffic, and social characteristics. Our key insight is to introduce a high-capacity, geo-location-based channel attention mechanism that effectively adapts to local nuances while also flexibly modeling similarities among regions in a data-driven manner. By optimizing a contrastive imitation objective, our proposed approach can efficiently scale across the inherently imbalanced data distributions and location-dependent events. We demonstrate the benefits of our GeCo agent across multiple datasets, cities, and scalable deployment paradigms, i.e., centralized, semi-supervised, and distributed agent training. Specifically, GeCo outperforms CIL baselines by over $14%$ in open-loop evaluation and $30%$ in closed-loop testing on CARLA.

**摘要:** 人类驾驶者可以在地理区域内随道路的不同条件和规则,例如左对右交通,无缝地调整驾驶决策。相反,现有的自主驾驶模型迄今只在有限的操作域内部署,即没有考虑到在不同区域内驾驶行为或模型可扩展性。通过优化对比仿真目标,我们提出的方法能够有效地扩展到本质上不平衡的数据分布和基于地点的事件。我们展示了我们的GeCo代理在多个数据集、城市和可扩展部署范式(即集中、半监督和分布代理培训)中的优势。具体而言,GeCo在开环评价中超过CIL基线14美元,在闭环测试中超过30%。

**[Paper URL](https://proceedings.mlr.press/v229/zhu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhu23c/zhu23c.pdf)** 

# MOTO: Offline Pre-training to Online Fine-tuning for Model-based Robot Learning
**题目:** MOTO:基于模型的机器人学习的在线预训练

**作者:** Rafael Rafailov, Kyle Beltran Hatch, Victor Kolev, John D. Martin, Mariano Phielipp, Chelsea Finn

**Abstract:** We study the problem of offline pre-training and online fine-tuning for reinforcement learning from high-dimensional observations in the context of realistic robot tasks. Recent offline model-free approaches successfully use online fine-tuning to either improve the performance of the agent over the data collection policy or adapt to novel tasks. At the same time, model-based RL algorithms have achieved significant progress in sample efficiency and the complexity of the tasks they can solve, yet remain under-utilized in the fine-tuning setting. In this work, we argue that existing methods for high-dimensional model-based offline RL are not suitable for offline-to-online fine-tuning due to issues with distribution shifts, off-dynamics data, and non-stationary rewards. We propose an on-policy model-based method that can efficiently reuse prior data through model-based value expansion and policy regularization, while preventing model exploitation by controlling epistemic uncertainty. We find that our approach successfully solves tasks from the MetaWorld benchmark, as well as the Franka Kitchen robot manipulation environment completely from images. To our knowledge, MOTO is the first and only method to solve this environment from pixels.

**摘要:** 我们研究了在现实机器人任务中,从高维观测中进行增强学习的在线预训练和在线微调问题。最近的在线模型自由方法成功地使用在线微调,以提高代理在数据收集政策上的表现或适应新任务。同时,基于模型的RL算法在样本效率和解决任务的复杂性方面取得了重大进展,但仍未被微调设置中充分利用。我们发现,我们的方法成功地解决了MetaWorld基准的任务,以及Franka Kitchen机器人操纵环境完全从图像中。我们知道,MOT是第一个和唯一的方法从像素解决这个环境。

**[Paper URL](https://proceedings.mlr.press/v229/rafailov23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rafailov23a/rafailov23a.pdf)** 

# Ready, Set, Plan! Planning to Goal Sets Using Generalized Bayesian Inference
**题目:** 准备好, 设置, 计划! 利用广义贝叶斯惯性规划目标设置

**作者:** Jana Pavlasek, Stanley Robert Lewis, Balakumar Sundaralingam, Fabio Ramos, Tucker Hermans

**Abstract:** Many robotic tasks can have multiple and diverse solutions and, as such, are naturally expressed as goal sets. Examples include navigating to a room, finding a feasible placement location for an object, or opening a drawer enough to reach inside. Using a goal set as a planning objective requires that a model for the objective be explicitly given by the user. However, some goals are intractable to model, leading to uncertainty over the goal (e.g. stable grasping of an object). In this work, we propose a technique for planning directly to a set of sampled goal configurations. We formulate a planning as inference problem with a novel goal likelihood evaluated against the goal samples. To handle the intractable goal likelihood, we employ Generalized Bayesian Inference to approximate the trajectory distribution. The result is a fully differentiable cost which generalizes across a diverse range of goal set objectives for which samples can be obtained. We show that by considering all goal samples throughout the planning process, our method reliably finds plans on manipulation and navigation problems where heuristic approaches fail.

**摘要:** 许多机器人任务可以具有多种和多样的解决方案,因此,它们自然地表达为目标设置。例如,导航到一个房间,寻找一个对象可行的配置位置,或打开一个足够接近内部的抽屉。使用目标设置为规划目标需要用户明确给出目标的模型。然而,一些目标是无法解决的模型,导致目标上的不确定性(例如目标的稳定把握)。我们证明,通过在整个规划过程中考虑所有目标样本,我们的方法能够可靠地找到在实验方法失败时的操纵和导航问题方案。

**[Paper URL](https://proceedings.mlr.press/v229/pavlasek23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/pavlasek23a/pavlasek23a.pdf)** 

# Online Model Adaptation with Feedforward Compensation
**题目:** Feedforward补偿的在线模型适应

**作者:** Abulikemu Abuduweili, Changliu Liu

**Abstract:** To cope with distribution shifts or non-stationarity in system dynamics, online adaptation algorithms have been introduced to update offline-learned prediction models in real-time. Existing online adaptation methods focus on optimizing the prediction model by utilizing feedback from the latest prediction error. Unfortunately, this feedback-based approach is susceptible to forgetting past information. This work proposes an online adaptation method with feedforward compensation, which uses critical data samples from a memory buffer, instead of the latest samples, to optimize the prediction model. We prove that the proposed approach achieves a smaller error bound compared to previously utilized methods in slow time-varying systems. We conducted experiments on several prediction tasks, which clearly illustrate the superiority of the proposed feedforward adaptation method. Furthermore, our feedforward adaptation technique is capable of estimating an uncertainty bound for predictions.

**摘要:** 为了应对系统动力学中的分布转变或非静态性,已引入在线适应算法,以实时更新在线学习预测模型。现有的在线适应方法集中于利用最新预测误差的反馈优化预测模型。不幸的是,这种基于反馈的方法容易忘记过去的信息。本文提出了一种基于反馈补偿的在线适应方法,该方法利用存储缓冲器的临界数据样本,而不是最新样本,以优化预测模型。

**[Paper URL](https://proceedings.mlr.press/v229/abuduweili23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/abuduweili23a/abuduweili23a.pdf)** 

# Generating Transferable Adversarial Simulation Scenarios for Self-Driving via Neural Rendering
**题目:** 通过神经渲染实现自驾驶的可移植敌对仿真场景

**作者:** Yasasa Abeysirigoonawardena, Kevin Xie, Chuhan Chen, Salar Hosseini Khorasgani, Ruiting Chen, Ruiqi Wang, Florian Shkurti

**Abstract:** Self-driving software pipelines include components that are learned from a significant number of training examples, yet it remains challenging to evaluate the overall system’s safety and generalization performance. Together with scaling up the real-world deployment of autonomous vehicles, it is of critical importance to automatically find simulation scenarios where the driving policies will fail. We propose a method that efficiently generates adversarial simulation scenarios for autonomous driving by solving an optimal control problem that aims to maximally perturb the policy from its nominal trajectory. Given an image-based driving policy, we show that we can inject new objects in a neural rendering representation of the deployment scene, and optimize their texture in order to generate adversarial sensor inputs to the policy. We demonstrate that adversarial scenarios discovered purely in the neural renderer (surrogate scene) can often be successfully transferred to the deployment scene, without further optimization. We demonstrate this transfer occurs both in simulated and real environments, provided the learned surrogate scene is sufficiently close to the deployment scene.

**摘要:** 自驾驶软件的管道包括从大量训练实例中学习的组件,但评估整个系统的安全和推广性能仍然是挑战性的。在扩大自驾驶车辆的实际部署的同时,自动找到驱动策略失败的仿真场景是至关重要的。我们提出了一种方法,通过解决最优控制问题,有效地生成自驾驶策略的敌对仿真场景,以最大程度扭曲该政策的名义轨迹。我们证明,纯粹在神经渲染器(替代场景)中发现的敌对场景经常可以成功地转移到部署场景,而无需进一步优化。我们证明,这种转移发生在模拟和实际环境中,只要学习的替代场景足够接近部署场景。

**[Paper URL](https://proceedings.mlr.press/v229/abeysirigoonawardena23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/abeysirigoonawardena23a/abeysirigoonawardena23a.pdf)** 

# STOW: Discrete-Frame Segmentation and Tracking of Unseen Objects for Warehouse Picking Robots
**题目:** STOW:仓库采摘机器人的离散框架分割和未见物的跟踪

**作者:** Yi Li, Muru Zhang, Markus Grotz, Kaichun Mo, Dieter Fox

**Abstract:** Segmentation and tracking of unseen object instances in discrete frames pose a significant challenge in dynamic industrial robotic contexts, such as distribution warehouses. Here, robots must handle object rearrangements, including shifting, removal, and partial occlusion by new items, and track these items after substantial temporal gaps. The task is further complicated when robots encounter objects beyond their training sets, thereby requiring the ability to segment and track previously unseen items. Considering that continuous observation is often inaccessible in such settings, our task involves working with a discrete set of frames separated by indefinite periods, during which substantial changes to the scene may occur. This task also translates to domestic robotic applications, such as table rearrangement. To address these demanding challenges, we introduce new synthetic and real-world datasets that replicate these industrial and household scenarios. Furthermore, we propose a novel paradigm for joint segmentation and tracking in discrete frames, alongside a transformer module that facilitates efficient inter-frame communication. Our approach significantly outperforms recent methods in our experiments. For additional results and videos, please visit https://sites.google.com/view/stow-corl23. Code and dataset will be released.

**摘要:** 在动态工业机器人环境中,如分布仓库中,对未见对象实例的分割和跟踪构成重大挑战。在这里,机器人必须处理对象的重新配置,包括由新对象移动、移除和部分封锁,并跟踪这些对象在重大时间缺口之后。当机器人遇到超出训练集的对象时,任务更加复杂,因此需要能够分割和跟踪以前未见的对象。考虑到在这些环境中经常无法访问的连续观察,我们的任务涉及与不确定期间分隔的离散帧组一起工作,在此期间可能发生重大场景变化。这项任务也适用于国内机器人应用,例如表的重新配置。此外,我们还提出了一种新型的联合分段和离散帧跟踪模式,与一个变换器模块相连,这有利于高效的帧间通信。我们的方法大大超过了我们实验中的最近的方法。

**[Paper URL](https://proceedings.mlr.press/v229/li23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/li23c/li23c.pdf)** 

# DORT: Modeling Dynamic Objects in Recurrent for Multi-Camera 3D Object Detection and Tracking
**题目:** DORT:多摄像头3D对象检测和跟踪中动态对象的建模

**作者:** Qing LIAN, Tai Wang, Dahua Lin, Jiangmiao Pang

**Abstract:** Recent multi-camera 3D object detectors usually leverage temporal information to construct multi-view stereo that alleviates the ill-posed depth estimation. However, they typically assume all the objects are static and directly aggregate features across frames. This work begins with a theoretical and empirical analysis to reveal that ignoring the motion of moving objects can result in serious localization bias. Therefore, we propose to model Dynamic Objects in RecurrenT (DORT) to tackle this problem. In contrast to previous global BirdEye-View (BEV) methods, DORT extracts object-wise local volumes for motion estimation that also alleviates the heavy computational burden. By iteratively refining the estimated object motion and location, the preceding features can be precisely aggregated to the current frame to mitigate the aforementioned adverse effects. The simple framework has two significant appealing properties. It is flexible and practical that can be plugged into most camera-based 3D object detectors. As there are predictions of object motion in the loop, it can easily track objects across frames according to their nearest center distances. Without bells and whistles, DORT outperforms all the previous methods on the nuScenes detection and tracking benchmarks with $62.8%$ NDS and $57.6%$ AMOTA, respectively. The source code will be available at https://github.com/OpenRobotLab/DORT.

**摘要:** 近来多摄像头3D对象检测器通常利用时间信息来构造多视角立体,从而减轻低位深度估计。然而,它们通常认为所有对象都是静态的,直接聚集在帧间的特征。这项工作由理论和经验分析开始,发现忽略移动对象的运动可能导致严重的定位偏差。因此,我们建议在Recurrent(DORT)中模拟动态对象来解决这个问题。它具有灵活性和实用性,可以在大多数基于摄像机的3D对象检测器中插入。由于循环中有对象运动的预报,它可以根据它们最近的中心距离轻易地跟踪对象。

**[Paper URL](https://proceedings.mlr.press/v229/lian23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lian23a/lian23a.pdf)** 

# Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition
**题目:** 提升和蒸馏:语言指导的机器人技能获取

**作者:** Huy Ha, Pete Florence, Shuran Song

**Abstract:** We present a framework for robot skill acquisition, which 1) efficiently scale up data generation of language-labelled robot data and 2) effectively distills this data down into a robust multi-task language-conditioned visuo-motor policy. For (1), we use a large language model (LLM) to guide high-level planning, and sampling-based robot planners (e.g. motion or grasp samplers) for generating diverse and rich manipulation trajectories. To robustify this data-collection process, the LLM also infers a code-snippet for the success condition of each task, simultaneously enabling the data-collection process to detect failure and retry as well as the automatic labeling of trajectories with success/failure. For (2), we extend the diffusion policy single-task behavior-cloning approach to multi-task settings with language conditioning. Finally, we propose a new multi-task benchmark with 18 tasks across five domains to test long-horizon behavior, common-sense reasoning, tool-use, and intuitive physics. We find that our distilled policy successfully learned the robust retrying behavior in its data collection procedure, while improving absolute success rates by $33.2%$ on average across five domains. Code, data, and additional qualitative results are available on https://www.cs.columbia.edu/ huy/scalingup/.

**摘要:** 我们提出了一种机器人技能获取的框架,该框架 1) 有效提高语言标记的机器人数据的数据生成量, 2) 有效将这些数据转化为具有鲁棒的多任务语言条件的视觉运动政策。 对于 (1), 我们使用大型语言模型 (LLM) 指导高层次规划,以及基于样本的机器人规划者 (例如运动或抓取样本器) 来生成多样和丰富的操纵轨迹。最后,我们提出了一项新的多任务基准,在5个领域内完成18个任务,以测试长期地平线行为、常识推理、工具使用和直观物理学。我们发现,我们所提取的政策在数据收集过程中成功地学习了 robust retrying behavior,同时在5个领域中平均提高了绝对成功率33.2%$。

**[Paper URL](https://proceedings.mlr.press/v229/ha23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ha23a/ha23a.pdf)** 

# Marginalized Importance Sampling for Off-Environment Policy Evaluation
**题目:** 非环境政策评价的边缘化重要度样本

**作者:** Pulkit Katdare, Nan Jiang, Katherine Rose Driggs-Campbell

**Abstract:** Reinforcement Learning (RL) methods are typically sample-inefficient, making it challenging to train and deploy RL-policies in real world robots. Even a robust policy trained in simulation requires a real-world deployment to assess their performance. This paper proposes a new approach to evaluate the real-world performance of agent policies prior to deploying them in the real world. Our approach incorporates a simulator along with real-world offline data to evaluate the performance of any policy using the framework of Marginalized Importance Sampling (MIS). Existing MIS methods face two challenges: (1) large density ratios that deviate from a reasonable range and (2) indirect supervision, where the ratio needs to be inferred indirectly, thus exacerbating estimation error. Our approach addresses these challenges by introducing the target policy’s occupancy in the simulator as an intermediate variable and learning the density ratio as the product of two terms that can be learned separately. The first term is learned with direct supervision and the second term has a small magnitude, thus making it computationally efficient. We analyze the sample complexity as well as error propagation of our two step-procedure. Furthermore, we empirically evaluate our approach on Sim2Sim environments such as Cartpole, Reacher, and Half-Cheetah. Our results show that our method generalizes well across a variety of Sim2Sim gap, target policies and offline data collection policies. We also demonstrate the performance of our algorithm on a Sim2Real task of validating the performance of a 7 DoF robotic arm using offline data along with the Gazebo simulator.

**摘要:** 强化学习(RL)方法通常不具有实例效率,使得在实时机器人中训练和部署RL政策具有挑战性。即使是在仿真中训练的强有力政策也需要实时部署来评估其性能。本论文提出了一种新的方法来评估代理政策在实时部署之前的实时性能。我们的方法包括一个模拟器和实时的非实时数据来评估任何政策的性能,使用边界重要性样本(MIS)框架。现有MIS方法面临两大挑战:(一)从合理的范围内偏离的大型密度比和(二)间接监督,其中比需要间接推导,从而加剧估计误差。我们通过引入目标策略在模拟器中作为中间变量和学习密度比作为两个术语的产物来解决这些挑战,第一个术语由直接监督学习,第二个术语具有小规模,从而使它具有计算效率。我们分析了样本的复杂性以及我们两个步骤程序的误差传播。此外,我们以实证的方式评估了我们对Sim2Sim环境的处理方法,例如卡特波尔、雷切尔和半豹。我们的结果表明,我们的方法在各种Sim2Sim差距、目标政策和非线性数据收集政策中具有普遍性。

**[Paper URL](https://proceedings.mlr.press/v229/katdare23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/katdare23a/katdare23a.pdf)** 

# Policy Stitching: Learning Transferable Robot Policies
**题目:** 政策刺激:学习可转让机器人政策

**作者:** Pingcheng Jian, Easop Lee, Zachary Bell, Michael M. Zavlanos, Boyuan Chen

**Abstract:** Training robots with reinforcement learning (RL) typically involves heavy interactions with the environment, and the acquired skills are often sensitive to changes in task environments and robot kinematics. Transfer RL aims to leverage previous knowledge to accelerate learning of new tasks or new body configurations. However, existing methods struggle to generalize to novel robot-task combinations and scale to realistic tasks due to complex architecture design or strong regularization that limits the capacity of the learned policy. We propose Policy Stitching, a novel framework that facilitates robot transfer learning for novel combinations of robots and tasks. Our key idea is to apply modular policy design and align the latent representations between the modular interfaces. Our method allows direct stitching of the robot and task modules trained separately to form a new policy for fast adaptation. Our simulated and real-world experiments on various 3D manipulation tasks demonstrate the superior zero-shot and few-shot transfer learning performances of our method.

**摘要:** 强化学习的训练机器人通常涉及与环境的重相互作用,所获得的技能往往对任务环境和机器人动力学的变化有敏感性。 transfer RL的目标是利用以前的知识加速学习新的任务或新的身体配置。然而,现有的方法难以推广到新的机器人任务组合和 scale to realistic tasks,因为复杂的架构设计或强的规范化限制了学习政策的能力。我们提出了 Policy Stitching,一种新框架,可以促进机器人和任务的新组合的机器人转移学习。我们的关键思想是应用模块化政策设计并调整模块化界面之间的潜在表现。我们对各种3D操作任务的模拟和现实实验证明了该方法的优越的零射和少射传输学习性能。

**[Paper URL](https://proceedings.mlr.press/v229/jian23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/jian23a/jian23a.pdf)** 

# Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation
**题目:** 序列敏捷:长期操纵的链路敏捷政策

**作者:** Yuanpei Chen, Chen Wang, Li Fei-Fei, Karen Liu

**Abstract:** Many real-world manipulation tasks consist of a series of subtasks that are significantly different from one another. Such long-horizon, complex tasks highlight the potential of dexterous hands, which possess adaptability and versatility, capable of seamlessly transitioning between different modes of functionality without the need for re-grasping or external tools. However, the challenges arise due to the high-dimensional action space of dexterous hand and complex compositional dynamics of the long-horizon tasks. We present Sequential Dexterity, a general system based on reinforcement learning (RL) that chains multiple dexterous policies for achieving long-horizon task goals. The core of the system is a transition feasibility function that progressively finetunes the sub-policies for enhancing chaining success rate, while also enables autonomous policy-switching for recovery from failures and bypassing redundant stages. Despite being trained only in simulation with a few task objects, our system demonstrates generalization capability to novel object shapes and is able to zero-shot transfer to a real-world robot equipped with a dexterous hand. Code and videos are available at https://sequential-dexterity.github.io.

**摘要:** 许多实际操作任务由一系列不同于彼此的次级任务组成,这些长期、复杂的任务突出了敏捷手的潜力,具有适应性和多变性,能够无缝地在不同功能模式之间进行转换,不需要重新剪接或外部工具。然而,由于敏捷手的高维动作空间和复杂复合动力学的长期任务所带来的挑战,我们提出了一种基于增强学习(RL)的通用系统Sequential Dexterity,它为实现长期任务目标 chains multiple dexterous policies。尽管只有少数任务对象进行仿真,但我们的系统展示了对新对象形状的一般化能力,并且能够将零射击转移到具有敏捷的手的现实世界中的机器人。代码和视频可访问 https://sequential-dexterity.github.io。

**[Paper URL](https://proceedings.mlr.press/v229/chen23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23e/chen23e.pdf)** 

# Deception Game: Closing the Safety-Learning Loop in Interactive Robot Autonomy
**题目:** 欺骗游戏:在交互机器人自主中关闭安全学习循环

**作者:** Haimin Hu, Zixu Zhang, Kensuke Nakamura, Andrea Bajcsy, Jaime Fernández Fisac

**Abstract:** An outstanding challenge for the widespread deployment of robotic systems like autonomous vehicles is ensuring safe interaction with humans without sacrificing performance. Existing safety methods often neglect the robot’s ability to learn and adapt at runtime, leading to overly conservative behavior. This paper proposes a new closed-loop paradigm for synthesizing safe control policies that explicitly account for the robot’s evolving uncertainty and its ability to quickly respond to future scenarios as they arise, by jointly considering the physical dynamics and the robot’s learning algorithm. We leverage adversarial reinforcement learning for tractable safety analysis under high-dimensional learning dynamics and demonstrate our framework’s ability to work with both Bayesian belief propagation and implicit learning through large pre-trained neural trajectory predictors.

**摘要:** 当前的安全方法往往忽略了机器人在运行时学习和适应的能力,导致过度保守的行为。本文提出了一种新的闭环模式来综合安全控制政策,明确说明机器人不断演变的不确定性及其能够在出现时迅速响应未来场景的能力,通过共同考虑机器人的物理动力学和学习算法。我们利用敌对强化学习,在高维学习动力学下进行可处理的安全分析,并展示了我们的框架能够通过大型预训练神经轨迹预测器与贝叶斯信仰传播和隐形学习相结合的能力。

**[Paper URL](https://proceedings.mlr.press/v229/hu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/hu23b/hu23b.pdf)** 

# Improving Behavioural Cloning with Positive Unlabeled Learning
**题目:** 积极无标签的学习改善行为克隆

**作者:** Qiang Wang, Robert McCarthy, David Cordova Bulens, Kevin McGuinness, Noel E. O’Connor, Francisco Roldan Sanchez, Nico Gürtler, Felix Widmaier, Stephen J. Redmond

**Abstract:** Learning control policies offline from pre-recorded datasets is a promising avenue for solving challenging real-world problems. However, available datasets are typically of mixed quality, with a limited number of the trajectories that we would consider as positive examples; i.e., high-quality demonstrations. Therefore, we propose a novel iterative learning algorithm for identifying expert trajectories in unlabeled mixed-quality robotics datasets given a minimal set of positive examples, surpassing existing algorithms in terms of accuracy. We show that applying behavioral cloning to the resulting filtered dataset outperforms several competitive offline reinforcement learning and imitation learning baselines. We perform experiments on a range of simulated locomotion tasks and on two challenging manipulation tasks on a real robotic system; in these experiments, our method showcases state-of-the-art performance. Our website: https://sites.google.com/view/offline-policy-learning-pubc.

**摘要:** 从预记录的数据集中获取的在线学习控制策略是解决挑战现实问题的一个有前途的途径。然而,现有的数据集通常具有混合质量,且有有限的轨迹,我们将将其视为正例;即高质量的演示。因此,我们提出了一种新的迭代学习算法,以识别未标记混合质量机器人数据集中的专家轨迹,给出了最小数量的正例,超越现有算法的准确性。

**[Paper URL](https://proceedings.mlr.press/v229/wang23f.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23f/wang23f.pdf)** 

# $α$-MDF: An Attention-based Multimodal Differentiable Filter for Robot State Estimation
**题目:** $α$-MDF:基于注意力的机器人状态估计多模微分滤波器

**作者:** Xiao Liu, Yifan Zhou, Shuhei Ikemoto, Heni Ben Amor

**Abstract:** Differentiable Filters are recursive Bayesian estimators that derive the state transition and measurement models from data alone. Their data-driven nature eschews the need for explicit analytical models, while remaining algorithmic components of the filtering process intact. As a result, the gain mechanism – a critical component of the filtering process – remains non-differentiable and cannot be adjusted to the specific nature of the task or context. In this paper, we propose an attention-based Multimodal Differentiable Filter ($\alpha$-MDF) which utilizes modern attention mechanisms to learn multimodal latent representations. Unlike previous differentiable filter frameworks, $\alpha$-MDF substitutes the traditional gain, e.g., the Kalman gain, with a neural attention mechanism. The approach generates specialized, context-dependent gains that can effectively combine multiple input modalities and observed variables. We validate $\alpha$-MDF on a diverse set of robot state estimation tasks in real world and simulation. Our results show $\alpha$-MDF achieves significant reductions in state estimation errors, demonstrating nearly 4-fold improvements compared to state-of-the-art sensor fusion strategies for rigid body robots. Additionally, the $\alpha$-MDF consistently outperforms differentiable filter baselines by up to $45%$ in soft robotics tasks. The project is available at alpha-mdf.github.io and the codebase is at github.com/ir-lab/alpha-MDF

**摘要:** 差分滤波器是一种递归的贝叶斯估计器,仅从数据中导出状态转换和测量模型。其数据驱动的性质避免了明确的分析模型的必要性,而滤波过程的剩余的算法组成部分则保持不变。因此,收益机制 — — 滤波过程的一个关键组成部分 — — 仍然是非差分的,无法适应任务或上下文的具体性质。我们对不同类型的机器人状态估计任务在现实世界中和仿真中验证了$alpha$-MDF。我们的结果表明,$alpha$-MDF在状态估计误差方面取得了显著的降低,与刚体机器人最先进的传感器融合策略相比,几乎有4倍的改进。此外,$alpha$-MDF在软机器人任务中持续超过可微分滤波基线,达到45%。该项目在alpha-mdf.github.io和 github.com/ir-lab/alpha-MDF

**[Paper URL](https://proceedings.mlr.press/v229/liu23h.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23h/liu23h.pdf)** 

# Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control
**题目:** 教学目标表述如下:半监督语言接口控制

**作者:** Vivek Myers, Andre Wang He, Kuan Fang, Homer Rich Walke, Philippe Hansen-Estruch, Ching-An Cheng, Mihai Jalobeanu, Andrey Kolobov, Anca Dragan, Sergey Levine

**Abstract:** Our goal is for robots to follow natural language instructions like “put the towel next to the microwave.” But getting large amounts of labeled data, i.e. data that contains demonstrations of tasks labeled with the language instruction, is prohibitive. In contrast, obtaining policies that respond to image goals is much easier, because any autonomous trial or demonstration can be labeled in hindsight with its final state as the goal. In this work, we contribute a method that taps into joint image- and goal- conditioned policies with language using only a small amount of language data. Prior work has made progress on this using vision-language models or by jointly training language-goal-conditioned policies, but so far neither method has scaled effectively to real-world robot tasks without significant human annotation. Our method achieves robust performance in the real world by learning an embedding from the labeled data that aligns language not to the goal image, but rather to the desired change between the start and goal images that the instruction corresponds to. We then train a policy on this embedding: the policy benefits from all the unlabeled data, but the aligned embedding provides an *interface* for language to steer the policy. We show instruction following across a variety of manipulation tasks in different scenes, with generalization to language instructions outside of the labeled data.

**摘要:** 我们的目标是让机器人遵循自然语言指令,例如“把毛巾放在微波旁边”。但获取大量标记数据,即包含与语言指令标记的任务的演示数据,是禁忌的。相反,获取响应图像目标的政策是更容易的,因为任何自主的试验或演示都可以以其最终状态为目标标记后视。我们的方法通过学习标签数据中的嵌入式来实现在现实世界中的鲁棒性能,该嵌入式使语言不会与目标图像相匹配,而是与指令与目标图像之间的期望变化相匹配。然后,我们训练了此嵌入式的政策:该政策从所有未标签的数据中获益,但匹配的嵌入式为语言提供指导政策的 * 接口*。

**[Paper URL](https://proceedings.mlr.press/v229/myers23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/myers23a/myers23a.pdf)** 

# Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions
**题目:** Q-变换器:通过自进式Q-函数进行可扩展的在线增强学习

**作者:** Yevgen Chebotar, Quan Vuong, Karol Hausman, Fei Xia, Yao Lu, Alex Irpan, Aviral Kumar, Tianhe Yu, Alexander Herzog, Karl Pertsch, Keerthana Gopalakrishnan, Julian Ibarz, Ofir Nachum, Sumedh Anand Sontakke, Grecia Salazar, Huong T. Tran, Jodilyn Peralta, Clayton Tan, Deeksha Manjunath, Jaspiar Singh, Brianna Zitkovich, Tomas Jackson, Kanishka Rao, Chelsea Finn, Sergey Levine

**Abstract:** In this work, we present a scalable reinforcement learning method for training multi-task policies from large offline datasets that can leverage both human demonstrations and autonomously collected data. Our method uses a Transformer to provide a scalable representation for Q-functions trained via offline temporal difference backups. We therefore refer to the method as Q-Transformer. By discretizing each action dimension and representing the Q-value of each action dimension as separate tokens, we can apply effective high-capacity sequence modeling techniques for Q-learning. We present several design decisions that enable good performance with offline RL training, and show that Q-Transformer outperforms prior offline RL algorithms and imitation learning techniques on a large diverse real-world robotic manipulation task suite.

**摘要:** 本文提出了一种可扩展增强学习方法,用于从大型非线性数据集中培训多任务政策,可利用人类的演示和自主收集的数据。我们的方法使用Transformer来提供通过非线性时间差备份训练的Q-函数的可扩展表示。因此,我们把该方法称为Q-Transformer。通过分离每个行动维度,并代表每个行动维度的Q-值作为单独的符号,我们可以应用有效的高容量序列建模技术来Q-学习。

**[Paper URL](https://proceedings.mlr.press/v229/chebotar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chebotar23a/chebotar23a.pdf)** 

# Preference learning for guiding the tree search in continuous POMDPs
**题目:** 在连续的POMDP中指导树搜索的优先学习

**作者:** Jiyong Ahn, Sanghyeon Son, Dongryung Lee, Jisu Han, Dongwon Son, Beomjoon Kim

**Abstract:** A robot operating in a partially observable environment must perform sensing actions to achieve a goal, such as clearing the objects in front of a shelf to better localize a target object at the back, and estimate its shape for grasping. A POMDP is a principled framework for enabling robots to perform such information-gathering actions. Unfortunately, while robot manipulation domains involve high-dimensional and continuous observation and action spaces, most POMDP solvers are limited to discrete spaces. Recently, POMCPOW has been proposed for continuous POMDPs, which handles continuity using sampling and progressive widening. However, for robot manipulation problems involving camera observations and multiple objects, POMCPOW is too slow to be practical. We take inspiration from the recent work in learning to guide task and motion planning to propose a framework that learns to guide POMCPOW from past planning experience. Our method uses preference learning that utilizes both success and failure trajectories, where the preference label is given by the results of the tree search. We demonstrate the efficacy of our framework in several continuous partially observable robotics domains, including real-world manipulation, where our framework explicitly reasons about the uncertainty in off-the-shelf segmentation and pose estimation algorithms.

**摘要:** 在部分可观测环境下运行的机器人必须执行感知动作,以达到目标,例如在架前清除目标对象以更好地定位目标对象后部,并估计其形状以便抓住。POMDP是允许机器人执行这些信息收集行动的原理框架。不幸的是,虽然机器人操纵领域涉及高维连续观察和行动空间,但大多数POMDP求解者只限于离散空间。最近,POMCPOW已被提议用于连续的POMDP,它使用采样和渐进的扩充来处理连续性。然而,对于涉及摄像机观察和多个对象的机器人操纵问题,POMCPOW太慢,无法实际应用。采用成功与失败的优先学习方法,以树搜索结果为优先标签。我们证明了我们的框架在几个连续部分可观察的机器人领域,包括实世界操作,其中我们的框架明确地解释了非市场分割中的不确定性,并提出估计算法。

**[Paper URL](https://proceedings.mlr.press/v229/ahn23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ahn23a/ahn23a.pdf)** 

# Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation
**题目:** Act3D:多任务机器人操纵的3D特征场变换器

**作者:** Theophile Gervet, Zhou Xian, Nikolaos Gkanatsios, Katerina Fragkiadaki

**Abstract:** 3D perceptual representations are well suited for robot manipulation as they easily encode occlusions and simplify spatial reasoning. Many manipulation tasks require high spatial precision in end-effector pose prediction, which typically demands high-resolution 3D feature grids that are computationally expensive to process. As a result, most manipulation policies operate directly in 2D, foregoing 3D inductive biases. In this paper, we introduce Act3D, a manipulation policy transformer that represents the robot’s workspace using a 3D feature field with adaptive resolutions dependent on the task at hand. The model lifts 2D pre-trained features to 3D using sensed depth, and attends to them to compute features for sampled 3D points. It samples 3D point grids in a coarse to fine manner, featurizes them using relative-position attention, and selects where to focus the next round of point sampling. In this way, it efficiently computes 3D action maps of high spatial resolution. Act3D sets a new state-of-the-art in RLBench, an established manipulation benchmark, where it achieves $10%$ absolute improvement over the previous SOTA 2D multi-view policy on 74 RLBench tasks and $22%$ absolute improvement with 3x less compute over the previous SOTA 3D policy. We quantify the importance of relative spatial attention, large-scale vision-language pre-trained 2D backbones, and weight tying across coarse-to-fine attentions in ablative experiments.

**摘要:** 3D感知表示非常适合机器人操纵,因为它容易编码隐蔽和简化空间推理。许多操纵任务需要高空间精确的最终效果者姿态预测,这通常要求处理高分辨率的3D特征网格。因此,大多数操纵政策直接在2D操作,避免了3D诱导偏见。本论文介绍了Act3D,一种基于任务的适应性分辨率的3D特征场的机器人工作空间的操纵政策变换器。Act3D为RLBench提供了一种新的最先进的操作基准,它在74个RLBench任务上比以前的SOTA 2D多视图政策取得10%美元的绝对改善,并且比以前的SOTA 3D政策减少3倍的计算量达到22%美元的绝对改善。

**[Paper URL](https://proceedings.mlr.press/v229/gervet23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gervet23a/gervet23a.pdf)** 

# Simultaneous Learning of Contact and Continuous Dynamics
**题目:** 同时学习接触与连续动力学

**作者:** Bibit Bianchini, Mathew Halm, Michael Posa

**Abstract:** Robotic manipulation can greatly benefit from the data efficiency, robustness, and predictability of model-based methods if robots can quickly generate models of novel objects they encounter. This is especially difficult when effects like complex joint friction lack clear first-principles models and are usually ignored by physics simulators. Further, numerically-stiff contact dynamics can make common model-building approaches struggle. We propose a method to simultaneously learn contact and continuous dynamics of a novel, possibly multi-link object by observing its motion through contact-rich trajectories. We formulate a system identification process with a loss that infers unmeasured contact forces, penalizing their violation of physical constraints and laws of motion given current model parameters. Our loss is unlike prediction-based losses used in differentiable simulation. Using a new dataset of real articulated object trajectories and an existing cube toss dataset, our method outperforms differentiable simulation and end-to-end alternatives with more data efficiency. See our project page for code, datasets, and media: https://sites.google.com/view/continuous-contact-nets/home

**摘要:** 如果机器人能够快速生成新对象模型,那么机器人的计算效率、鲁棒性和可预测性将大大提高,特别是当复杂关节摩擦等效应缺乏明确的初始原理模型,通常被物理模拟器忽略时,这种问题尤为困难。此外,数值刚度的接触动力学可以使共同的模型建构方法陷入困境。我们提出了一种方法,通过接触丰富轨迹观察新对象的接触和连续动力学,同时学习其运动,从而推导非测量的接触力,惩罚它们违反现有模型参数的物理约束和运动规律。采用一种新的实物线性轨迹数据集和一个现有的立方体投掷数据集,我们的方法比可微分模拟和端到端的替代方案更高效。请参阅我们的代码、数据集和媒体项目网页: https://sites.google.com/view/continuous-contact-nets/home

**[Paper URL](https://proceedings.mlr.press/v229/bianchini23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/bianchini23a/bianchini23a.pdf)** 

