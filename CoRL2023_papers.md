# Expansive Latent Planning for Sparse Reward Offline Reinforcement Learning
**题目:** 稀疏奖励离线强化学习的扩展潜在规划

**作者:** Robert Gieselmann, Florian T. Pokorny

**Abstract:** Sampling-based motion planning algorithms excel at searching global solution paths in geometrically complex settings. However, classical approaches, such as RRT, are difficult to scale beyond low-dimensional search spaces and rely on privileged knowledge e.g. about collision detection and underlying state distances. In this work, we take a step towards the integration of sampling-based planning into the reinforcement learning framework to solve sparse-reward control tasks from high-dimensional inputs. Our method, called VELAP, determines sequences of waypoints through sampling-based exploration in a learned state embedding. Unlike other sampling-based techniques, we iteratively expand a tree-based memory of visited latent areas, which is leveraged to explore a larger portion of the latent space for a given number of search iterations. We demonstrate state-of-the-art results in learning control from offline data in the context of vision-based manipulation under sparse reward feedback. Our method extends the set of available planning tools in model-based reinforcement learning by adding a latent planner that searches globally for feasible paths instead of being bound to a fixed prediction horizon.

**摘要:** 基于采样的运动规划算法擅长在几何复杂的环境中搜索全局解路径。然而，经典的方法，如RRT，很难扩展到低维搜索空间之外，并依赖于特权知识，例如关于碰撞检测和潜在状态距离的知识。在这项工作中，我们朝着将基于抽样的规划集成到强化学习框架中来解决来自高维输入的稀疏奖励控制任务迈出了一步。我们的方法，称为VELAP，通过学习状态嵌入中基于采样的探索来确定路点序列。与其他基于采样的技术不同，我们迭代地扩展访问潜在区域的基于树的记忆，该记忆被用于在给定的搜索迭代次数中探索更大部分的潜在空间。我们展示了在稀疏奖励反馈下基于视觉的操作环境下从离线数据学习控制的最新结果。我们的方法扩展了基于模型的强化学习中可用的规划工具集，增加了一个全局搜索可行路径的潜在规划器，而不是局限于固定的预测水平。

**[Paper URL](https://proceedings.mlr.press/v229/gieselmann23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gieselmann23a/gieselmann23a.pdf)** 

# Expansive Latent Planning for Sparse Reward Offline Reinforcement Learning
**题目:** 稀疏奖励离线强化学习的扩展潜在规划

**作者:** Robert Gieselmann, Florian T. Pokorny

**Abstract:** Sampling-based motion planning algorithms excel at searching global solution paths in geometrically complex settings. However, classical approaches, such as RRT, are difficult to scale beyond low-dimensional search spaces and rely on privileged knowledge e.g. about collision detection and underlying state distances. In this work, we take a step towards the integration of sampling-based planning into the reinforcement learning framework to solve sparse-reward control tasks from high-dimensional inputs. Our method, called VELAP, determines sequences of waypoints through sampling-based exploration in a learned state embedding. Unlike other sampling-based techniques, we iteratively expand a tree-based memory of visited latent areas, which is leveraged to explore a larger portion of the latent space for a given number of search iterations. We demonstrate state-of-the-art results in learning control from offline data in the context of vision-based manipulation under sparse reward feedback. Our method extends the set of available planning tools in model-based reinforcement learning by adding a latent planner that searches globally for feasible paths instead of being bound to a fixed prediction horizon.

**摘要:** 基于采样的运动规划算法擅长在几何复杂的环境中搜索全局解路径。然而，经典的方法，如RRT，很难扩展到低维搜索空间之外，并依赖于特权知识，例如关于碰撞检测和潜在状态距离的知识。在这项工作中，我们朝着将基于抽样的规划集成到强化学习框架中来解决来自高维输入的稀疏奖励控制任务迈出了一步。我们的方法，称为VELAP，通过学习状态嵌入中基于采样的探索来确定路点序列。与其他基于采样的技术不同，我们迭代地扩展访问潜在区域的基于树的记忆，该记忆被用于在给定的搜索迭代次数中探索更大部分的潜在空间。我们展示了在稀疏奖励反馈下基于视觉的操作环境下从离线数据学习控制的最新结果。我们的方法扩展了基于模型的强化学习中可用的规划工具集，增加了一个全局搜索可行路径的潜在规划器，而不是局限于固定的预测水平。

**[Paper URL](https://proceedings.mlr.press/v229/gieselmann23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gieselmann23a/gieselmann23a.pdf)** 

# SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning
**题目:** SayPlan：使用3D场景图为大型语言模型基础，以实现可扩展机器人任务规划

**作者:** Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, Niko Suenderhauf

**Abstract:** Large language models (LLMs) have demonstrated impressive results in developing generalist planning agents for diverse tasks. However, grounding these plans in expansive, multi-floor, and multi-room environments presents a significant challenge for robotics. We introduce SayPlan, a scalable approach to LLM-based, large-scale task planning for robotics using 3D scene graph (3DSG) representations. To ensure the scalability of our approach, we: (1) exploit the hierarchical nature of 3DSGs to allow LLMs to conduct a "semantic search" for task-relevant subgraphs from a smaller, collapsed representation of the full graph; (2) reduce the planning horizon for the LLM by integrating a classical path planner and (3) introduce an "iterative replanning" pipeline that refines the initial plan using feedback from a scene graph simulator, correcting infeasible actions and avoiding planning failures. We evaluate our approach on two large-scale environments spanning up to 3 floors and 36 rooms with 140 assets and objects and show that our approach is capable of grounding large-scale, long-horizon task plans from abstract, and natural language instruction for a mobile manipulator robot to execute. We provide real robot video demonstrations on our project page https://sayplan.github.io.

**摘要:** 大型语言模型(LLM)在为不同任务开发通用计划代理方面取得了令人印象深刻的结果。然而，在宽敞、多层和多房间的环境中将这些计划接地对机器人来说是一个巨大的挑战。我们介绍了SayPlan，这是一种可扩展的基于LLM的大规模任务规划方法，使用3D场景图(3DSG)表示。为了确保该方法的可扩展性，我们：(1)利用3DSG的层次化性质，允许LLM从完整图的较小的折叠表示中进行与任务相关的子图的“语义搜索”；(2)通过集成经典的路径规划器来减少LLM的规划范围；(3)引入“迭代重新规划”流水线，该流水线使用场景图模拟器的反馈来优化初始规划，纠正不可行的动作并避免规划失败。我们在两个大型环境中对我们的方法进行了评估，该环境跨越3层楼和36个房间，包含140个资产和对象，并表明我们的方法能够从抽象和自然语言指令中获取大规模、长期的任务计划，供移动机械手机器人执行。我们在我们的项目页面https://sayplan.github.io.上提供真实的机器人视频演示

**[Paper URL](https://proceedings.mlr.press/v229/rana23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rana23a/rana23a.pdf)** 

# SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning
**题目:** SayPlan：使用3D场景图为大型语言模型基础，以实现可扩展机器人任务规划

**作者:** Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, Niko Suenderhauf

**Abstract:** Large language models (LLMs) have demonstrated impressive results in developing generalist planning agents for diverse tasks. However, grounding these plans in expansive, multi-floor, and multi-room environments presents a significant challenge for robotics. We introduce SayPlan, a scalable approach to LLM-based, large-scale task planning for robotics using 3D scene graph (3DSG) representations. To ensure the scalability of our approach, we: (1) exploit the hierarchical nature of 3DSGs to allow LLMs to conduct a "semantic search" for task-relevant subgraphs from a smaller, collapsed representation of the full graph; (2) reduce the planning horizon for the LLM by integrating a classical path planner and (3) introduce an "iterative replanning" pipeline that refines the initial plan using feedback from a scene graph simulator, correcting infeasible actions and avoiding planning failures. We evaluate our approach on two large-scale environments spanning up to 3 floors and 36 rooms with 140 assets and objects and show that our approach is capable of grounding large-scale, long-horizon task plans from abstract, and natural language instruction for a mobile manipulator robot to execute. We provide real robot video demonstrations on our project page https://sayplan.github.io.

**摘要:** 大型语言模型(LLM)在为不同任务开发通用计划代理方面取得了令人印象深刻的结果。然而，在宽敞、多层和多房间的环境中将这些计划接地对机器人来说是一个巨大的挑战。我们介绍了SayPlan，这是一种可扩展的基于LLM的大规模任务规划方法，使用3D场景图(3DSG)表示。为了确保该方法的可扩展性，我们：(1)利用3DSG的层次化性质，允许LLM从完整图的较小的折叠表示中进行与任务相关的子图的“语义搜索”；(2)通过集成经典的路径规划器来减少LLM的规划范围；(3)引入“迭代重新规划”流水线，该流水线使用场景图模拟器的反馈来优化初始规划，纠正不可行的动作并避免规划失败。我们在两个大型环境中对我们的方法进行了评估，该环境跨越3层楼和36个房间，包含140个资产和对象，并表明我们的方法能够从抽象和自然语言指令中获取大规模、长期的任务计划，供移动机械手机器人执行。我们在我们的项目页面https://sayplan.github.io.上提供真实的机器人视频演示

**[Paper URL](https://proceedings.mlr.press/v229/rana23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rana23a/rana23a.pdf)** 

# Robot Parkour Learning
**题目:** 机器人跑酷学习

**作者:** Ziwen Zhuang, Zipeng Fu, Jianren Wang, Christopher G. Atkeson, Sören Schwertfeger, Chelsea Finn, Hang Zhao

**Abstract:** Parkour is a grand challenge for legged locomotion that requires robots to overcome various obstacles rapidly in complex environments. Existing methods can generate either diverse but blind locomotion skills or vision-based but specialized skills by using reference animal data or complex rewards. However, autonomous parkour requires robots to learn generalizable skills that are both vision-based and diverse to perceive and react to various scenarios. In this work, we propose a system for learning a single end-to-end vision-based parkour policy of diverse parkour skills using a simple reward without any reference motion data. We develop a reinforcement learning method inspired by direct collocation to generate parkour skills, including climbing over high obstacles, leaping over large gaps, crawling beneath low barriers, squeezing through thin slits, and running. We distill these skills into a single vision-based parkour policy and transfer it to a quadrupedal robot using its egocentric depth camera. We demonstrate that our system can empower low-cost quadrupedal robots to autonomously select and execute appropriate parkour skills to traverse challenging environments in the real world. Project website: https://robot-parkour.github.io/

**摘要:** 跑酷是步行运动的一项重大挑战，需要机器人在复杂的环境中快速克服各种障碍。现有的方法可以通过使用参考动物数据或复杂的奖励来产生多样化但盲目的运动技能，或基于视觉但专门的技能。然而，自主跑酷要求机器人学习基于视觉的通用技能，并对各种场景做出感知和反应。在这项工作中，我们提出了一个系统，用于学习单一的端到端基于视觉的跑酷策略，使用简单的奖励来学习不同的跑酷技能，而不需要任何参考运动数据。我们开发了一种受直接搭配启发的强化学习方法，以生成跑酷技能，包括爬过高障碍、跳过大缺口、在低障碍下爬行、挤过细缝和跑步。我们将这些技能提炼成一个基于视觉的跑酷策略，并使用其以自我为中心的深度摄像头将其传输到四足机器人。我们证明，我们的系统可以使低成本的四足机器人自主选择和执行适当的跑酷技能，以穿越现实世界中具有挑战性的环境。项目网站：https://robot-parkour.github.io/

**[Paper URL](https://proceedings.mlr.press/v229/zhuang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhuang23a/zhuang23a.pdf)** 

# Task-Oriented Koopman-Based Control with Contrastive Encoder
**题目:** 使用对比编码器的面向任务Koopman控制

**作者:** Xubo Lyu, Hanyang Hu, Seth Siriya, Ye Pu, Mo Chen

**Abstract:** We present task-oriented Koopman-based control that utilizes end-to-end reinforcement learning and contrastive encoder to simultaneously learn the Koopman latent embedding, operator, and associated linear controller within an iterative loop. By prioritizing the task cost as the main objective for controller learning, we reduce the reliance of controller design on a well-identified model, which, for the first time to the best of our knowledge, extends Koopman control from low to high-dimensional, complex nonlinear systems, including pixel-based tasks and a real robot with lidar observations. Code and videos are available: https://sites.google.com/view/kpmlilatsupp/.

**摘要:** 我们提出了面向任务的基于Koopman的控制，它利用端到端强化学习和对比编码器在迭代循环中同时学习Koopman潜在嵌入、操作符和相关线性控制器。通过优先考虑任务成本作为控制器学习的主要目标，我们减少了控制器设计对良好识别模型的依赖，据我们所知，该模型首次将库普曼控制从低维扩展到高维复杂非线性系统，包括基于像素的任务和具有激光雷达观察的真实机器人。代码和视频可获取：https://sites.google.com/view/kpmlilatsupp/。

**[Paper URL](https://proceedings.mlr.press/v229/lyu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lyu23a/lyu23a.pdf)** 

# On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills
**题目:** 论库普曼运算理论在学习灵巧操作技能中的应用

**作者:** Yunhai Han, Mandy Xie, Ye Zhao, Harish Ravichandar

**Abstract:** Despite impressive dexterous manipulation capabilities enabled by learning-based approaches, we are yet to witness widespread adoption beyond well-resourced laboratories. This is likely due to practical limitations, such as significant computational burden, inscrutable learned behaviors, sensitivity to initialization, and the considerable technical expertise required for implementation. In this work, we investigate the utility of Koopman operator theory in alleviating these limitations. Koopman operators are simple yet powerful control-theoretic structures to represent complex nonlinear dynamics as linear systems in higher dimensions. Motivated by the fact that complex nonlinear dynamics underlie dexterous manipulation, we develop a Koopman operator-based imitation learning framework to learn the desired motions of both the robotic hand and the object simultaneously. We show that Koopman operators are surprisingly effective for dexterous manipulation and offer a number of unique benefits. Notably, policies can be learned analytically, drastically reducing computation burden and eliminating sensitivity to initialization and the need for painstaking hyperparameter optimization. Our experiments reveal that a Koopman operator-based approach can perform comparably to state-of-the-art imitation learning algorithms in terms of success rate and sample efficiency, while being an order of magnitude faster. Policy videos can be viewed at https://sites.google.com/view/kodex-corl.

**摘要:** 尽管基于学习的方法实现了令人印象深刻的灵活操作能力，但除了资源充足的实验室之外，我们还没有看到广泛采用。这可能是由于实际限制，例如巨大的计算负担、难以理解的学习行为、对初始化的敏感性以及实现所需的大量技术专业知识。在这项工作中，我们研究了库普曼算子理论在缓解这些限制方面的效用。库普曼算子是一种简单而强大的控制理论结构，可以将复杂的非线性动力学表示为更高维的线性系统。基于复杂的非线性动力学是灵巧操作的基础，我们开发了一种基于Koopman算子的模仿学习框架来同时学习机械手和物体的期望运动。我们证明了库普曼算子对于灵巧的操作是令人惊讶的有效的，并提供了许多独特的好处。值得注意的是，可以通过分析来学习策略，从而极大地减少了计算负担，消除了对初始化的敏感性以及对费力的超参数优化的需要。我们的实验表明，基于Koopman算子的方法在成功率和样本效率方面可以与最先进的模仿学习算法相媲美，而速度要快一个数量级。政策视频可在https://sites.google.com/view/kodex-corl.上查看

**[Paper URL](https://proceedings.mlr.press/v229/han23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/han23a/han23a.pdf)** 

# Rearrangement Planning for General Part Assembly
**题目:** 通用零件装配的重新排列规划

**作者:** Yulong Li, Andy Zeng, Shuran Song

**Abstract:** Most successes in autonomous robotic assembly have been restricted to single target or category. We propose to investigate general part assembly, the task of creating novel target assemblies with unseen part shapes. As a fundamental step to a general part assembly system, we tackle the task of determining the precise poses of the parts in the target assembly, which we term “rearrangement planning". We present General Part Assembly Transformer (GPAT), a transformer-based model architecture that accurately predicts part poses by inferring how each part shape corresponds to the target shape. Our experiments on both 3D CAD models and real-world scans demonstrate GPAT’s generalization abilities to novel and diverse target and part shapes.

**摘要:** 自主机器人组装的大多数成功仅限于单一目标或类别。我们建议研究一般零件装配，即创建具有不可见零件形状的新型目标装配的任务。作为通用零件装配系统的基本步骤，我们要解决确定目标装配中零件的精确姿态的任务，我们将其称为“重新排列规划”。我们介绍了通用零件装配Transformer（GMAT），这是一种基于变形器的模型架构，通过推断每个零件形状如何与目标形状相对应来准确预测零件姿态。我们对3D CAD模型和现实世界扫描的实验证明了GMAT对新颖且多样化的目标和零件形状的概括能力。

**[Paper URL](https://proceedings.mlr.press/v229/li23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/li23a/li23a.pdf)** 

# Language-Guided Traffic Simulation via Scene-Level Diffusion
**题目:** 基于场景级扩散的图像引导交通模拟

**作者:** Ziyuan Zhong, Davis Rempe, Yuxiao Chen, Boris Ivanovic, Yulong Cao, Danfei Xu, Marco Pavone, Baishakhi Ray

**Abstract:** Realistic and controllable traffic simulation is a core capability that is necessary to accelerate autonomous vehicle (AV) development. However, current approaches for controlling learning-based traffic models require significant domain expertise and are difficult for practitioners to use. To remedy this, we present CTG++, a scene-level conditional diffusion model that can be guided by language instructions. Developing this requires tackling two challenges: the need for a realistic and controllable traffic model backbone, and an effective method to interface with a traffic model using language. To address these challenges, we first propose a scene-level diffusion model equipped with a spatio-temporal transformer backbone, which generates realistic and controllable traffic. We then harness a large language model (LLM) to convert a user’s query into a loss function, guiding the diffusion model towards query-compliant generation. Through comprehensive evaluation, we demonstrate the effectiveness of our proposed method in generating realistic, query-compliant traffic simulations.

**摘要:** 逼真、可控的交通仿真是加速自动驾驶汽车发展的核心能力。然而，当前用于控制基于学习的流量模型的方法需要大量的领域专业知识，而且从业者很难使用。为了解决这一问题，我们提出了CTG++，这是一个可以由语言指令指导的场景级条件扩散模型。开发这一点需要解决两个挑战：需要一个现实的和可控的交通模型主干，以及使用语言与交通模型进行交互的有效方法。为了应对这些挑战，我们首先提出了一种场景级扩散模型，该模型配备了时空转换器骨干，可以生成逼真且可控的流量。然后，我们利用一个大型语言模型(LLM)将用户的查询转换为损失函数，将扩散模型引导到符合查询的生成。通过综合评估，我们证明了我们提出的方法在生成逼真的、符合查询的交通模拟方面的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/zhong23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhong23a/zhong23a.pdf)** 

# Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping
**题目:** 语言嵌入辐射场，用于零镜头面向任务的抓取

**作者:** Adam Rashid, Satvik Sharma, Chung Min Kim, Justin Kerr, Lawrence Yunliang Chen, Angjoo Kanazawa, Ken Goldberg

**Abstract:** Grasping objects by a specific subpart is often crucial for safety and for executing downstream tasks. We propose LERF-TOGO, Language Embedded Radiance Fields for Task-Oriented Grasping of Objects, which uses vision-language models zero-shot to output a grasp distribution over an object given a natural language query. To accomplish this, we first construct a LERF of the scene, which distills CLIP embeddings into a multi-scale 3D language field queryable with text. However, LERF has no sense of object boundaries, so its relevancy outputs often return incomplete activations over an object which are insufficient for grasping. LERF-TOGO mitigates this lack of spatial grouping by extracting a 3D object mask via DINO features and then conditionally querying LERF on this mask to obtain a semantic distribution over the object to rank grasps from an off-the-shelf grasp planner. We evaluate LERF-TOGO’s ability to grasp task-oriented object parts on 31 physical objects, and find it selects grasps on the correct part in $81%$ of trials and grasps successfully in $69%$. Code, data, appendix, and details are available at: lerftogo.github.io

**摘要:** 通过特定的子部件抓取对象通常对安全和执行下游任务至关重要。我们提出了LERF-TOGO，LERF-TOGO，用于面向任务的对象抓取的语言嵌入辐射场，它使用视觉-语言模型零镜头输出给定自然语言查询的对象上的抓取分布。为此，我们首先构建了场景的LERF，它将剪辑嵌入到可用文本查询的多尺度3D语言字段中。然而，LERF没有对象边界的感觉，因此它的相关性输出经常返回对对象的不完全激活，这不足以抓取。LERF-TOGO通过Dino特征提取3D对象掩码，然后在该掩码上有条件地查询LERF，以获得对象上的语义分布，以从现有的抓取规划器对抓取进行排名，从而缓解了这种缺乏空间分组的问题。我们评估了LERF-TOGO在31个物理对象上抓取面向任务的对象部分的能力，发现它在$81%的试验中选择了正确的部分抓取，并在$69%的试验中成功抓取。代码、数据、附录和详细信息请访问：lerftogo.gihub.io

**[Paper URL](https://proceedings.mlr.press/v229/rashid23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rashid23a/rashid23a.pdf)** 

# MimicPlay: Long-Horizon Imitation Learning by Watching Human Play
**题目:** MimicPlay：通过观看人类游戏进行长视野模仿学习

**作者:** Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, Anima Anandkumar

**Abstract:** Imitation learning from human demonstrations is a promising paradigm for teaching robots manipulation skills in the real world. However, learning complex long-horizon tasks often requires an unattainable amount of demonstrations. To reduce the high data requirement, we resort to human play data - video sequences of people freely interacting with the environment using their hands. Even with different morphologies, we hypothesize that human play data contain rich and salient information about physical interactions that can readily facilitate robot policy learning. Motivated by this, we introduce a hierarchical learning framework named MimicPlay that learns latent plans from human play data to guide low-level visuomotor control trained on a small number of teleoperated demonstrations. With systematic evaluations of 14 long-horizon manipulation tasks in the real world, we show that MimicPlay outperforms state-of-the-art imitation learning methods in task success rate, generalization ability, and robustness to disturbances. Code and videos are available at https://mimic-play.github.io.

**摘要:** 从人类演示中模仿学习是在现实世界中教授机器人操作技能的一个很有前途的范例。然而，学习复杂的长期任务往往需要大量的演示。为了降低对数据的高要求，我们求助于人类播放数据-人们使用他们的手与环境自由交互的视频序列。即使有不同的形态，我们假设人类的游戏数据包含关于物理交互的丰富和显著的信息，这些信息可以很容易地促进机器人的策略学习。受此启发，我们引入了一个名为MimicPlay的分层学习框架，该框架从人类游戏数据中学习潜在的计划，以指导在少量远程操作演示上训练的低级视觉运动控制。通过对现实世界中14个长时间操纵任务的系统评估，我们发现MimicPlay在任务成功率、泛化能力和对干扰的鲁棒性方面都优于最先进的模仿学习方法。代码和视频可在https://mimic-play.github.io.上查看

**[Paper URL](https://proceedings.mlr.press/v229/wang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23a/wang23a.pdf)** 

# Continual Vision-based Reinforcement Learning with Group Symmetries
**题目:** 具有群体对称性的基于连续视觉的强化学习

**作者:** Shiqi Liu, Mengdi Xu, Peide Huang, Xilun Zhang, Yongkang Liu, Kentaro Oguchi, Ding Zhao

**Abstract:** Continual reinforcement learning aims to sequentially learn a variety of tasks, retaining the ability to perform previously encountered tasks while simultaneously developing new policies for novel tasks. However, current continual RL approaches overlook the fact that certain tasks are identical under basic group operations like rotations or translations, especially with visual inputs. They may unnecessarily learn and maintain a new policy for each similar task, leading to poor sample efficiency and weak generalization capability. To address this, we introduce a unique Continual Vision-based Reinforcement Learning method that recognizes Group Symmetries, called COVERS, cultivating a policy for each group of equivalent tasks rather than an individual task. COVERS employs a proximal-policy-gradient-based (PPO-based) algorithm to train each policy, which contains an equivariant feature extractor and takes inputs with different modalities, including image observations and robot proprioceptive states. It also utilizes an unsupervised task grouping mechanism that relies on 1-Wasserstein distance on the extracted invariant features. We evaluate COVERS on a sequence of table-top manipulation tasks in simulation and on a real robot platform. Our results show that COVERS accurately assigns tasks to their respective groups and significantly outperforms baselines by generalizing to unseen but equivariant tasks in seen task groups. Demos are available on our project page: https://sites.google.com/view/rl-covers/.

**摘要:** 持续强化学习的目标是顺序地学习各种任务，保持执行以前遇到的任务的能力，同时为新任务开发新的策略。然而，当前连续的RL方法忽略了这样一个事实，即某些任务在旋转或平移等基本分组操作下是相同的，特别是在视觉输入下。他们可能不必要地为每个相似的任务学习和维护新的策略，导致样本效率低和泛化能力弱。为了解决这个问题，我们引入了一种独特的基于连续视觉的强化学习方法，该方法识别组对称性，称为Covers，为每组等价的任务而不是单个任务培养策略。Covers使用基于最近策略梯度(PPO-Based)的算法来训练每个策略，该算法包含一个等变特征提取器，并以不同的方式获取输入，包括图像观察和机器人本体感知状态。它还利用了一种无监督的任务分组机制，该机制依赖于提取的不变特征的1-Wasserstein距离。我们在模拟和真实的机器人平台上评估了一系列桌面操作任务的覆盖。我们的结果表明，COVERs能准确地将任务分配给各自的组，并且通过将其推广到SEW任务组中看不见但相同的任务，显著地优于基线。演示可在我们的项目页面上找到：https://sites.google.com/view/rl-covers/.

**[Paper URL](https://proceedings.mlr.press/v229/liu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23a/liu23a.pdf)** 

# HACMan: Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation
**题目:** HACMan：学习混合演员-评论家地图以实现6D非预设操纵

**作者:** Wenxuan Zhou, Bowen Jiang, Fan Yang, Chris Paxton, David Held

**Abstract:** Manipulating objects without grasping them is an essential component of human dexterity, referred to as non-prehensile manipulation. Non-prehensile manipulation may enable more complex interactions with the objects, but also presents challenges in reasoning about gripper-object interactions. In this work, we introduce Hybrid Actor-Critic Maps for Manipulation (HACMan), a reinforcement learning approach for 6D non-prehensile manipulation of objects using point cloud observations. HACMan proposes a temporally-abstracted and spatially-grounded object-centric action representation that consists of selecting a contact location from the object point cloud and a set of motion parameters describing how the robot will move after making contact. We modify an existing off-policy RL algorithm to learn in this hybrid discrete-continuous action representation. We evaluate HACMan on a 6D object pose alignment task in both simulation and in the real world. On the hardest version of our task, with randomized initial poses, randomized 6D goals, and diverse object categories, our policy demonstrates strong generalization to unseen object categories without a performance drop, achieving an $89%$ success rate on unseen objects in simulation and $50%$ success rate with zero-shot transfer in the real world. Compared to alternative action representations, HACMan achieves a success rate more than three times higher than the best baseline. With zero-shot sim2real transfer, our policy can successfully manipulate unseen objects in the real world for challenging non-planar goals, using dynamic and contact-rich non-prehensile skills. Videos can be found on the project website: https://hacman-2023.github.io.

**摘要:** 在不抓住物体的情况下操纵物体是人类灵巧性的重要组成部分，被称为非抓握操纵。非抓取操作可能会实现与对象的更复杂的交互，但也给关于抓手-对象交互的推理带来了挑战。在这项工作中，我们介绍了混合参与者-批评者操纵地图(HACMan)，一种强化学习方法，用于使用点云观测对物体进行6D非卷曲操纵。HACMan提出了一种时间抽象和空间基础的以对象为中心的动作表示，该表示包括从对象点云中选择接触位置和一组描述机器人在接触后如何移动的运动参数。我们对现有的非策略RL算法进行了修改，使其能够在这种离散-连续混合动作表示中学习。我们在仿真和真实世界中对HACMan进行了6D目标姿态对准任务的评估。在我们任务的最困难的版本上，具有随机的初始姿势、随机的6D目标和不同的对象类别，我们的策略在不降低性能的情况下对看不见的对象类别表现出很强的泛化能力，在模拟中对看不见的对象实现了$89%的成功率，在真实世界中实现了$50%的零镜头传输成功率。与其他操作表示相比，HACMan的成功率比最佳基准高出三倍以上。通过零射Sim2Real传输，我们的策略可以使用动态和接触丰富的非抓取技能，成功地操纵现实世界中看不见的物体，以挑战非平面目标。视频可在项目网站上找到：https://hacman-2023.github.io.

**[Paper URL](https://proceedings.mlr.press/v229/zhou23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhou23a/zhou23a.pdf)** 

# Hijacking Robot Teams Through Adversarial Communication
**题目:** 通过对抗性沟通劫持机器人团队

**作者:** Zixuan Wu, Sean Charles Ye, Byeolyi Han, Matthew Gombolay

**Abstract:** Communication is often necessary for robot teams to collaborate and complete a decentralized task. Multi-agent reinforcement learning (MARL) systems allow agents to learn how to collaborate and communicate to complete a task. These domains are ubiquitous and include safety-critical domains such as wildfire fighting, traffic control, or search and rescue missions. However, critical vulnerabilities may arise in communication systems as jamming the signals can interrupt the robot team. This work presents a framework for applying black-box adversarial attacks to learned MARL policies by manipulating only the communication signals between agents. Our system only requires observations of MARL policies after training is complete, as this is more realistic than attacking the training process. To this end, we imitate a learned policy of the targeted agents without direct interaction with the environment or ground truth rewards. Instead, we infer the rewards by only observing the behavior of the targeted agents. Our framework reduces reward by $201%$ compared to an equivalent baseline method and also shows favorable results when deployed in real swarm robots. Our novel attack methodology within MARL systems contributes to the field by enhancing our understanding on the reliability of multi-agent systems.

**摘要:** 通信对于机器人团队的协作和完成分散的任务通常是必要的。多智能体强化学习(MAIL)系统允许智能体学习如何协作和通信来完成任务。这些领域无处不在，包括野火扑救、交通控制或搜救任务等安全关键领域。然而，通信系统中可能会出现严重的漏洞，因为干扰信号可能会中断机器人团队。这项工作提出了一个框架，通过只操作代理之间的通信信号，将黑盒对抗攻击应用于学习的Marl策略。我们的系统只需要在培训完成后观察Marl政策，因为这比攻击培训过程更现实。为此，我们模仿目标代理人的学习策略，而不与环境或地面事实奖励直接互动。相反，我们只通过观察目标特工的行为来推断奖励。与同等的基线方法相比，我们的框架减少了201%美元的奖励，在实际的群体机器人中也显示了良好的效果。我们在MAIL系统中的新攻击方法通过增强我们对多代理系统可靠性的理解，为该领域做出了贡献。

**[Paper URL](https://proceedings.mlr.press/v229/wu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wu23a/wu23a.pdf)** 

# GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields
**题目:** GNFactor：具有可推广神经特征场的多任务真实机器人学习

**作者:** Yanjie Ze, Ge Yan, Yueh-Hua Wu, Annabella Macaluso, Yuying Ge, Jianglong Ye, Nicklas Hansen, Li Erran Li, Xiaolong Wang

**Abstract:** It is a long-standing problem in robotics to develop agents capable of executing diverse manipulation tasks from visual observations in unstructured real-world environments. To achieve this goal, the robot will need to have a comprehensive understanding of the 3D structure and semantics of the scene. In this work, we present GNFactor, a visual behavior cloning agent for multi-task robotic manipulation with Generalizable Neural feature Fields. GNFactor jointly optimizes a neural radiance field (NeRF) as a reconstruction module and a Perceiver Transformer as a decision-making module, leveraging a shared deep 3D voxel representation. To incorporate semantics in 3D, the reconstruction module incorporates a vision-language foundation model (e.g., Stable Diffusion) to distill rich semantic information into the deep 3D voxel. We evaluate GNFactor on 3 real-robot tasks and perform detailed ablations on 10 RLBench tasks with a limited number of demonstrations. We observe a substantial improvement of GNFactor over current state-of-the-art methods in seen and unseen tasks, demonstrating the strong generalization ability of GNFactor. Project website: https://yanjieze.com/GNFactor/

**摘要:** 开发能够在非结构化真实环境中根据视觉观察执行各种操作任务的代理是机器人学中的一个长期问题。为了实现这一目标，机器人需要对场景的3D结构和语义有一个全面的了解。在这项工作中，我们提出了GNFactor，一个用于多任务机器人操作的视觉行为克隆代理，它具有可泛化的神经特征场。GNFactor利用共享的深3D体素表示，联合优化神经辐射场(NERF)作为重建模块，并将感知器变压器作为决策模块。为了将语义结合到3D中，重建模块结合了视觉-语言基础模型(例如，稳定扩散)以将丰富的语义信息提取到深层3D体素中。我们在3个真实的机器人任务上评估了GNFactor，并通过有限数量的演示对10个RLBch任务进行了详细的消融。我们观察到GNFactor在可见和不可见任务上比目前最先进的方法有了很大的改进，表明了GNFactor强大的泛化能力。项目网站：https://yanjieze.com/GNFactor/

**[Paper URL](https://proceedings.mlr.press/v229/ze23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ze23a/ze23a.pdf)** 

# Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance
**题目:** 引导您自己的技能：学习使用大型语言模型指导解决新任务

**作者:** Jesse Zhang, Jiahui Zhang, Karl Pertsch, Ziyi Liu, Xiang Ren, Minsuk Chang, Shao-Hua Sun, Joseph J. Lim

**Abstract:** We propose BOSS, an approach that automatically learns to solve new long-horizon, complex, and meaningful tasks by growing a learned skill library with minimal supervision. Prior work in reinforcement learning require expert supervision, in the form of demonstrations or rich reward functions, to learn long-horizon tasks. Instead, our approach BOSS (BOotStrapping your own Skills) learns to accomplish new tasks by performing "skill bootstrapping," where an agent with a set of primitive skills interacts with the environment to practice new skills without receiving reward feedback for tasks outside of the initial skill set. This bootstrapping phase is guided by large language models (LLMs) that inform the agent of meaningful skills to chain together. Through this process, BOSS builds a wide range of complex and useful behaviors from a basic set of primitive skills. We demonstrate through experiments in realistic household environments that agents trained with our LLM-guided bootstrapping procedure outperform those trained with naive bootstrapping as well as prior unsupervised skill acquisition methods on zero-shot execution of unseen, long-horizon tasks in new environments. Website at clvrai.com/boss.

**摘要:** 我们提出BOSS，这是一种自动学习解决新的长期、复杂和有意义的任务的方法，通过在最少的监督下增长学习的技能库来实现。强化学习之前的工作需要专家的监督，以演示或丰富的奖励功能的形式，以学习长期任务。相反，我们的方法老板(引导你自己的技能)通过执行“技能引导”来学习完成新任务，在这种情况下，拥有一组原始技能的代理与环境交互，练习新技能，而不会收到对初始技能集以外的任务的奖励反馈。这个引导阶段由大型语言模型(LLM)指导，这些模型向代理提供有意义的技能，以便将它们链接在一起。通过这个过程，BOSS从一套基本的原始技能中建立了一系列复杂而有用的行为。我们通过在现实家庭环境中的实验证明，使用LLM指导的引导程序训练的代理在新环境中零命中执行看不见的长期任务方面优于使用朴素引导引导和先前的无监督技能获取方法训练的代理。网站：clvrai.com/oss。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23a/zhang23a.pdf)** 

# DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control
**题目:** DATT：用于四螺旋桨控制的深度自适应轨迹跟踪

**作者:** Kevin Huang, Rwik Rana, Alexander Spitzer, Guanya Shi, Byron Boots

**Abstract:** Precise arbitrary trajectory tracking for quadrotors is challenging due to unknown nonlinear dynamics, trajectory infeasibility, and actuation limits. To tackle these challenges, we present DATT, a learning-based approach that can precisely track arbitrary, potentially infeasible trajectories in the presence of large disturbances in the real world. DATT builds on a novel feedforward-feedback-adaptive control structure trained in simulation using reinforcement learning. When deployed on real hardware, DATT is augmented with a disturbance estimator using $\mathcal{L}_1$ adaptive control in closed-loop, without any fine-tuning. DATT significantly outperforms competitive adaptive nonlinear and model predictive controllers for both feasible smooth and infeasible trajectories in unsteady wind fields, including challenging scenarios where baselines completely fail. Moreover, DATT can efficiently run online with an inference time less than 3.2ms, less than 1/4 of the adaptive nonlinear model predictive control baseline.

**摘要:** 由于未知的非线性动力学、轨迹不可行和驱动限制，四旋翼飞行器精确的任意轨迹跟踪是具有挑战性的。为了应对这些挑战，我们提出了DATT，这是一种基于学习的方法，可以在现实世界中存在大扰动的情况下精确跟踪任意的、潜在不可行的轨迹。DATT建立在一种新颖的前馈-反馈-自适应控制结构的基础上，通过强化学习进行仿真训练。当部署在实际硬件上时，DATT在没有任何微调的情况下，在闭环系统中使用$\数学{L}_1$自适应控制来增加扰动估计器。在非定常风场中，无论是可行的平滑轨迹还是不可行的轨迹，包括基线完全失效的挑战性场景，DATT的性能都明显优于竞争性的自适应非线性和模型预测控制器。此外，DATT可以在线高效运行，推理时间小于3.2ms，不到自适应非线性模型预测控制基线的1/4。

**[Paper URL](https://proceedings.mlr.press/v229/huang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/huang23a/huang23a.pdf)** 

# HANDLOOM: Learned Tracing of One-Dimensional Objects for Inspection and Manipulation
**题目:** HANDLOOM：用于检查和操纵的一维物体的学习追踪

**作者:** Vainavi Viswanath, Kaushik Shivakumar, Mallika Parulekar, Jainil Ajmera, Justin Kerr, Jeffrey Ichnowski, Richard Cheng, Thomas Kollar, Ken Goldberg

**Abstract:** Tracing – estimating the spatial state of – long deformable linear objects such as cables, threads, hoses, or ropes, is useful for a broad range of tasks in homes, retail, factories, construction, transportation, and healthcare. For long deformable linear objects (DLOs or simply cables) with many (over 25) crossings, we present HANDLOOM (Heterogeneous Autoregressive Learned Deformable Linear Object Observation and Manipulation) a learning-based algorithm that fits a trace to a greyscale image of cables. We evaluate HANDLOOM on semi-planar DLO configurations where each crossing involves at most 2 segments. HANDLOOM makes use of neural networks trained with 30,000 simulated examples and 568 real examples to autoregressively estimate traces of cables and classify crossings. Experiments find that in settings with multiple identical cables, HANDLOOM can trace each cable with $80%$ accuracy. In single-cable images, HANDLOOM can trace and identify knots with $77%$ accuracy. When HANDLOOM is incorporated into a bimanual robot system, it enables state-based imitation of knot tying with $80%$ accuracy, and it successfully untangles $64%$ of cable configurations across 3 levels of difficulty. Additionally, HANDLOOM demonstrates generalization to knot types and materials (rubber, cloth rope) not present in the training dataset with $85%$ accuracy. Supplementary material, including all code and an annotated dataset of RGB-D images of cables along with ground-truth traces, is at https://sites.google.com/view/cable-tracing.

**摘要:** 跟踪-估计长可变形线性对象的空间状态，如电缆、线程、软管或绳索，对于家庭、零售、工厂、建筑、交通和医疗保健中的广泛任务很有用。对于具有多个(超过25个)交叉点的长可变形线状物体(DLO或简单电缆)，我们提出了一种基于学习的算法，该算法将轨迹与线缆的灰度图像相匹配。我们在半平面DLO配置上评估了手摇织机，其中每个交叉最多涉及2个线段。手摇织机利用3万个模拟样本和568个真实样本训练的神经网络来自动回归估计电缆的痕迹并对交叉路口进行分类。实验发现，在具有多条相同电缆的设置中，手摇织机可以以$80%$的精度跟踪每条电缆。在单线图像中，手摇织机可以追踪和识别结，精确度为77%$。当手摇织布机集成到双手机器人系统中时，它能够以80%的精度实现基于状态的打结模拟，并成功地解开了$%$的电缆配置，跨越了3个难度级别。此外，手织机演示了对训练数据集中不存在的结类型和材料(橡胶、布绳)的泛化，准确率为85%$。补充材料，包括所有代码和带注释的电缆RGB-D图像数据集，以及地面真实踪迹，请访问https://sites.google.com/view/cable-tracing.

**[Paper URL](https://proceedings.mlr.press/v229/viswanath23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/viswanath23a/viswanath23a.pdf)** 

# Predicting Object Interactions with Behavior Primitives: An Application in Stowing Tasks
**题目:** 使用行为基元预测对象交互：在存储任务中的应用

**作者:** Haonan Chen, Yilong Niu, Kaiwen Hong, Shuijing Liu, Yixuan Wang, Yunzhu Li, Katherine Rose Driggs-Campbell

**Abstract:** Stowing, the task of placing objects in cluttered shelves or bins, is a common task in warehouse and manufacturing operations. However, this task is still predominantly carried out by human workers as stowing is challenging to automate due to the complex multi-object interactions and long-horizon nature of the task. Previous works typically involve extensive data collection and costly human labeling of semantic priors across diverse object categories. This paper presents a method to learn a generalizable robot stowing policy from predictive model of object interactions and a single demonstration with behavior primitives. We propose a novel framework that utilizes Graph Neural Networks (GNNs) to predict object interactions within the parameter space of behavioral primitives. We further employ primitive-augmented trajectory optimization to search the parameters of a predefined library of heterogeneous behavioral primitives to instantiate the control action. Our framework enables robots to proficiently execute long-horizon stowing tasks with a few keyframes (3-4) from a single demonstration. Despite being solely trained in a simulation, our framework demonstrates remarkable generalization capabilities. It efficiently adapts to a broad spectrum of real-world conditions, including various shelf widths, fluctuating quantities of objects, and objects with diverse attributes such as sizes and shapes.

**摘要:** 堆放，即将物品放在杂乱的货架或垃圾箱中，是仓库和制造作业中常见的任务。然而，这项任务仍然主要由人类工人执行，因为由于任务的复杂的多对象相互作用和长期性质，装载自动化是具有挑战性的。以前的工作通常涉及广泛的数据收集和昂贵的人工标记跨不同对象类别的语义先验。提出了一种从对象交互的预测模型和行为基元的单次演示中学习可推广的机器人装载策略的方法。我们提出了一种新的框架，它利用图神经网络(GNN)来预测行为基元参数空间内的对象交互。我们进一步使用基元增广轨迹优化来搜索预定义的异类行为基元库的参数，以实例化控制动作。我们的框架使机器人能够熟练地执行来自单个演示的几个关键帧(3-4个)的长视距装载任务。尽管我们的框架仅在模拟中接受过训练，但它表现出了卓越的泛化能力。它有效地适应了广泛的现实世界条件，包括各种货架宽度、波动的对象数量以及具有不同属性(如大小和形状)的对象。

**[Paper URL](https://proceedings.mlr.press/v229/chen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23a/chen23a.pdf)** 

# Language to Rewards for Robotic Skill Synthesis
**题目:** 机器人技能合成的奖励语言

**作者:** Wenhao Yu, Nimrod Gileadi, Chuyuan Fu, Sean Kirmani, Kuang-Huei Lee, Montserrat Gonzalez Arenas, Hao-Tien Lewis Chiang, Tom Erez, Leonard Hasenclever, Jan Humplik, Brian Ichter, Ted Xiao, Peng Xu, Andy Zeng, Tingnan Zhang, Nicolas Heess, Dorsa Sadigh, Jie Tan, Yuval Tassa, Fei Xia

**Abstract:** Large language models (LLMs) have demonstrated exciting progress in acquiring diverse new capabilities through in-context learning, ranging from logical reasoning to code-writing. Robotics researchers have also explored using LLMs to advance the capabilities of robotic control. However, since low-level robot actions are hardware-dependent and underrepresented in LLM training corpora, existing efforts in applying LLMs to robotics have largely treated LLMs as semantic planners or relied on human-engineered control primitives to interface with the robot. On the other hand, reward functions are shown to be flexible representations that can be optimized for control policies to achieve diverse tasks, while their semantic richness makes them suitable to be specified by LLMs. In this work, we introduce a new paradigm that harnesses this realization by utilizing LLMs to define reward parameters that can be optimized and accomplish variety of robotic tasks. Using reward as the intermediate interface generated by LLMs, we can effectively bridge the gap between high-level language instructions or corrections to low-level robot actions. Meanwhile, combining this with a real-time optimizer, MuJoCo MPC, empowers an interactive behavior creation experience where users can immediately observe the results and provide feedback to the system. To systematically evaluate the performance of our proposed method, we designed a total of 17 tasks for a simulated quadruped robot and a dexterous manipulator robot. We demonstrate that our proposed method reliably tackles $90%$ of the designed tasks, while a baseline using primitive skills as the interface with Code-as-policies achieves $50%$ of the tasks. We further validated our method on a real robot arm where complex manipulation skills such as non-prehensile pushing emerge through our interactive system.

**摘要:** 大型语言模型(LLM)在通过从逻辑推理到代码编写的上下文学习获得各种新功能方面取得了令人兴奋的进展。机器人学研究人员还探索了使用LLMS来提高机器人控制能力。然而，由于低级机器人的动作依赖于硬件，并且在LLM训练语料库中的代表性很低，现有的将LLM应用于机器人学的努力在很大程度上将LLM视为语义规划器或依赖于人类工程的控制原语来与机器人交互。另一方面，奖励函数是一种灵活的表示，可以针对控制策略进行优化以实现不同的任务，而其丰富的语义使其适合于由LLMS指定。在这项工作中，我们引入了一种新的范例，通过利用LLMS来定义可以优化并完成各种机器人任务的奖励参数，从而利用这种实现。使用奖励作为LLMS生成的中间接口，可以有效地弥合高层语言指令或更正与低级机器人动作之间的差距。同时，将其与实时优化器MuJoCo MPC相结合，可以实现交互式行为创建体验，用户可以立即观察结果并向系统提供反馈。为了系统地评价我们提出的方法的性能，我们设计了一个模拟的四足机器人和一个灵巧的机械手机器人的总共17个任务。我们演示了我们提出的方法可靠地处理了$90%$的设计任务，而使用原始技能作为与代码即策略的接口的基线实现了$50%$的任务。我们在一个真实的机器人手臂上进一步验证了我们的方法，在这个手臂上，复杂的操作技能，如非抓取推送，通过我们的交互系统出现。

**[Paper URL](https://proceedings.mlr.press/v229/yu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yu23a/yu23a.pdf)** 

# Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation
**题目:** 蒸馏特征字段启用少镜头缩放引导操纵

**作者:** William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack Kaelbling, Phillip Isola

**Abstract:** Self-supervised and language-supervised image models contain rich knowledge of the world that is important for generalization. Many robotic tasks, however, require a detailed understanding of 3D geometry, which is often lacking in 2D image features. This work bridges this 2D-to-3D gap for robotic manipulation by leveraging distilled feature fields to combine accurate 3D geometry with rich semantics from 2D foundation models. We present a few-shot learning method for 6-DOF grasping and placing that harnesses these strong spatial and semantic priors to achieve in-the-wild generalization to unseen objects. Using features distilled from a vision-language model, CLIP, we present a way to designate novel objects for manipulation via free-text natural language, and demonstrate its ability to generalize to unseen expressions and novel categories of objects. Project website: https://f3rm.csail.mit.edu

**摘要:** 自我监督和语言监督的图像模型包含丰富的世界知识，这对于概括非常重要。然而，许多机器人任务需要详细了解3D几何形状，而2D图像特征通常缺乏这一点。这项工作通过利用提取的特征场将准确的3D几何形状与来自2D基础模型的丰富语义相结合，弥合了机器人操纵的2D到3D差距。我们提出了一种用于6自由度抓取和放置的几次学习方法，该方法利用这些强大的空间和语义先验来实现对不可见对象的野外概括。使用从视觉语言模型CLIP中提取的特征，我们提出了一种通过自由文本自然语言指定新颖对象进行操作的方法，并展示了其概括为不可见的表达和新颖对象类别的能力。项目网站：https://f3rm.csail.mit.edu

**[Paper URL](https://proceedings.mlr.press/v229/shen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shen23a/shen23a.pdf)** 

# Finetuning Offline World Models in the Real World
**题目:** 在现实世界中微调线下世界模特

**作者:** Yunhai Feng, Nicklas Hansen, Ziyan Xiong, Chandramouli Rajagopalan, Xiaolong Wang

**Abstract:** Reinforcement Learning (RL) is notoriously data-inefficient, which makes training on a real robot difficult. While model-based RL algorithms (world models) improve data-efficiency to some extent, they still require hours or days of interaction to learn skills. Recently, offline RL has been proposed as a framework for training RL policies on pre-existing datasets without any online interaction. However, constraining an algorithm to a fixed dataset induces a state-action distribution shift between training and inference, and limits its applicability to new tasks. In this work, we seek to get the best of both worlds: we consider the problem of pretraining a world model with offline data collected on a real robot, and then finetuning the model on online data collected by planning with the learned model. To mitigate extrapolation errors during online interaction, we propose to regularize the planner at test-time by balancing estimated returns and (epistemic) model uncertainty. We evaluate our method on a variety of visuo-motor control tasks in simulation and on a real robot, and find that our method enables few-shot finetuning to seen and unseen tasks even when offline data is limited. Videos are available at https://yunhaifeng.com/FOWM

**摘要:** 强化学习(RL)的数据效率低是出了名的，这使得在真实机器人上进行训练变得困难。虽然基于模型的RL算法(世界模型)在一定程度上提高了数据效率，但它们仍然需要数小时或数天的交互来学习技能。最近，离线RL被提出作为一个框架，用于在没有任何在线交互的情况下在现有数据集上训练RL策略。然而，将算法约束到固定的数据集会导致训练和推理之间的状态-动作分布转移，并限制其对新任务的适用性。在这项工作中，我们试图两全其美：我们考虑了用在真实机器人上收集的离线数据预先训练世界模型，然后根据学习到的模型规划收集的在线数据对模型进行微调的问题。为了减少在线交互中的外推误差，我们建议通过平衡估计回报和(认知)模型的不确定性来正规化测试时的计划者。我们在仿真和真实机器人上对各种视觉电机控制任务进行了评估，发现即使在离线数据有限的情况下，我们的方法也能够对可见和不可见的任务进行少镜头微调。有关视频，请访问https://yunhaifeng.com/FOWM

**[Paper URL](https://proceedings.mlr.press/v229/feng23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/feng23a/feng23a.pdf)** 

# Intent-Aware Planning in Heterogeneous Traffic via Distributed Multi-Agent Reinforcement Learning
**题目:** 通过分布式多智能体强化学习实现异类交通中的意图感知规划

**作者:** Xiyang Wu, Rohan Chandra, Tianrui Guan, Amrit Bedi, Dinesh Manocha

**Abstract:** Navigating safely and efficiently in dense and heterogeneous traffic scenarios is challenging for autonomous vehicles (AVs) due to their inability to infer the behaviors or intentions of nearby drivers. In this work, we introduce a distributed multi-agent reinforcement learning (MARL) algorithm for joint trajectory and intent prediction for autonomous vehicles in dense and heterogeneous environments. Our approach for intent-aware planning, iPLAN, allows agents to infer nearby drivers’ intents solely from their local observations. We model an explicit representation of agents’ private incentives: Behavioral Incentive for high-level decision-making strategy that sets planning sub-goals and Instant Incentive for low-level motion planning to execute sub-goals. Our approach enables agents to infer their opponents’ behavior incentives and integrate this inferred information into their decision-making and motion-planning processes. We perform experiments on two simulation environments, Non-Cooperative Navigation and Heterogeneous Highway. In Heterogeneous Highway, results show that, compared with centralized training decentralized execution (CTDE) MARL baselines such as QMIX and MAPPO, our method yields a $4.3%$ and $38.4%$ higher episodic reward in mild and chaotic traffic, with $48.1%$ higher success rate and $80.6%$ longer survival time in chaotic traffic. We also compare with a decentralized training decentralized execution (DTDE) baseline IPPO and demonstrate a higher episodic reward of $12.7%$ and $6.3%$ in mild traffic and chaotic traffic, $25.3%$ higher success rate, and $13.7%$ longer survival time.

**摘要:** 由于自动驾驶车辆(AVs)无法推断附近驾驶员的行为或意图，因此在密集和异质交通场景中安全高效地导航是具有挑战性的。在这项工作中，我们介绍了一种分布式多智能体强化学习(MAIL)算法，用于密集和异质环境中自主车辆的联合轨迹和意图预测。我们的意图感知规划方法iPlan允许代理仅根据他们当地的观察来推断附近司机的意图。我们建立了智能体私人激励的显式表示：高层决策策略设置规划子目标的行为激励和低层运动规划执行子目标的即时激励。我们的方法使代理能够推断对手的行为动机，并将推断的信息整合到他们的决策和运动规划过程中。我们在非合作导航和异质公路两个仿真环境上进行了实验。在异质公路上的实验结果表明，与QMIX和MAPPO等集中式训练分散执行(CTDE)方法相比，我们的方法在温和和混乱的交通中分别获得了4.3%和38.4%的情节回报，在混乱的交通中的成功率和生存时间分别高出48.1%和80.6%。我们还与分散训练分散执行(DTDE)基线IPPO进行了比较，显示出在温和交通和混乱交通中更高的插曲奖励12.7%$和6.3%$，成功率高25.3%$，生存时间长13.7%$。

**[Paper URL](https://proceedings.mlr.press/v229/wu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wu23b/wu23b.pdf)** 

# PreCo: Enhancing Generalization in Co-Design of Modular Soft Robots via Brain-Body Pre-Training
**题目:** PreCo：通过脑体预训练增强模块化软机器人联合设计的通用性

**作者:** Yuxing Wang, Shuang Wu, Tiantian Zhang, Yongzhe Chang, Haobo Fu, QIANG FU, Xueqian Wang

**Abstract:** Brain-body co-design, which involves the collaborative design of control strategies and morphologies, has emerged as a promising approach to enhance a robot’s adaptability to its environment. However, the conventional co-design process often starts from scratch, lacking the utilization of prior knowledge. This can result in time-consuming and costly endeavors. In this paper, we present PreCo, a novel methodology that efficiently integrates brain-body pre-training into the co-design process of modular soft robots. PreCo is based on the insight of embedding co-design principles into models, achieved by pre-training a universal co-design policy on a diverse set of tasks. This pre-trained co-designer is utilized to generate initial designs and control policies, which are then fine-tuned for specific co-design tasks. Through experiments on a modular soft robot system, our method demonstrates zero-shot generalization to unseen co-design tasks, facilitating few-shot adaptation while significantly reducing the number of policy iterations required.

**摘要:** 脑体协同设计涉及控制策略和形态的协同设计，是提高机器人对环境适应性的一种很有前途的方法。然而，传统的协同设计过程往往从头开始，缺乏对先验知识的利用。这可能会导致耗时和昂贵的努力。在本文中，我们提出了一种新的方法PRECO，它将脑体预训练有效地集成到模块化软机器人的协同设计过程中。PRECO基于将协同设计原则嵌入到模型中的洞察力，通过对一系列不同任务的通用协同设计政策进行预培训来实现。这个经过预先培训的协作设计者被用来生成初始设计和控制策略，然后针对特定的协作设计任务进行微调。通过在模块化软机器人系统上的实验，我们的方法展示了对看不见的协同设计任务的零镜头泛化，在促进少镜头适应的同时显著减少了所需的策略迭代次数。

**[Paper URL](https://proceedings.mlr.press/v229/wang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23b/wang23b.pdf)** 

# Diff-LfD: Contact-aware Model-based Learning from Visual Demonstration for Robotic Manipulation via Differentiable Physics-based Simulation and Rendering
**题目:** 差异-LfD：通过基于可区分物理的模拟和渲染从机器人操纵视觉演示中进行接触感知模型的学习

**作者:** Xinghao Zhu, JingHan Ke, Zhixuan Xu, Zhixin Sun, Bizhe Bai, Jun Lv, Qingtao Liu, Yuwei Zeng, Qi Ye, Cewu Lu, Masayoshi Tomizuka, Lin Shao

**Abstract:** Learning from Demonstration (LfD) is an efficient technique for robots to acquire new skills through expert observation, significantly mitigating the need for laborious manual reward function design. This paper introduces a novel framework for model-based LfD in the context of robotic manipulation. Our proposed pipeline is underpinned by two primary components: self-supervised pose and shape estimation and contact sequence generation. The former utilizes differentiable rendering to estimate object poses and shapes from demonstration videos, while the latter iteratively optimizes contact points and forces using differentiable simulation, consequently effectuating object transformations. Empirical evidence demonstrates the efficacy of our LfD pipeline in acquiring manipulation actions from human demonstrations. Complementary to this, ablation studies focusing on object tracking and contact sequence inference underscore the robustness and efficiency of our approach in generating long-horizon manipulation actions, even amidst environmental noise. Validation of our results extends to real-world deployment of the proposed pipeline. Supplementary materials and videos are available on our webpage.

**摘要:** 从演示中学习(LFD)是机器人通过专家观察获得新技能的一种有效技术，显著减少了人工设计奖励函数的繁琐需求。在机器人操作的背景下，提出了一种新的基于模型的LFD框架。我们提出的流水线由两个主要部分支撑：自监督姿态和形状估计以及接触序列生成。前者利用可微渲染从演示视频中估计对象的姿态和形状，而后者使用可微模拟迭代优化接触点和力，从而实现对象变换。经验证据表明，我们的LFD管道在从人类演示中获取操纵操作方面是有效的。与此相补充的是，专注于目标跟踪和接触序列推断的消融研究强调了我们的方法在生成长期操纵动作方面的健壮性和效率，即使在环境噪声中也是如此。对我们的结果的验证延伸到拟议的管道的实际部署。补充材料和视频可在我们的网页上找到。

**[Paper URL](https://proceedings.mlr.press/v229/zhu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhu23a/zhu23a.pdf)** 

# Surrogate Assisted Generation of Human-Robot Interaction Scenarios
**题目:** 代理人辅助生成人机交互场景

**作者:** Varun Bhatt, Heramb Nemlekar, Matthew Christopher Fontaine, Bryon Tjanaka, Hejia Zhang, Ya-Chuan Hsu, Stefanos Nikolaidis

**Abstract:** As human-robot interaction (HRI) systems advance, so does the difficulty of evaluating and understanding the strengths and limitations of these systems in different environments and with different users. To this end, previous methods have algorithmically generated diverse scenarios that reveal system failures in a shared control teleoperation task. However, these methods require directly evaluating generated scenarios by simulating robot policies and human actions. The computational cost of these evaluations limits their applicability in more complex domains. Thus, we propose augmenting scenario generation systems with surrogate models that predict both human and robot behaviors. In the shared control teleoperation domain and a more complex shared workspace collaboration task, we show that surrogate assisted scenario generation efficiently synthesizes diverse datasets of challenging scenarios. We demonstrate that these failures are reproducible in real-world interactions.

**摘要:** 随着人机交互（HRI）系统的发展，评估和理解这些系统在不同环境和不同用户中的优势和局限性的难度也在增加。为此，之前的方法通过算法生成了各种场景，这些场景揭示了共享控制遥操作任务中的系统故障。然而，这些方法需要通过模拟机器人策略和人类行为来直接评估生成的场景。这些评估的计算成本限制了它们在更复杂领域的适用性。因此，我们建议使用预测人类和机器人行为的代理模型来增强场景生成系统。在共享控制远程操作领域和更复杂的共享工作空间协作任务中，我们表明代理辅助场景生成可以有效地合成具有挑战性场景的各种数据集。我们证明这些失败在现实世界的互动中是可重复的。

**[Paper URL](https://proceedings.mlr.press/v229/bhatt23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/bhatt23a/bhatt23a.pdf)** 

# VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models
**题目:** VoxPoser：使用语言模型进行机器人操纵的可组合3D价值图

**作者:** Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei

**Abstract:** Large language models (LLMs) are shown to possess a wealth of actionable knowledge that can be extracted for robot manipulation in the form of reasoning and planning. Despite the progress, most still rely on pre-defined motion primitives to carry out the physical interactions with the environment, which remains a major bottleneck. In this work, we aim to synthesize robot trajectories, i.e., a dense sequence of 6-DoF end-effector waypoints, for a large variety of manipulation tasks given an open-set of instructions and an open-set of objects. We achieve this by first observing that LLMs excel at inferring affordances and constraints given a free-form language instruction. More importantly, by leveraging their code-writing capabilities, they can interact with a vision-language model (VLM) to compose 3D value maps to ground the knowledge into the observation space of the agent. The composed value maps are then used in a model-based planning framework to zero-shot synthesize closed-loop robot trajectories with robustness to dynamic perturbations. We further demonstrate how the proposed framework can benefit from online experiences by efficiently learning a dynamics model for scenes that involve contact-rich interactions. We present a large-scale study of the proposed method in both simulated and real-robot environments, showcasing the ability to perform a large variety of everyday manipulation tasks specified in free-form natural language.

**摘要:** 大型语言模型(LLM)具有丰富的可操作知识，可以通过推理和规划的形式提取出来用于机器人操作。尽管取得了进展，但大多数仍然依赖于预定义的运动基本体来与环境进行物理交互，这仍然是一个主要的瓶颈。在这项工作中，我们的目标是在给定开放指令集和开放对象集的情况下，为各种操作任务综合机器人轨迹，即密集的6-DOF末端执行器路点序列。我们首先观察到，在提供自由形式的语言教学的情况下，LLMS擅长推断负担能力和约束条件，从而实现了这一点。更重要的是，通过利用他们的代码编写能力，他们可以与视觉语言模型(VLM)交互，组成3D价值地图，将知识植根于代理的观察空间。然后将合成的值图用于基于模型的规划框架中，以零射击合成对动态扰动具有鲁棒性的闭环系统机器人轨迹。我们进一步演示了所提出的框架如何通过有效地学习涉及大量联系人交互的场景的动态模型来从在线经验中获益。我们在模拟和真实的机器人环境中对提出的方法进行了大规模研究，展示了执行自由形式自然语言指定的各种日常操作任务的能力。

**[Paper URL](https://proceedings.mlr.press/v229/huang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/huang23b/huang23b.pdf)** 

# Stabilize to Act: Learning to Coordinate for Bimanual Manipulation
**题目:** 稳定下来采取行动：学会协调双手操纵

**作者:** Jennifer Grannen, Yilin Wu, Brandon Vu, Dorsa Sadigh

**Abstract:** Key to rich, dexterous manipulation in the real world is the ability to coordinate control across two hands. However, while the promise afforded by bimanual robotic systems is immense, constructing control policies for dual arm autonomous systems brings inherent difficulties. One such difficulty is the high-dimensionality of the bimanual action space, which adds complexity to both model-based and data-driven methods. We counteract this challenge by drawing inspiration from humans to propose a novel role assignment framework: a stabilizing arm holds an object in place to simplify the environment while an acting arm executes the task. We instantiate this framework with BimanUal Dexterity from Stabilization (BUDS), which uses a learned restabilizing classifier to alternate between updating a learned stabilization position to keep the environment unchanged, and accomplishing the task with an acting policy learned from demonstrations. We evaluate BUDS on four bimanual tasks of varying complexities on real-world robots, such as zipping jackets and cutting vegetables. Given only 20 demonstrations, BUDS achieves $76.9%$ task success across our task suite, and generalizes to out-of-distribution objects within a class with a $52.7%$ success rate. BUDS is $56.0%$ more successful than an unstructured baseline that instead learns a BC stabilizing policy due to the precision required of these complex tasks. Supplementary material and videos can be found at https://tinyurl.com/stabilizetoact.

**摘要:** 在现实世界中，丰富而灵活的操作的关键是协调两只手的控制能力。然而，尽管双臂机器人系统的前景是巨大的，但为双臂自主系统构建控制策略带来了固有的困难。其中一个困难是双工动作空间的高维，这增加了基于模型的方法和数据驱动的方法的复杂性。我们从人类那里汲取灵感，提出了一个新的角色分配框架，以应对这一挑战：稳定的手臂将物体固定在适当的位置，以简化环境，而行动手臂则执行任务。我们用来自稳定的双手灵活性(BUDS)实例化了这个框架，它使用一个学习的重新稳定分类器来交替更新学习的稳定位置以保持环境不变，以及用从演示中学习的行动策略完成任务。我们评估了现实世界机器人上四个不同复杂程度的双工任务中的芽，例如拉链夹克和切蔬菜。如果只有20个演示，Buds在我们的任务套件中实现了$76.9%$任务成功，并以$52.7%$成功率概括到一个类中的分发外对象。BUDS$56.0%$比非结构化基线更成功，后者由于这些复杂任务所需的精确度而学习BC稳定政策。补充材料和视频可在https://tinyurl.com/stabilizetoact.上找到

**[Paper URL](https://proceedings.mlr.press/v229/grannen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/grannen23a/grannen23a.pdf)** 

# How to Learn and Generalize From Three Minutes of Data: Physics-Constrained and Uncertainty-Aware Neural Stochastic Differential Equations
**题目:** 如何从三分钟数据中学习和概括：物理约束和不确定性意识的神经随机方程

**作者:** Franck Djeumou, Cyrus Neary, Ufuk Topcu

**Abstract:** We present a framework and algorithms to learn controlled dynamics models using neural stochastic differential equations (SDEs)—SDEs whose drift and diffusion terms are both parametrized by neural networks. We construct the drift term to leverage a priori physics knowledge as inductive bias, and we design the diffusion term to represent a distance-aware estimate of the uncertainty in the learned model’s predictions—it matches the system’s underlying stochasticity when evaluated on states near those from the training dataset, and it predicts highly stochastic dynamics when evaluated on states beyond the training regime. The proposed neural SDEs can be evaluated quickly enough for use in model predictive control algorithms, or they can be used as simulators for model-based reinforcement learning. Furthermore, they make accurate predictions over long time horizons, even when trained on small datasets that cover limited regions of the state space. We demonstrate these capabilities through experiments on simulated robotic systems, as well as by using them to model and control a hexacopter’s flight dynamics: A neural SDE trained using only three minutes of manually collected flight data results in a model-based control policy that accurately tracks aggressive trajectories that push the hexacopter’s velocity and Euler angles to nearly double the maximum values observed in the training dataset.

**摘要:** 提出了一种利用神经随机微分方程(SDE)学习受控动力学模型的框架和算法，SDE的漂移和扩散项均由神经网络参数化。我们构造漂移项来利用先验物理知识作为归纳偏差，并且我们设计扩散项来表示对学习模型预测中的不确定性的距离感知估计-当对训练数据集中的状态附近的状态进行评估时，它与系统潜在的随机性相匹配，当对训练机制之外的状态进行评估时，它预测高度随机的动力学。所提出的神经SDE可以足够快地被评估用于模型预测控制算法，或者它们可以用作基于模型的强化学习的仿真器。此外，即使在覆盖状态空间有限区域的小数据集上进行训练，它们也能在长期范围内做出准确的预测。我们通过在模拟机器人系统上的实验以及使用它们来建模和控制六角飞行器的飞行动力学来展示这些能力：仅使用三分钟手动收集的飞行数据训练的神经SDE产生了基于模型的控制策略，该策略精确地跟踪攻击性轨迹，将六角飞行器的速度和欧拉角推到训练数据集中观察到的最大值的近两倍。

**[Paper URL](https://proceedings.mlr.press/v229/djeumou23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/djeumou23a/djeumou23a.pdf)** 

# Measuring Interpretability of Neural Policies of Robots with Disentangled Representation
**题目:** 用解纠缠表示测量机器人神经策略的可解释性

**作者:** Tsun-Hsuan Wang, Wei Xiao, Tim Seyde, Ramin Hasani, Daniela Rus

**Abstract:** The advancement of robots, particularly those functioning in complex human-centric environments, relies on control solutions that are driven by machine learning. Understanding how learning-based controllers make decisions is crucial since robots are mostly safety-critical systems. This urges a formal and quantitative understanding of the explanatory factors in the interpretability of robot learning. In this paper, we aim to study interpretability of compact neural policies through the lens of disentangled representation. We leverage decision trees to obtain factors of variation [1] for disentanglement in robot learning; these encapsulate skills, behaviors, or strategies toward solving tasks. To assess how well networks uncover the underlying task dynamics, we introduce interpretability metrics that measure disentanglement of learned neural dynamics from a concentration of decisions, mutual information and modularity perspective. We showcase the effectiveness of the connection between interpretability and disentanglement consistently across extensive experimental analysis.

**摘要:** 机器人的进步，特别是那些在复杂的以人为中心的环境中发挥作用的机器人，依赖于由机器学习驱动的控制解决方案。了解基于学习的控制器如何做出决策至关重要，因为机器人大多是安全关键型系统。这要求对机器人学习的可解释性中的解释性因素进行正式和定量的理解。在本文中，我们旨在通过解缠表示的透镜来研究紧致神经策略的可解释性。我们利用决策树来获得在机器人学习中解开纠缠的变异因素[1]；这些因素概括了解决任务的技能、行为或策略。为了评估网络揭示潜在任务动态的程度，我们引入了可解释性度量，从决策、互信息和模块化的角度衡量学习的神经动态的解缠程度。我们通过广泛的实验分析一致地展示了可解释性和解缠性之间的联系的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/wang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23c/wang23c.pdf)** 

# RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools
**题目:** RoboCook：使用多种工具进行长视野弹塑性物体操纵

**作者:** Haochen Shi, Huazhe Xu, Samuel Clarke, Yunzhu Li, Jiajun Wu

**Abstract:** Humans excel in complex long-horizon soft body manipulation tasks via flexible tool use: bread baking requires a knife to slice the dough and a rolling pin to flatten it. Often regarded as a hallmark of human cognition, tool use in autonomous robots remains limited due to challenges in understanding tool-object interactions. Here we develop an intelligent robotic system, RoboCook, which perceives, models, and manipulates elasto-plastic objects with various tools. RoboCook uses point cloud scene representations, models tool-object interactions with Graph Neural Networks (GNNs), and combines tool classification with self-supervised policy learning to devise manipulation plans. We demonstrate that from just 20 minutes of real-world interaction data per tool, a general-purpose robot arm can learn complex long-horizon soft object manipulation tasks, such as making dumplings and alphabet letter cookies. Extensive evaluations show that RoboCook substantially outperforms state-of-the-art approaches, exhibits robustness against severe external disturbances, and demonstrates adaptability to different materials.

**摘要:** 通过使用灵活的工具，人类在复杂的长视距软体操纵任务中表现出色：烘焙面包需要一把刀来切面团，需要一根滚棍来压平它。通常被认为是人类认知的标志，由于在理解工具-物体相互作用方面的挑战，工具在自主机器人中的使用仍然有限。在这里，我们开发了一个智能机器人系统RoboCook，它可以感知、建模并使用各种工具操纵弹塑性物体。RoboCook使用点云场景表示，使用图形神经网络(GNN)对工具-对象交互进行建模，并将工具分类与自我监督策略学习相结合来设计操作计划。我们演示了，从每个工具仅20分钟的真实交互数据，通用机械臂可以学习复杂的长视距软对象操作任务，如包饺子和字母饼干。广泛的评估表明，RoboCook的性能大大优于最先进的方法，表现出对严重外部干扰的稳健性，并表现出对不同材料的适应性。

**[Paper URL](https://proceedings.mlr.press/v229/shi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shi23a/shi23a.pdf)** 

# Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners
**题目:** 寻求帮助的机器人：大型语言模型规划者的不确定性对齐

**作者:** Allen Z. Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, Zhenjia Xu, Dorsa Sadigh, Andy Zeng, Anirudha Majumdar

**Abstract:** Large language models (LLMs) exhibit a wide range of promising capabilities — from step-by-step planning to commonsense reasoning — that may provide utility for robots, but remain prone to confidently hallucinated predictions. In this work, we present KnowNo, a framework for measuring and aligning the uncertainty of LLM-based planners, such that they know when they don’t know, and ask for help when needed. KnowNo builds on the theory of conformal prediction to provide statistical guarantees on task completion while minimizing human help in complex multi-step planning settings. Experiments across a variety of simulated and real robot setups that involve tasks with different modes of ambiguity (for example, from spatial to numeric uncertainties, from human preferences to Winograd schemas) show that KnowNo performs favorably over modern baselines (which may involve ensembles or extensive prompt tuning) in terms of improving efficiency and autonomy, while providing formal assurances. KnowNo can be used with LLMs out-of-the-box without model-finetuning, and suggests a promising lightweight approach to modeling uncertainty that can complement and scale with the growing capabilities of foundation models.

**摘要:** 大型语言模型(LLM)展示了从循序渐进的规划到常识性推理的广泛前景的能力，这些能力可能会为机器人提供实用工具，但仍然容易出现自信的幻觉预测。在这项工作中，我们提出了KnowNo，一个用于测量和调整基于LLM的规划者的不确定性的框架，以便他们知道他们何时不知道，并在需要时寻求帮助。KnowNo建立在保形预测理论的基础上，在复杂的多步骤规划环境中最大限度地减少人工帮助的同时，提供任务完成的统计保证。在各种模拟和真实的机器人设置上进行的实验表明，KnowNo在提高效率和自主性方面表现良好，同时提供了正式的保证。这些设置涉及不同模式的模糊任务(例如，从空间到数字的不确定性，从人类偏好到Winograd模式)。KnowNo可以与LLMS一起使用，而无需模型优化，并建议了一种有前途的轻量级建模不确定性方法，可以补充和扩展基础模型不断增长的能力。

**[Paper URL](https://proceedings.mlr.press/v229/ren23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ren23a/ren23a.pdf)** 

# Robot Learning with Sensorimotor Pre-training
**题目:** 使用感觉运动预训练的机器人学习

**作者:** Ilija Radosavovic, Baifeng Shi, Letian Fu, Ken Goldberg, Trevor Darrell, Jitendra Malik

**Abstract:** We present a self-supervised sensorimotor pre-training approach for robotics. Our model, called RPT, is a Transformer that operates on sequences of sensorimotor tokens. Given a sequence of camera images, proprioceptive robot states, and actions, we encode the sequence into tokens, mask out a subset, and train a model to predict the missing content from the rest. We hypothesize that if a robot can predict the masked-out content it will have acquired a good model of the physical world that can enable it to act. RPT is designed to operate on latent visual representations which makes prediction tractable, enables scaling to larger models, and allows fast inference on a real robot. To evaluate our approach, we collected a dataset of 20,000 real-world trajectories over 9 months using a combination of motion planning and grasping algorithms. We find that sensorimotor pre-training consistently outperforms training from scratch, has favorable scaling properties, and enables transfer across different tasks, environments, and robots.

**摘要:** 我们提出了一种自监督的机器人感知器预训练方法。我们的模型被称为RPT，是一个在感觉运动令牌序列上运行的变压器。给定一系列摄像机图像、本体感知机器人状态和动作，我们将该序列编码成令牌，掩蔽出一个子集，并训练一个模型来从其余内容中预测缺失的内容。我们假设，如果机器人能够预测被遮盖的内容，它就会获得一个很好的物理世界模型，使它能够行动。RPT被设计成对潜在的视觉表示进行操作，这使得预测变得容易处理，能够缩放到更大的模型，并允许在真实机器人上进行快速推理。为了评估我们的方法，我们使用运动规划和抓取算法的组合收集了9个月来20,000个真实世界轨迹的数据集。我们发现，感觉运动预训练始终优于从头开始的训练，具有良好的可伸缩性，并能够跨不同的任务、环境和机器人进行转移。

**[Paper URL](https://proceedings.mlr.press/v229/radosavovic23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/radosavovic23a/radosavovic23a.pdf)** 

# RVT: Robotic View Transformer for 3D Object Manipulation
**题目:** RVT：用于3D对象操纵的机器人视图Transformer

**作者:** Ankit Goyal, Jie Xu, Yijie Guo, Valts Blukis, Yu-Wei Chao, Dieter Fox

**Abstract:** For 3D object manipulation, methods that build an explicit 3D representation perform better than those relying only on camera images. But using explicit 3D representations like voxels comes at large computing cost, adversely affecting scalability. In this work, we propose RVT, a multi-view transformer for 3D manipulation that is both scalable and accurate. Some key features of RVT are an attention mechanism to aggregate information across views and re-rendering of the camera input from virtual views around the robot workspace. In simulations, we find that a single RVT model works well across 18 RLBench tasks with 249 task variations, achieving $26%$ higher relative success than the existing state-of-the-art method (PerAct). It also trains 36X faster than PerAct for achieving the same performance and achieves 2.3X the inference speed of PerAct. Further, RVT can perform a variety of manipulation tasks in the real world with just a few ($\sim$10) demonstrations per task. Visual results, code, and trained model are provided at: https://robotic-view-transformer.github.io/.

**摘要:** 对于3D对象操作，构建显式3D表示的方法比仅依赖相机图像的方法执行得更好。但使用像体素这样的显式3D表示会带来很大的计算成本，对可伸缩性产生不利影响。在这项工作中，我们提出了一种既可伸缩又准确的3D操作多视点变换RVT。RVT的一些关键功能是一种注意力机制，用于跨视图聚合信息，并从机器人工作空间周围的虚拟视图重新渲染相机输入。在模拟中，我们发现一个RVT模型可以很好地处理18个RLBtch任务和249个任务变化，获得的相对成功比现有的最先进的方法(PerAct)高26%$。为了达到相同的性能，它的训练速度比PerAct快36倍，推理速度是PerAct的2.3倍。此外，RVT可以在现实世界中执行各种操作任务，每个任务只需几个($\sim$10)演示。可视化结果、代码和经过训练的模型可在以下网址提供：https://robotic-view-transformer.github.io/.

**[Paper URL](https://proceedings.mlr.press/v229/goyal23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/goyal23a/goyal23a.pdf)** 

# ViNT: A Foundation Model for Visual Navigation
**题目:** ViNT：视觉导航的基础模型

**作者:** Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachowicz, Kevin Black, Noriaki Hirose, Sergey Levine

**Abstract:** General-purpose pre-trained models (“foundation models”) have enabled practitioners to produce generalizable solutions for individual machine learning problems with datasets that are significantly smaller than those required for learning from scratch. Such models are typically trained on large and diverse datasets with weak supervision, consuming much more training data than is available for any individual downstream application. In this paper, we describe the Visual Navigation Transformer (ViNT), a foundation model that aims to bring the success of general-purpose pre-trained models to vision-based robotic navigation. ViNT is trained with a general goal-reaching objective that can be used with any navigation dataset, and employs a flexible Transformer-based architecture to learn navigational affordances and enable efficient adaptation to a variety of downstream navigational tasks. ViNT is trained on a number of existing navigation datasets, comprising hundreds of hours of robotic navigation from a variety of different robotic platforms, and exhibits positive transfer, outperforming specialist models trained on narrower datasets. ViNT can be augmented with diffusion-based goal proposals to explore novel environments, and can solve kilometer-scale navigation problems when equipped with long-range heuristics. ViNT can also be adapted to novel task specifications with a technique inspired by prompt-tuning, where the goal encoder is replaced by an encoding of another task modality (e.g., GPS waypoints or turn-by-turn directions) embedded into the same space of goal tokens. This flexibility and ability to accommodate a variety of downstream problem domains establish ViNT as an effective foundation model for mobile robotics.

**摘要:** 通用的预先训练的模型(“基础模型”)使从业者能够用比从头开始学习所需的数据集小得多的数据集，为个别机器学习问题产生可推广的解决方案。这类模型通常是在监督较弱的大而多样的数据集上进行训练的，消耗的训练数据比任何单独的下游应用程序都多得多。在本文中，我们描述了视觉导航转换器(VINT)，这是一个基础模型，旨在将通用的预训练模型成功地引入基于视觉的机器人导航。VINT采用可用于任何导航数据集的通用目标实现目标进行培训，并采用灵活的基于Transformer的架构来学习导航负担并实现对各种下游导航任务的高效适应。Vint在许多现有的导航数据集上进行了训练，包括从各种不同的机器人平台进行数百小时的机器人导航，并表现出正迁移，表现出比在较窄数据集上训练的专家模型更好的表现。VINT可以用基于扩散的目标建议来扩展，以探索新的环境，并且当配备远程启发式算法时，可以解决公里级的导航问题。VINT还可以通过一种受提示调整启发的技术来适应新的任务规范，其中，目标编码器被嵌入到目标令牌的相同空间中的另一任务形态(例如，GPS路点或逐个转弯方向)的编码所取代。这种灵活性和适应各种下游问题领域的能力使VINT成为移动机器人的有效基础模型。

**[Paper URL](https://proceedings.mlr.press/v229/shah23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shah23a/shah23a.pdf)** 

# What Went Wrong? Closing the Sim-to-Real Gap via Differentiable Causal Discovery
**题目:** 出了什么问题？通过发现差异性原因缩小模拟与真实的差距

**作者:** Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis, Bingqing Chen, Ding Zhao

**Abstract:** Training control policies in simulation is more appealing than on real robots directly, as it allows for exploring diverse states in an efficient manner. Yet, robot simulators inevitably exhibit disparities from the real-world \rebut{dynamics}, yielding inaccuracies that manifest as the dynamical simulation-to-reality (sim-to-real) gap. Existing literature has proposed to close this gap by actively modifying specific simulator parameters to align the simulated data with real-world observations. However, the set of tunable parameters is usually manually selected to reduce the search space in a case-by-case manner, which is hard to scale up for complex systems and requires extensive domain knowledge. To address the scalability issue and automate the parameter-tuning process, we introduce COMPASS, which aligns the simulator with the real world by discovering the causal relationship between the environment parameters and the sim-to-real gap. Concretely, our method learns a differentiable mapping from the environment parameters to the differences between simulated and real-world robot-object trajectories. This mapping is governed by a simultaneously learned causal graph to help prune the search space of parameters, provide better interpretability, and improve generalization on unseen parameters. We perform experiments to achieve both sim-to-sim and sim-to-real transfer, and show that our method has significant improvements in trajectory alignment and task success rate over strong baselines in several challenging manipulation tasks. Demos are available on our project website: https://sites.google.com/view/sim2real-compass.

**摘要:** 在仿真中训练控制策略比直接在真实机器人上训练更有吸引力，因为它允许以有效的方式探索不同的状态。然而，机器人模拟器不可避免地表现出与真实世界的差异，产生的不准确表现为动态模拟与现实(sim-to-reale)的差距。现有的文献建议通过积极修改特定的模拟器参数来缩小这一差距，以使模拟数据与真实世界的观测结果保持一致。然而，可调参数集通常是手动选择的，以逐个案例的方式减少搜索空间，这对于复杂系统来说很难扩展，并且需要广泛的领域知识。为了解决可伸缩性问题并自动化参数调整过程，我们引入了COMPASS，它通过发现环境参数和模拟与真实之间的差距之间的因果关系来使模拟器与真实世界保持一致。具体地说，我们的方法学习了从环境参数到模拟的和真实的机器人-对象轨迹之间的差异的可微映射。这种映射由同时学习的因果图来管理，以帮助修剪参数的搜索空间，提供更好的可解释性，并改进对不可见参数的泛化。我们通过实验实现了SIM-to-SIM和SIM-TO-REAL的转换，实验结果表明，在强基线条件下，在多个具有挑战性的操作任务中，我们的方法在轨迹对齐和任务成功率方面都有显著的提高。演示可在我们的项目网站上找到：https://sites.google.com/view/sim2real-compass.

**[Paper URL](https://proceedings.mlr.press/v229/huang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/huang23c/huang23c.pdf)** 

# Scalable Deep Kernel Gaussian Process for Vehicle Dynamics in Autonomous Racing
**题目:** 自主赛车车辆动力学的可扩展深核高斯过程

**作者:** Jingyun Ning, Madhur Behl

**Abstract:** Autonomous racing presents a challenging environment for testing the limits of autonomous vehicle technology. Accurately modeling the vehicle dynamics (with all forces and tires) is critical for high-speed racing, but it remains a difficult task and requires an intricate balance between run-time computational demands and modeling complexity. Researchers have proposed utilizing learning-based methods such as Gaussian Process (GP) for learning vehicle dynamics. However, current approaches often oversimplify the modeling process or apply strong assumptions, leading to unrealistic results that cannot translate to real-world settings. In this paper, we proposed DKL-SKIP method for vehicle dynamics modeling. Our approach outperforms standard GP methods and the N4SID system identification technique in terms of prediction accuracy. In addition to evaluating DKL-SKIP on real-world data, we also evaluate its performance using a high-fidelity autonomous racing AutoVerse simulator. The results highlight the potential of DKL-SKIP as a promising tool for modeling complex vehicle dynamics in both real-world and simulated environments.

**摘要:** 自动驾驶比赛为测试自动驾驶汽车技术的极限提供了一个具有挑战性的环境。准确地建模车辆动力学(使用所有的力和轮胎)对于高速比赛至关重要，但这仍然是一项艰巨的任务，需要在运行时计算需求和建模复杂性之间取得复杂的平衡。研究人员提出了利用基于学习的方法，如高斯过程(GP)来学习车辆动力学。然而，当前的方法经常过度简化建模过程或应用强烈的假设，导致不现实的结果无法转化为真实世界的设置。本文提出了用于车辆动力学建模的DKL-SKIP方法。在预测精度方面，我们的方法优于标准GP方法和N4SID系统辨识技术。除了在真实数据上评估DKL-SKIP之外，我们还使用高保真自主赛车AutoVerse模拟器来评估其性能。研究结果表明，DKL-SKIP是一种在真实环境和仿真环境中建模复杂车辆动力学的有前途的工具。

**[Paper URL](https://proceedings.mlr.press/v229/ning23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ning23a/ning23a.pdf)** 

# Autonomous Robotic Reinforcement Learning with Asynchronous Human Feedback
**题目:** 具有非同步人类反馈的自主机器人强化学习

**作者:** Max Balsells, Marcel Torne Villasevil, Zihan Wang, Samedh Desai, Pulkit Agrawal, Abhishek Gupta

**Abstract:** Ideally, we would place a robot in a real-world environment and leave it there improving on its own by gathering more experience autonomously. However, algorithms for autonomous robotic learning have been challenging to realize in the real world. While this has often been attributed to the challenge of sample complexity, even sample-efficient techniques are hampered by two major challenges - the difficulty of providing well “shaped" rewards, and the difficulty of continual reset-free training. In this work, we describe a system for real-world reinforcement learning that enables agents to show continual improvement by training directly in the real world without requiring painstaking effort to hand-design reward functions or reset mechanisms. Our system leverages occasional non-expert human-in-the-loop feedback from remote users to learn informative distance functions to guide exploration while leveraging a simple self-supervised learning algorithm for goal-directed policy learning. We show that in the absence of resets, it is particularly important to account for the current “reachability" of the exploration policy when deciding which regions of the space to explore. Based on this insight, we instantiate a practical learning system - GEAR, which enables robots to simply be placed in real-world environments and left to train autonomously without interruption. The system streams robot experience to a web interface only requiring occasional asynchronous feedback from remote, crowdsourced, non-expert humans in the form of binary comparative feedback. We evaluate this system on a suite of robotic tasks in simulation and demonstrate its effectiveness at learning behaviors both in simulation and the real world. Project website https://guided-exploration-autonomous-rl.github.io/GEAR/.

**摘要:** 理想情况下，我们会把机器人放在现实世界的环境中，让它通过自主收集更多经验来自我改进。然而，用于自主机器人学习的算法在现实世界中的实现一直是具有挑战性的。虽然这常常被归因于样本复杂性的挑战，但即使是样本效率高的技术也受到两大挑战的阻碍--难以提供良好的“成形”奖励，以及难以持续地进行无重置训练。在这项工作中，我们描述了一个真实世界的强化学习系统，它使代理能够通过直接在现实世界中进行训练来表现出持续的改进，而不需要费力地手动设计奖励函数或重置机制。我们的系统利用远程用户偶尔提供的非专家人在环中反馈来学习信息性距离函数，以指导探索，同时利用简单的自我监督学习算法进行目标导向的策略学习。我们表明，在没有重置的情况下，在决定探索空间的哪些区域时，考虑到目前勘探政策的“可达性”尤为重要。基于这一见解，我们实例化了一个实际的学习系统-Gear，它使机器人能够简单地放置在真实世界的环境中，并不间断地自主训练。该系统将机器人体验传输到网络界面，只需偶尔以二进制比较反馈的形式从远程、众包、非专家人类那里获得异步反馈。我们在一组机器人仿真任务上对该系统进行了评估，并展示了它在仿真和真实世界中学习行为的有效性。项目网站https://guided-exploration-autonomous-rl.github.io/GEAR/.

**[Paper URL](https://proceedings.mlr.press/v229/balsells23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/balsells23a/balsells23a.pdf)** 

# Learning Realistic Traffic Agents in Closed-loop
**题目:** 闭环学习现实交通代理

**作者:** Chris Zhang, James Tu, Lunjun Zhang, Kelvin Wong, Simon Suo, Raquel Urtasun

**Abstract:** Realistic traffic simulation is crucial for developing self-driving software in a safe and scalable manner prior to real-world deployment. Typically, imitation learning (IL) is used to learn human-like traffic agents directly from real-world observations collected offline, but without explicit specification of traffic rules, agents trained from IL alone frequently display unrealistic infractions like collisions and driving off the road. This problem is exacerbated in out-of-distribution and long-tail scenarios. On the other hand, reinforcement learning (RL) can train traffic agents to avoid infractions, but using RL alone results in unhuman-like driving behaviors. We propose Reinforcing Traffic Rules (RTR), a holistic closed-loop learning objective to match expert demonstrations under a traffic compliance constraint, which naturally gives rise to a joint IL + RL approach, obtaining the best of both worlds. Our method learns in closed-loop simulations of both nominal scenarios from real-world datasets as well as procedurally generated long-tail scenarios. Our experiments show that RTR learns more realistic and generalizable traffic simulation policies, achieving significantly better tradeoffs between human-like driving and traffic compliance in both nominal and long-tail scenarios. Moreover, when used as a data generation tool for training prediction models, our learned traffic policy leads to considerably improved downstream prediction metrics compared to baseline traffic agents.

**摘要:** 逼真的交通模拟对于在现实世界部署之前以安全和可扩展的方式开发自动驾驶软件至关重要。通常，模仿学习(IL)用于直接从线下收集的真实世界观察中学习类似人类的交通智能体，但在没有明确指定交通规则的情况下，仅由模仿学习训练的智能体经常表现出不切实际的违规行为，如碰撞和驾驶偏离道路。这个问题在分配外和长尾的情况下会加剧。另一方面，强化学习(RL)可以训练交通智能体避免违规，但单独使用RL会导致非人类的驾驶行为。我们提出了加强交通规则(RTR)，这是一个整体的闭环学习目标，在交通遵从性约束下匹配专家演示，这自然产生了IL+RL的联合方法，获得了两个世界的最佳结果。我们的方法从真实世界数据集的名义场景以及程序生成的长尾场景的闭环模拟中学习。我们的实验表明，RTR学习了更现实和更具泛化的交通模拟策略，在名义和长尾场景下都实现了更好的模拟驾驶和交通遵从性之间的权衡。此外，当被用作训练预测模型的数据生成工具时，我们的学习交通策略导致与基准交通代理相比，下行预测度量显著改善。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23b/zhang23b.pdf)** 

# Leveraging 3D Reconstruction for Mechanical Search on Cluttered Shelves
**题目:** 利用3D重建在凌乱的架子上进行机械搜索

**作者:** Seungyeon Kim, Young Hun Kim, Yonghyeon Lee, Frank C. Park

**Abstract:** Finding and grasping a target object on a cluttered shelf, especially when the target is occluded by other unknown objects and initially invisible, remains a significant challenge in robotic manipulation. While there have been advances in finding the target object by rearranging surrounding objects using specialized tools, developing algorithms that work with standard robot grippers remains an unresolved issue. In this paper, we introduce a novel framework for finding and grasping the target object using a standard gripper, employing pushing and pick and-place actions. To achieve this, we introduce two indicator functions: (i) an existence function, determining the potential presence of the target, and (ii) a graspability function, assessing the feasibility of grasping the identified target. We then formulate a model-based optimal control problem. The core component of our approach involves leveraging a 3D recognition model, enabling efficient estimation of the proposed indicator functions and their associated dynamics models. Our method succeeds in finding and grasping the target object using a standard robot gripper in both simulations and real-world settings. In particular, we demonstrate the adaptability and robustness of our method in the presence of noise in real-world vision sensor data. The code for our framework is available at https://github.com/seungyeon-k/Search-for-Grasp-public.

**摘要:** 在杂乱的货架上寻找和抓住目标对象，特别是当目标被其他未知对象遮挡并且最初不可见的情况下，仍然是机器人操作中的一个重大挑战。虽然在通过使用专门工具重新排列周围物体来寻找目标物体方面取得了进展，但开发与标准机器人抓手配合使用的算法仍然是一个悬而未决的问题。在本文中，我们介绍了一种新的框架，使用标准的抓取器，采用推送和拾取放置动作来寻找和抓取目标对象。为了实现这一点，我们引入了两个指示函数：(I)确定目标潜在存在的存在函数，和(Ii)评估抓住所识别目标的可行性的可抓取性函数。然后，我们建立了一个基于模型的最优控制问题。我们方法的核心部分涉及利用3D识别模型，使得能够有效地估计所提议的指标函数及其关联的动力学模型。我们的方法在模拟和真实环境中都成功地使用标准的机器人抓手找到并抓取了目标对象。特别是，我们展示了我们的方法在真实世界视觉传感器数据中存在噪声的情况下的适应性和稳健性。我们框架的代码可以在https://github.com/seungyeon-k/Search-for-Grasp-public.上找到

**[Paper URL](https://proceedings.mlr.press/v229/kim23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kim23a/kim23a.pdf)** 

# SCONE: A Food Scooping Robot Learning Framework with Active Perception
**题目:** SCONE：具有主动感知的食物铲机器人学习框架

**作者:** Yen-Ling Tai, Yu Chien Chiu, Yu-Wei Chao, Yi-Ting Chen

**Abstract:** Effectively scooping food items poses a substantial challenge for current robotic systems, due to the intricate states and diverse physical properties of food. To address this challenge, we believe in the importance of encoding food items into meaningful representations for effective food scooping. However, the distinctive properties of food items, including deformability, fragility, fluidity, or granularity, pose significant challenges for existing representations. In this paper, we investigate the potential of active perception for learning meaningful food representations in an implicit manner. To this end, we present SCONE, a food-scooping robot learning framework that leverages representations gained from active perception to facilitate food scooping policy learning. SCONE comprises two crucial encoding components: the interactive encoder and the state retrieval module. Through the encoding process, SCONE is capable of capturing properties of food items and vital state characteristics. In our real-world scooping experiments, SCONE excels with a $71%$ success rate when tasked with 6 previously unseen food items across three different difficulty levels, surpassing state-of-theart methods. This enhanced performance underscores SCONE’s stability, as all food items consistently achieve task success rates exceeding $50%$. Additionally, SCONE’s impressive capacity to accommodate diverse initial states enables it to precisely evaluate the present condition of the food, resulting in a compelling scooping success rate. For further information, please visit our website: https://sites.google.com/view/corlscone/home.

**摘要:** 由于食品的复杂状态和不同的物理特性，有效地挖掘食品对当前的机器人系统构成了巨大的挑战。为了应对这一挑战，我们相信将食物编码成有意义的表示以有效地挖掘食物的重要性。然而，食品的独特属性，包括变形性、脆性、流动性或粒度，对现有的表示法构成了巨大的挑战。在这篇文章中，我们调查了主动知觉的潜力，以内隐的方式学习有意义的食物表征。为此，我们提出了Scon，一个食物铲机器人学习框架，它利用从主动感知获得的表征来促进食物铲策略学习。SCONE包括两个关键的编码组件：交互式编码器和状态检索模块。通过编码过程，SCONE能够捕获食物的属性和生命状态特征。在我们的现实世界勺子实验中，当我们在三个不同的难度级别上处理6种以前从未见过的食物时，Scon以71%的成功率出众，超过了最先进的方法。这一增强的表现突显了司康的稳定性，因为所有食品项目的任务成功率始终超过50%$。此外，Scon令人印象深刻的适应不同初始状态的能力使其能够准确地评估食物的现状，从而产生令人信服的挖取成功率。欲了解更多信息，请访问我们的网站：https://sites.google.com/view/corlscone/home.

**[Paper URL](https://proceedings.mlr.press/v229/tai23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tai23a/tai23a.pdf)** 

# Fine-Tuning Generative Models as an Inference Method for Robotic Tasks
**题目:** 微调生成模型作为机器人任务的推理方法

**作者:** Orr Krupnik, Elisei Shafer, Tom Jurgenson, Aviv Tamar

**Abstract:** Adaptable models could greatly benefit robotic agents operating in the real world, allowing them to deal with novel and varying conditions. While approaches such as Bayesian inference are well-studied frameworks for adapting models to evidence, we build on recent advances in deep generative models which have greatly affected many areas of robotics. Harnessing modern GPU acceleration, we investigate how to quickly adapt the sample generation of neural network models to observations in robotic tasks. We propose a simple and general method that is applicable to various deep generative models and robotic environments. The key idea is to quickly fine-tune the model by fitting it to generated samples matching the observed evidence, using the cross-entropy method. We show that our method can be applied to both autoregressive models and variational autoencoders, and demonstrate its usability in object shape inference from grasping, inverse kinematics calculation, and point cloud completion.

**摘要:** 自适应的模型可以极大地有利于在现实世界中操作的机器人代理，使它们能够处理新颖且变化的条件。虽然Bayesian推理等方法是经过充分研究的框架，用于使模型适应证据，但我们以深度生成模型的最新进展为基础，这些进展极大地影响了机器人的许多领域。利用现代图形处理器加速，我们研究如何快速调整神经网络模型的样本生成以适应机器人任务中的观察。我们提出了一种简单而通用的方法，适用于各种深度生成模型和机器人环境。关键思想是使用交叉信息量法将模型与与观察到的证据匹配的生成样本进行匹配，从而快速微调模型。我们表明，我们的方法可以应用于自回归模型和变分自动编码器，并证明了其在抓取、逆运动学计算和点云完成的物体形状推断中的可用性。

**[Paper URL](https://proceedings.mlr.press/v229/krupnik23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/krupnik23a/krupnik23a.pdf)** 

# Learning to Design and Use Tools for Robotic Manipulation
**题目:** 学习设计和使用机器人操纵工具

**作者:** Ziang Liu, Stephen Tian, Michelle Guo, Karen Liu, Jiajun Wu

**Abstract:** When limited by their own morphologies, humans and some species of animals have the remarkable ability to use objects from the environment toward accomplishing otherwise impossible tasks. Robots might similarly unlock a range of additional capabilities through tool use. Recent techniques for jointly optimizing morphology and control via deep learning are effective at designing locomotion agents. But while outputting a single morphology makes sense for locomotion, manipulation involves a variety of strategies depending on the task goals at hand. A manipulation agent must be capable of rapidly prototyping specialized tools for different goals. Therefore, we propose learning a designer policy, rather than a single design. A designer policy is conditioned on task information and outputs a tool design that helps solve the task. A design-conditioned controller policy can then perform manipulation using these tools. In this work, we take a step towards this goal by introducing a reinforcement learning framework for jointly learning these policies. Through simulated manipulation tasks, we show that this framework is more sample efficient than prior methods in multi-goal or multi-variant settings, can perform zero-shot interpolation or fine-tuning to tackle previously unseen goals, and allows tradeoffs between the complexity of design and control policies under practical constraints. Finally, we deploy our learned policies onto a real robot. Please see our supplementary video and website at https://robotic-tool-design.github.io/ for visualizations.

**摘要:** 当受到自身形态的限制时，人类和某些种类的动物具有非凡的能力，可以利用环境中的物体来完成原本不可能完成的任务。类似地，机器人可能会通过使用工具来释放一系列额外的功能。最近通过深度学习联合优化形态和控制的技术在运动智能体的设计中是有效的。但是，虽然输出单一的形态对于运动是有意义的，但操纵涉及到各种策略，这取决于手头的任务目标。操纵剂必须能够为不同的目标快速制作专用工具的原型。因此，我们建议学习设计师的策略，而不是单一的设计。设计器策略以任务信息为条件，并输出帮助解决任务的工具设计。然后，受设计条件限制的控制器策略可以使用这些工具执行操作。在这项工作中，我们通过引入一个强化学习框架来联合学习这些策略，从而朝着这一目标迈出了一步。通过模拟操作任务，我们表明该框架在多目标或多变量环境下比以往的方法更有效，可以执行零点内插或微调来处理先前看不到的目标，并允许在实际约束下在设计策略和控制策略的复杂性之间进行权衡。最后，我们将学习到的策略部署到一个真正的机器人上。有关可视化内容，请参阅我们的补充视频和网站https://robotic-tool-design.github.io/。

**[Paper URL](https://proceedings.mlr.press/v229/liu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23b/liu23b.pdf)** 

# CLUE: Calibrated Latent Guidance for Offline Reinforcement Learning
**题目:** 线索：离线强化学习的校准潜在指南

**作者:** Jinxin Liu, Lipeng Zu, Li He, Donglin Wang

**Abstract:** Offline reinforcement learning (RL) aims to learn an optimal policy from pre-collected and labeled datasets, which eliminates the time-consuming data collection in online RL. However, offline RL still bears a large burden of specifying/handcrafting extrinsic rewards for each transition in the offline data. As a remedy for the labor-intensive labeling, we propose to endow offline RL tasks with a few expert data and utilize the limited expert data to drive intrinsic rewards, thus eliminating the need for extrinsic rewards. To achieve that, we introduce Calibrated Latent gUidancE (CLUE), which utilizes a conditional variational auto-encoder to learn a latent space such that intrinsic rewards can be directly qualified over the latent space. CLUE’s key idea is to align the intrinsic rewards consistent with the expert intention via enforcing the embeddings of expert data to a calibrated contextual representation. We instantiate the expert-driven intrinsic rewards in sparse-reward offline RL tasks, offline imitation learning (IL) tasks, and unsupervised offline RL tasks. Empirically, we find that CLUE can effectively improve the sparse-reward offline RL performance, outperform the state-of-the-art offline IL baselines, and discover diverse skills from static reward-free offline data.

**摘要:** 离线强化学习旨在从预先收集和标注的数据集中学习最优策略，消除了在线强化学习中耗时的数据收集问题。然而，离线RL仍然承担着为离线数据中的每一次转换指定/手工制作外部奖励的很大负担。作为对劳动密集型标注的补救措施，我们建议赋予离线RL任务一些专家数据，并利用有限的专家数据来驱动内在奖励，从而消除了对外部奖励的需要。为了实现这一点，我们引入了校准潜在制导(CLUE)，它利用条件变分自动编码器来学习潜在空间，使得内在奖励可以直接在潜在空间上被限定。CLUE的关键思想是通过将专家数据嵌入到校准的上下文表示中，使内在奖励与专家意图保持一致。我们在稀疏奖励的离线RL任务、离线模仿学习(IL)任务和无监督离线RL任务中实例化了专家驱动的内在奖励。实证研究发现，CLUE能够有效地提高稀疏奖励线下RL性能，表现优于最新的线下IL基线，并从静态无奖励线下数据中发现不同技能。

**[Paper URL](https://proceedings.mlr.press/v229/liu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23c/liu23c.pdf)** 

# DEFT: Dexterous Fine-Tuning for Hand Policies
**题目:** DEFT：手部政策的灵巧微调

**作者:** Aditya Kannan, Kenneth Shaw, Shikhar Bahl, Pragna Mannam, Deepak Pathak

**Abstract:** Dexterity is often seen as a cornerstone of complex manipulation. Humans are able to perform a host of skills with their hands, from making food to operating tools. In this paper, we investigate these challenges, especially in the case of soft, deformable objects as well as complex, relatively long-horizon tasks. Although, learning such behaviors from scratch can be data inefficient. To circumvent this, we propose a novel approach, DEFT (DExterous Fine-Tuning for Hand Policies), that leverages human-driven priors, which are executed directly in the real world. In order to improve upon these priors, DEFT involves an efficient online optimization procedure. With the integration of human-based learning and online fine-tuning, coupled with a soft robotic hand, DEFT demonstrates success across various tasks, establishing a robust, data-efficient pathway toward general dexterous manipulation. Please see our website at https://dexterousfinetuning.github.io for video results.

**摘要:** 灵活性通常被视为复杂操纵的基石。人类能够用双手执行一系列技能，从制作食物到操作工具。在本文中，我们研究了这些挑战，特别是在柔软、可变形物体以及复杂、相对长期任务的情况下。不过，从头开始学习此类行为可能会导致数据效率低下。为了避免这个问题，我们提出了一种新颖的方法，DEFT（Dexterous Fine-Tuning for Hand Policy），它利用了人类驱动的先验，这些先验直接在现实世界中执行。为了改进这些先验信息，DEFT涉及一个高效的在线优化过程。通过将基于人类的学习和在线微调相结合，再加上柔软的机器人手，DEFT在各种任务中取得了成功，建立了一条强大、数据高效的通用灵巧操纵途径。请访问我们的网站https://dexterousfinetuning.github.io了解视频结果。

**[Paper URL](https://proceedings.mlr.press/v229/kannan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kannan23a/kannan23a.pdf)** 

# One-Shot Imitation Learning: A Pose Estimation Perspective
**题目:** 一次性模仿学习：姿势估计的角度

**作者:** Pietro Vitiello, Kamil Dreczkowski, Edward Johns

**Abstract:** In this paper, we study imitation learning under the challenging setting of: (1) only a single demonstration, (2) no further data collection, and (3) no prior task or object knowledge. We show how, with these constraints, imitation learning can be formulated as a combination of trajectory transfer and unseen object pose estimation. To explore this idea, we provide an in-depth study on how state-of-the-art unseen object pose estimators perform for one-shot imitation learning on ten real-world tasks, and we take a deep dive into the effects that camera calibration, pose estimation error, and spatial generalisation have on task success rates. For videos, please visit www.robot-learning.uk/pose-estimation-perspective.

**摘要:** 在本文中，我们在具有挑战性的环境下研究模仿学习：（1）只有一次演示，（2）没有进一步的数据收集，（3）没有先前的任务或对象知识。我们展示了在这些约束下，如何将模仿学习公式化为轨迹传输和未见对象姿态估计的组合。为了探索这个想法，我们深入研究了最先进的未见对象姿态估计器如何在十个现实世界任务上进行一次性模仿学习，并深入研究了相机校准、姿态估计误差和空间概括对任务成功率的影响。有关视频，请访问www.robot-learning.uk/pose-estimation-perspective。

**[Paper URL](https://proceedings.mlr.press/v229/vitiello23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/vitiello23a/vitiello23a.pdf)** 

# Semantic Mechanical Search with Large Vision and Language Models
**题目:** 使用大视野和语言模型的语义机械搜索

**作者:** Satvik Sharma, Huang Huang, Kaushik Shivakumar, Lawrence Yunliang Chen, Ryan Hoque, Brian Ichter, Ken Goldberg

**Abstract:** Moving objects to find a fully-occluded target object, known as mechanical search, is a challenging problem in robotics. As objects are often organized semantically, we conjecture that semantic information about object relationships can facilitate mechanical search and reduce search time. Large pretrained vision and language models (VLMs and LLMs) have shown promise in generalizing to uncommon objects and previously unseen real-world environments. In this work, we propose a novel framework called Semantic Mechanical Search (SMS). SMS conducts scene understanding and generates a semantic occupancy distribution explicitly using LLMs. Compared to methods that rely on visual similarities offered by CLIP embeddings, SMS leverages the deep reasoning capabilities of LLMs. Unlike prior work that uses VLMs and LLMs as end-to-end planners, which may not integrate well with specialized geometric planners, SMS can serve as a plug-in semantic module for downstream manipulation or navigation policies. For mechanical search in closed-world settings such as shelves, we compare with a geometric-based planner and show that SMS improves mechanical search performance by $24%$ across the pharmacy, kitchen, and office domains in simulation and $47.1%$ in physical experiments. For open-world real environments, SMS can produce better semantic distributions compared to CLIP-based methods, with the potential to be integrated with downstream navigation policies to improve object navigation tasks. Code, data, videos, and Appendix are available here.

**摘要:** 移动物体以寻找完全遮挡的目标物体，称为机械搜索，是机器人学中的一个具有挑战性的问题。由于对象通常是按语义组织的，我们推测关于对象关系的语义信息可以促进机械搜索，减少搜索时间。大型预先训练的视觉和语言模型(VLM和LLM)在推广到不常见的对象和以前未见过的真实世界环境方面显示出了希望。在这项工作中，我们提出了一个新的框架，称为语义机械搜索(SMS)。短消息进行场景理解，并使用LLMS显式地生成语义占用率分布。与依赖于片段嵌入提供的视觉相似性的方法相比，短信利用了LLMS的深度推理能力。以前的工作使用VLM和LLM作为端到端规划器，可能无法与专门的几何规划器很好地集成，与此不同，SMS可以作为下游操作或导航策略的插件语义模块。对于封闭世界环境中的机械搜索，如货架，我们与基于几何的规划器进行了比较，结果表明，在模拟和物理实验中，SMS在药房、厨房和办公室领域的机械搜索性能分别提高了24%和47.1%。对于开放世界的真实环境，与基于剪辑的方法相比，短信可以产生更好的语义分布，并有可能与下游导航策略相集成，以改进对象导航任务。代码、数据、视频和附录都可以在这里找到。

**[Paper URL](https://proceedings.mlr.press/v229/sharma23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sharma23a/sharma23a.pdf)** 

# KITE: Keypoint-Conditioned Policies for Semantic Manipulation
**题目:** KITE：基于关键点条件的语义操纵策略

**作者:** Priya Sundaresan, Suneel Belkhale, Dorsa Sadigh, Jeannette Bohg

**Abstract:** While natural language offers a convenient shared interface for humans and robots, enabling robots to interpret and follow language commands remains a longstanding challenge in manipulation. A crucial step to realizing a performant instruction-following robot is achieving semantic manipulation – where a robot interprets language at different specificities, from high-level instructions like "Pick up the stuffed animal" to more detailed inputs like "Grab the left ear of the elephant." To tackle this, we propose Keypoints + Instructions to Execution, a two-step framework for semantic manipulation which attends to both scene semantics (distinguishing between different objects in a visual scene) and object semantics (precisely localizing different parts within an object instance). KITE first grounds an input instruction in a visual scene through 2D image keypoints, providing a highly accurate object-centric bias for downstream action inference. Provided an RGB-D scene observation, KITE then executes a learned keypoint-conditioned skill to carry out the instruction. The combined precision of keypoints and parameterized skills enables fine-grained manipulation with generalization to scene and object variations. Empirically, we demonstrate KITE in 3 real-world environments: long-horizon 6-DoF tabletop manipulation, semantic grasping, and a high-precision coffee-making task. In these settings, KITE achieves a $75%$, $70%$, and $71%$ overall success rate for instruction-following, respectively. KITE outperforms frameworks that opt for pre-trained visual language models over keypoint-based grounding, or omit skills in favor of end-to-end visuomotor control, all while being trained from fewer or comparable amounts of demonstrations. Supplementary material, datasets, code, and videos can be found on our website: https://tinyurl.com/kite-site.

**摘要:** 虽然自然语言为人类和机器人提供了一个方便的共享界面，但使机器人能够解释和遵循语言命令仍然是操作中的一个长期挑战。实现执行指令的机器人的关键一步是实现语义操作--在语义操作中，机器人以不同的细节解释语言，从高级指令到更详细的输入，从“拿起填充动物”到“抓住大象的左耳”。为了解决这个问题，我们提出了KeyPoints+Instructions to Execution，这是一个两步语义操作框架，既考虑了场景语义(区分可视场景中不同的对象)，也考虑了对象语义(精确定位对象实例中的不同部分)。Kite首先通过2D图像关键点将输入指令接地在视觉场景中，为下游动作推理提供高度精确的以对象为中心的偏差。在提供RGB-D场景观察的情况下，Kite然后执行学习的关键点条件技能来执行指令。关键点的组合精度和参数化技能使细粒度的操作能够概括到场景和对象的变化。经验上，我们在3个真实世界环境中演示了风筝：长视距6-DOF桌面操纵、语义抓取和高精度咖啡煮任务。在这些情况下，Kite在遵循指导方面的总体成功率分别为75%、70%和71%。Kite的表现优于那些选择预先训练的视觉语言模型而不是基于关键点的基础的框架，或者省略技能而支持端到端的视觉运动控制，所有这些都是通过较少或类似数量的演示进行训练的。补充材料、数据集、代码和视频可在我们的网站上找到：https://tinyurl.com/kite-site.

**[Paper URL](https://proceedings.mlr.press/v229/sundaresan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sundaresan23a/sundaresan23a.pdf)** 

# BM2CP: Efficient Collaborative Perception with LiDAR-Camera Modalities
**题目:** BM 2 CP：利用LiDART相机模式实现高效协作感知

**作者:** Binyu Zhao, Wei ZHANG, Zhaonian Zou

**Abstract:** Collaborative perception enables agents to share complementary perceptual information with nearby agents. This can significantly benefit the perception performance and alleviate the issues of single-view perception, such as occlusion and sparsity. Most proposed approaches mainly focus on single modality (especially LiDAR), and not fully exploit the superiority of multi-modal perception. We propose an collaborative perception paradigm, BM2CP, which employs LiDAR and camera to achieve efficient multi-modal perception. BM2CP utilizes LiDAR-guided modal fusion, cooperative depth generation and modality-guided intermediate fusion to acquire deep interactions between modalities and agents. Moreover, it is capable to cope with the special case that one of the sensors is unavailable. Extensive experiments validate that it outperforms the state-of-the-art methods with 50X lower communication volumes in real-world autonomous driving scenarios. Our code is available at supplementary materials.

**摘要:** 协作感知使代理能够与附近的代理共享补充感知信息。这可以显着提高感知性能并缓解单视图感知的问题，例如遮挡和稀疏。大多数提出的方法主要关注单一模式（尤其是LiDART），而没有充分利用多模式感知的优势。我们提出了一种协作感知范式BM 2 CP，它利用LiDART和相机来实现高效的多模式感知。BM 2 CP利用LiDART引导的模式融合、协作深度生成和模式引导的中间融合来获取模式和智能体之间的深度交互。此外，它能够应对其中一个传感器不可用的特殊情况。大量实验证实，它优于最先进的方法，在现实世界的自动驾驶场景中通信量降低了50倍。我们的代码可在补充材料中找到。

**[Paper URL](https://proceedings.mlr.press/v229/zhao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhao23a/zhao23a.pdf)** 

# That Sounds Right: Auditory Self-Supervision for Dynamic Robot Manipulation
**题目:** 听起来不错：动态机器人操纵的听觉自我监督

**作者:** Abitha Thankaraj, Lerrel Pinto

**Abstract:** Learning to produce contact-rich, dynamic behaviors from raw sensory data has been a longstanding challenge in robotics. Prominent approaches primarily focus on using visual and tactile sensing. However, pure vision often fails to capture high-frequency interaction, while current tactile sensors can be too delicate for large-scale data collection. In this work, we propose a data-centric approach to dynamic manipulation that uses an often ignored source of information – sound. We first collect a dataset of 25k interaction-sound pairs across five dynamic tasks using contact microphones. Then, given this data, we leverage self-supervised learning to accelerate behavior prediction from sound. Our experiments indicate that this self-supervised ‘pretraining’ is crucial to achieving high performance, with a $34.5%$ lower MSE than plain supervised learning and a $54.3%$ lower MSE over visual training. Importantly, we find that when asked to generate desired sound profiles, online rollouts of our models on a UR10 robot can produce dynamic behavior that achieves an average of $11.5%$ improvement over supervised learning on audio similarity metrics. Videos and audio data are best seen on our project website: aurl-anon.github.io

**摘要:** 学习从原始感官数据中产生接触丰富、动态的行为一直是机器人学中的一个长期挑战。突出的方法主要集中在使用视觉和触觉感知。然而，纯视觉往往无法捕捉到高频相互作用，而目前的触觉传感器对于大规模数据收集来说可能过于精致。在这项工作中，我们提出了一种以数据为中心的动态操作方法，该方法使用了一个经常被忽略的信息源-声音。我们首先使用接触式麦克风收集了五个动态任务中25k个交互声音对的数据集。然后，在给定这些数据的情况下，我们利用自我监督学习来加速从声音中预测行为。我们的实验表明，这种自我监督的预训练对于获得高性能至关重要，比普通监督学习的MSE低34.5%，比视觉训练的MSE低54.3%。重要的是，我们发现，当被要求生成所需的声音配置文件时，我们在UR10机器人上在线推出的模型可以产生动态行为，在音频相似性度量方面比监督学习平均提高11.5%$。视频和音频数据最好在我们的项目网站上查看：aurl-anon.githorb.io

**[Paper URL](https://proceedings.mlr.press/v229/thankaraj23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/thankaraj23a/thankaraj23a.pdf)** 

# ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting
**题目:** Manicast：具有成本意识的人类预测的协作操纵

**作者:** Kushal Kedia, Prithwish Dan, Atiksh Bhardwaj, Sanjiban Choudhury

**Abstract:** Seamless human-robot manipulation in close proximity relies on accurate forecasts of human motion. While there has been significant progress in learning forecast models at scale, when applied to manipulation tasks, these models accrue high errors at critical transition points leading to degradation in downstream planning performance. Our key insight is that instead of predicting the most likely human motion, it is sufficient to produce forecasts that capture how future human motion would affect the cost of a robot’s plan. We present ManiCast, a novel framework that learns cost-aware human forecasts and feeds them to a model predictive control planner to execute collaborative manipulation tasks. Our framework enables fluid, real-time interactions between a human and a 7-DoF robot arm across a number of real-world tasks such as reactive stirring, object handovers, and collaborative table setting. We evaluate both the motion forecasts and the end-to-end forecaster-planner system against a range of learned and heuristic baselines while additionally contributing new datasets. We release our code and datasets at https://portal-cornell.github.io/manicast/.

**摘要:** 近距离无缝的人类-机器人操作依赖于对人类运动的准确预测。虽然在大规模学习预测模型方面取得了重大进展，但当应用于操纵任务时，这些模型在关键转换点会产生很高的误差，导致下游计划绩效下降。我们的主要见解是，不是预测最有可能的人类运动，而是产生预测，以捕捉未来人类运动将如何影响机器人计划的成本。我们提出了ManiCast，这是一个新的框架，它学习有成本意识的人类预测，并将它们提供给模型预测控制规划器来执行协作操作任务。我们的框架支持人和7自由度机械臂之间流畅、实时的交互，跨越许多真实世界的任务，如反应搅拌、物体移交和协作桌子设置。我们根据一系列学习和启发式基线对运动预测和端到端预报员-计划员系统进行评估，同时还提供了新的数据集。我们在https://portal-cornell.github.io/manicast/.发布我们的代码和数据集

**[Paper URL](https://proceedings.mlr.press/v229/kedia23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kedia23a/kedia23a.pdf)** 

# Predicting Routine Object Usage for Proactive Robot Assistance
**题目:** 预测主动机器人辅助的常规对象使用情况

**作者:** Maithili Patel, Aswin Gururaj Prakash, Sonia Chernova

**Abstract:** Proactivity in robot assistance refers to the robot’s ability to anticipate user needs and perform assistive actions without explicit requests. This requires understanding user routines, predicting consistent activities, and actively seeking information to predict inconsistent behaviors. We propose SLaTe-PRO (Sequential Latent Temporal model for Predicting Routine Object usage), which improves upon prior state-of-the-art by combining object and user action information, and conditioning object usage predictions on past history. Additionally, we find some human behavior to be inherently stochastic and lacking in contextual cues that the robot can use for proactive assistance. To address such cases, we introduce an interactive query mechanism that can be used to ask queries about the user’s intended activities and object use to improve prediction. We evaluate our approach on longitudinal data from three households, spanning 24 activity classes. SLaTe-PRO performance raises the F1 score metric to 0.57 without queries, and 0.60 with user queries, over a score of 0.43 from prior work. We additionally present a case study with a fully autonomous household robot.

**摘要:** 机器人辅助中的主动性是指机器人在没有明确请求的情况下预测用户需求并执行辅助动作的能力。这需要了解用户例程，预测一致的活动，并主动寻找信息来预测不一致的行为。我们提出了用于预测常规对象使用的序列潜在时间模型(SLATE-PRO)，该模型结合了对象和用户的行为信息，并根据过去的历史条件进行对象使用预测，从而改进了现有的研究现状。此外，我们发现一些人类行为具有内在的随机性，缺乏可供机器人用于主动协助的上下文线索。为了解决这种情况，我们引入了一种交互式查询机制，可以用来询问有关用户的预期活动和对象使用的查询，以提高预测能力。我们对我们的方法进行了评估，来自三个家庭的纵向数据，跨越24个活动类别。Slate-Pro性能将F1得分指标在没有查询的情况下提高到0.57，在有用户查询的情况下提高到0.60，高于之前工作的0.43分。此外，我们还介绍了一个完全自主的家用机器人的案例研究。

**[Paper URL](https://proceedings.mlr.press/v229/patel23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/patel23a/patel23a.pdf)** 

# Grounding Complex Natural Language Commands for Temporal Tasks in Unseen Environments
**题目:** 为不可见环境中的临时任务提供复杂自然语言命令的基础

**作者:** Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex, Ankit Shah

**Abstract:** Grounding navigational commands to linear temporal logic (LTL) leverages its unambiguous semantics for reasoning about long-horizon tasks and verifying the satisfaction of temporal constraints. Existing approaches require training data from the specific environment and landmarks that will be used in natural language to understand commands in those environments. We propose Lang2LTL, a modular system and a software package that leverages large language models (LLMs) to ground temporal navigational commands to LTL specifications in environments without prior language data. We comprehensively evaluate Lang2LTL for five well-defined generalization behaviors. Lang2LTL demonstrates the state-of-the-art ability of a single model to ground navigational commands to diverse temporal specifications in 21 city-scaled environments. Finally, we demonstrate a physical robot using Lang2LTL can follow 52 semantically diverse navigational commands in two indoor environments.

**摘要:** 将导航命令基于线性时态逻辑（LTL）利用其明确的语义来推理长期任务并验证时态约束的满足度。现有的方法需要来自特定环境和地标的训练数据，这些数据将用于自然语言来理解这些环境中的命令。我们提出了Lang2LTL，这是一个模块化系统和一个软件包，它利用大型语言模型（LLM）在没有先验语言数据的环境中将临时导航命令基础到LTL规范。我们对Lang2LTL的五种定义明确的概括行为进行了全面评估。Lang2LTL展示了单个模型的最先进能力，可以在21个城市规模的环境中根据不同的时间规范执行导航命令。最后，我们展示了使用Lang2LTL的物理机器人可以在两个室内环境中遵循52个语义不同的导航命令。

**[Paper URL](https://proceedings.mlr.press/v229/liu23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23d/liu23d.pdf)** 

# HOI4ABOT: Human-Object Interaction Anticipation for Human Intention Reading Collaborative roBOTs
**题目:** HOI 4ABOT：人类意图阅读协作roBOT的人-物交互预期

**作者:** Esteve Valls Mascaro, Daniel Sliwowski, Dongheui Lee

**Abstract:** Robots are becoming increasingly integrated into our lives, assisting us in various tasks. To ensure effective collaboration between humans and robots, it is essential that they understand our intentions and anticipate our actions. In this paper, we propose a Human-Object Interaction (HOI) anticipation framework for collaborative robots. We propose an efficient and robust transformer-based model to detect and anticipate HOIs from videos. This enhanced anticipation empowers robots to proactively assist humans, resulting in more efficient and intuitive collaborations. Our model outperforms state-of-the-art results in HOI detection and anticipation in VidHOI dataset with an increase of $1.76%$ and $1.04%$ in mAP respectively while being 15.4 times faster. We showcase the effectiveness of our approach through experimental results in a real robot, demonstrating that the robot’s ability to anticipate HOIs is key for better Human-Robot Interaction.

**摘要:** 机器人越来越融入我们的生活，协助我们完成各种任务。为了确保人类和机器人之间的有效合作，它们必须了解我们的意图并预测我们的行为。在本文中，我们提出了一个用于协作机器人的人机交互（HOI）预期框架。我们提出了一种高效且稳健的基于变换器的模型来检测和预测视频中的HOI。这种增强的预期使机器人能够主动协助人类，从而实现更高效、更直观的协作。我们的模型在VidHOI数据集中的HOI检测和预测方面优于最先进的结果，mAP分别增加了1.76%美元和1.04%美元，同时快了15.4倍。我们通过真实机器人的实验结果展示了我们方法的有效性，证明机器人预测HOI的能力是更好的人机交互的关键。

**[Paper URL](https://proceedings.mlr.press/v229/mascaro23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mascaro23a/mascaro23a.pdf)** 

# Reinforcement Learning Enables Real-Time Planning and Control of Agile Maneuvers for Soft Robot Arms
**题目:** 强化学习实现软机器人手臂敏捷机动的实时规划和控制

**作者:** Rianna Jitosho, Tyler Ga Wei Lum, Allison Okamura, Karen Liu

**Abstract:** Control policies for soft robot arms typically assume quasi-static motion or require a hand-designed motion plan. To achieve real-time planning and control for tasks requiring highly dynamic maneuvers, we apply deep reinforcement learning to train a policy entirely in simulation, and we identify strategies and insights that bridge the gap between simulation and reality. In particular, we strengthen the policy’s tolerance for inaccuracies with domain randomization and implement crucial simulator modifications that improve actuation and sensor modeling, enabling zero-shot sim-to-real transfer without requiring high-fidelity soft robot dynamics. We demonstrate the effectiveness of this approach with experiments on physical hardware and show that our soft robot can reach target positions that require dynamic swinging motions. This is the first work to achieve such agile maneuvers on a physical soft robot, advancing the field of soft robot arm planning and control. Our code and videos are publicly available at https://sites.google.com/view/rl-soft-robot.

**摘要:** 软机械臂的控制策略通常假定为准静态运动或需要手动设计的运动规划。为了实现对需要高度动态机动的任务的实时规划和控制，我们应用深度强化学习来完全在模拟中训练策略，并识别出弥合模拟和现实之间差距的策略和见解。特别是，我们通过域随机化加强了策略对不准确性的容忍度，并实施了关键的模拟器修改，改进了驱动和传感器建模，实现了零射击模拟到真实的传输，而不需要高保真的软机器人动力学。我们通过在物理硬件上的实验验证了该方法的有效性，并表明我们的软机器人可以到达需要动态摆动运动的目标位置。这是第一个在物理软机器人上实现这种灵活动作的工作，推动了软机器人手臂规划和控制领域的发展。我们的代码和视频可在https://sites.google.com/view/rl-soft-robot.上公开获取

**[Paper URL](https://proceedings.mlr.press/v229/jitosho23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/jitosho23a/jitosho23a.pdf)** 

# A Policy Optimization Method Towards Optimal-time Stability
**题目:** 实现最佳时间稳定性的政策优化方法

**作者:** Shengjie Wang, Lan Fengb, Xiang Zheng, Yuxue Cao, Oluwatosin OluwaPelumi Oseni, Haotian Xu, Tao Zhang, Yang Gao

**Abstract:** In current model-free reinforcement learning (RL) algorithms, stability criteria based on sampling methods are commonly utilized to guide policy optimization. However, these criteria only guarantee the infinite-time convergence of the system’s state to an equilibrium point, which leads to sub-optimality of the policy. In this paper, we propose a policy optimization technique incorporating sampling-based Lyapunov stability. Our approach enables the system’s state to reach an equilibrium point within an optimal time and maintain stability thereafter, referred to as "optimal-time stability". To achieve this, we integrate the optimization method into the Actor-Critic framework, resulting in the development of the Adaptive Lyapunov-based Actor-Critic (ALAC) algorithm. Through evaluations conducted on ten robotic tasks, our approach outperforms previous studies significantly, effectively guiding the system to generate stable patterns.

**摘要:** 在当前的无模型强化学习（RL）算法中，通常使用基于抽样方法的稳定性标准来指导策略优化。然而，这些标准只能保证系统状态无限时间收敛到平衡点，从而导致政策的次优性。在本文中，我们提出了一种结合基于采样的李雅普诺夫稳定性的政策优化技术。我们的方法使系统的状态能够在最佳时间内达到平衡点，并在此后保持稳定性，称为“最佳时间稳定性”。为了实现这一目标，我们将优化方法集成到Acor-Critic框架中，从而开发了基于自适应Lyapunov的Acor-Critic（LGA）算法。通过对十个机器人任务进行的评估，我们的方法显着优于之前的研究，有效地指导系统生成稳定的模式。

**[Paper URL](https://proceedings.mlr.press/v229/wang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23d/wang23d.pdf)** 

# An Unbiased Look at Datasets for Visuo-Motor Pre-Training
**题目:** 视觉运动预训练数据集的公正观察

**作者:** Sudeep Dasari, Mohan Kumar Srirama, Unnat Jain, Abhinav Gupta

**Abstract:** Visual representation learning hold great promise for robotics, but is severely hampered by the scarcity and homogeneity of robotics datasets. Recent works address this problem by pre-training visual representations on large-scale but out-of-domain data (e.g., videos of egocentric interactions) and then transferring them to target robotics tasks. While the field is heavily focused on developing better pre-training algorithms, we find that dataset choice is just as important to this paradigm’s success. After all, the representation can only learn the structures or priors present in the pre-training dataset. To this end, we flip the focus on algorithms, and instead conduct a dataset centric analysis of robotic pre-training. Our findings call into question some common wisdom in the field. We observe that traditional vision datasets (like ImageNet, Kinetics and 100 Days of Hands) are surprisingly competitive options for visuo-motor representation learning, and that the pre-training dataset’s image distribution matters more than its size. Finally, we show that common simulation benchmarks are not a reliable proxy for real world performance and that simple regularization strategies can dramatically improve real world policy learning.

**摘要:** 视觉表示学习为机器人学带来了巨大的希望，但由于机器人数据集的稀缺性和同质性，视觉表示学习受到了严重的阻碍。最近的工作通过在大规模但域外的数据(例如，以自我为中心的交互的视频)上预训练视觉表示，然后将它们传输到目标机器人任务来解决这个问题。虽然该领域非常专注于开发更好的预训练算法，但我们发现数据集的选择对这个范例的成功同样重要。毕竟，表示只能学习训练前数据集中存在的结构或先验。为此，我们将重点转移到算法上，转而进行以数据集为中心的机器人预训练分析。我们的发现对该领域的一些普遍看法提出了质疑。我们观察到，传统的视觉数据集(如ImageNet、Kinetics和100天的手)对于视觉-运动表征学习来说是令人惊讶的竞争选项，而且训练前数据集的图像分布比它的大小更重要。最后，我们表明，常见的模拟基准并不是真实世界绩效的可靠代理，简单的正则化策略可以显著改善现实世界的政策学习。

**[Paper URL](https://proceedings.mlr.press/v229/dasari23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/dasari23a/dasari23a.pdf)** 

# Equivariant Motion Manifold Primitives
**题目:** 等变运动Manifold Primitivity

**作者:** Byeongho Lee, Yonghyeon Lee, Seungyeon Kim, MinJun Son, Frank C. Park

**Abstract:** Existing movement primitive models for the most part focus on representing and generating a single trajectory for a given task, limiting their adaptability to situations in which unforeseen obstacles or new constraints may arise. In this work we propose Motion Manifold Primitives (MMP), a movement primitive paradigm that encodes and generates, for a given task, a continuous manifold of trajectories each of which can achieve the given task. To address the challenge of learning each motion manifold from a limited amount of data, we exploit inherent symmetries in the robot task by constructing motion manifold primitives that are equivariant with respect to given symmetry groups. Under the assumption that each of the MMPs can be smoothly deformed into each other, an autoencoder framework is developed to encode the MMPs and also generate solution trajectories. Experiments involving synthetic and real-robot examples demonstrate that our method outperforms existing manifold primitive methods by significant margins. Code is available at https://github.com/dlsfldl/EMMP-public.

**摘要:** 现有的运动基元模型大多侧重于表示和生成给定任务的单一轨迹，限制了它们对可能出现不可预见的障碍或新约束的情况的适应性。在这项工作中，我们提出了运动流形基元，这是一种运动基元范式，它为给定的任务编码并生成连续的轨迹流形，每个轨迹都可以完成给定的任务。为了解决从有限数量的数据中学习每个运动流形的挑战，我们通过构建相对于给定对称群等变的运动流形基元来利用机器人任务中固有的对称性。在假设每个MMP可以平滑变形为彼此的前提下，开发了一个自动编码框架来对MMP进行编码并生成解轨迹。合成和真实机器人实例的实验表明，我们的方法在很大程度上优于现有的流形基元方法。代码可在https://github.com/dlsfldl/EMMP-public.上找到

**[Paper URL](https://proceedings.mlr.press/v229/lee23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lee23a/lee23a.pdf)** 

# FlowBot++: Learning Generalized Articulated Objects Manipulation via Articulation Projection
**题目:** FlowBot++：通过关节投影学习广义关节对象操纵

**作者:** Harry Zhang, Ben Eisner, David Held

**Abstract:** Understanding and manipulating articulated objects, such as doors and drawers, is crucial for robots operating in human environments. We wish to develop a system that can learn to articulate novel objects with no prior interaction, after training on other articulated objects. Previous approaches for articulated object manipulation rely on either modular methods which are brittle or end-to-end methods, which lack generalizability. This paper presents FlowBot++, a deep 3D vision-based robotic system that predicts dense per-point motion and dense articulation parameters of articulated objects to assist in downstream manipulation tasks. FlowBot++ introduces a novel per-point representation of the articulated motion and articulation parameters that are combined to produce a more accurate estimate than either method on their own. Simulated experiments on the PartNet-Mobility dataset validate the performance of our system in articulating a wide range of objects, while real-world experiments on real objects’ point clouds and a Sawyer robot demonstrate the generalizability and feasibility of our system in real-world scenarios. Videos are available on our anonymized website https://sites.google.com/view/flowbotpp/home

**摘要:** 理解和操纵铰接式物体，如门和抽屉，对机器人在人类环境中运行至关重要。我们希望开发一种系统，在对其他铰接式对象进行培训后，可以在没有事先交互的情况下学习表达新对象。以前的关节对象操作方法要么依赖于脆弱的模块化方法，要么依赖于缺乏通用性的端到端方法。本文介绍了FlowBot++，这是一个基于深度3D视觉的机器人系统，它预测关节对象的密集逐点运动和密集关节参数，以辅助下游操作任务。FlowBot++引入了一种新颖的关节运动和关节参数的逐点表示法，它们结合在一起可以产生比单独使用任何一种方法都更准确的估计。在Partnet-Mobility数据集上的仿真实验验证了该系统对多种对象的关联性能，而在真实对象的点云和Sawyer机器人上的真实世界实验证明了该系统在真实场景中的通用性和可行性。视频可以在我们的匿名网站https://sites.google.com/view/flowbotpp/home上找到

**[Paper URL](https://proceedings.mlr.press/v229/zhang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23c/zhang23c.pdf)** 

# Geometry Matching for Multi-Embodiment Grasping
**题目:** 多实体抓取的几何匹配

**作者:** Maria Attarian, Muhammad Adil Asif, Jingzhou Liu, Ruthrash Hari, Animesh Garg, Igor Gilitschenski, Jonathan Tompson

**Abstract:** While significant progress has been made on the problem of generating grasps, many existing learning-based approaches still concentrate on a single embodiment, provide limited generalization to higher DoF end-effectors and cannot capture a diverse set of grasp modes. In this paper, we tackle the problem of grasping multi-embodiments through the viewpoint of learning rich geometric representations for both objects and end-effectors using Graph Neural Networks (GNN). Our novel method – GeoMatch – applies supervised learning on grasping data from multiple embodiments, learning end-to-end contact point likelihood maps as well as conditional autoregressive prediction of grasps keypoint-by-keypoint. We compare our method against 3 baselines that provide multi-embodiment support. Our approach performs better across 3 end-effectors, while also providing competitive diversity of grasps. Examples can be found at geomatch.github.io.

**摘要:** 虽然在生成抓取的问题上取得了重大进展，但许多现有的基于学习的方法仍然集中在单个实施例上，为更高DoF末端效应器提供有限的概括，并且无法捕获多样化的抓取模式集。在本文中，我们通过使用图形神经网络（GNN）学习对象和末端效应器的丰富几何表示的观点来解决掌握多个实施例的问题。我们的新方法-- GeoMatch --将监督学习应用于抓取来自多个实施例的数据、学习端到端接触点似然图以及逐个关键点的抓取的条件自回归预测。我们将我们的方法与提供多实施例支持的3个基线进行比较。我们的方法在3个末端效应器上表现更好，同时还提供了有竞争力的抓取多样性。示例可在geomatch.github.io找到。

**[Paper URL](https://proceedings.mlr.press/v229/attarian23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/attarian23a/attarian23a.pdf)** 

# Contrastive Value Learning: Implicit Models for Simple Offline RL
**题目:** 对比价值学习：简单离线RL的内隐模型

**作者:** Bogdan Mazoure, Benjamin Eysenbach, Ofir Nachum, Jonathan Tompson

**Abstract:** Model-based reinforcement learning (RL) methods are appealing in the offline setting because they allow an agent to reason about the consequences of actions without interacting with the environment. While conventional model-based methods learn a 1-step model, predicting the immediate next state, these methods must be plugged into larger planning or RL systems to yield a policy. Can we model the environment dynamics in a different way, such that the learned model directly indicates the value of each action? In this paper, we propose Contrastive Value Learning (CVL), which learns an implicit, multi-step dynamics model. This model can be learned without access to reward functions, but nonetheless can be used to directly estimate the value of each action, without requiring any TD learning. Because this model represents the multi-step transitions implicitly, it avoids having to predict high-dimensional observations and thus scales to high-dimensional tasks. Our experiments demonstrate that CVL outperforms prior offline RL methods on complex robotics benchmarks.

**摘要:** 基于模型的强化学习(RL)方法在离线环境中很有吸引力，因为它们允许代理在不与环境交互的情况下对操作的结果进行推理。虽然传统的基于模型的方法学习一步模型，预测下一个状态，但这些方法必须插入到更大的规划或RL系统中，才能产生策略。我们能否以不同的方式对环境动态进行建模，以使学习到的模型直接指示每项行动的价值？在本文中，我们提出了对比值学习(CVL)，它学习一个隐式的、多步骤的动力学模型。该模型可以在不访问奖励功能的情况下学习，但仍然可以用于直接估计每个行动的价值，而不需要任何TD学习。由于该模型隐含地表示了多步骤转换，因此它避免了预测高维观测，从而扩展到高维任务。我们的实验表明，在复杂的机器人基准测试中，CVL的性能优于以前的离线RL方法。

**[Paper URL](https://proceedings.mlr.press/v229/mazoure23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mazoure23a/mazoure23a.pdf)** 

# Parting with Misconceptions about Learning-based Vehicle Motion Planning
**题目:** 告别对基于学习的车辆运动规划的误解

**作者:** Daniel Dauner, Marcel Hallgarten, Andreas Geiger, Kashyap Chitta

**Abstract:** The release of nuPlan marks a new era in vehicle motion planning research, offering the first large-scale real-world dataset and evaluation schemes requiring both precise short-term planning and long-horizon ego-forecasting. Existing systems struggle to simultaneously meet both requirements. Indeed, we find that these tasks are fundamentally misaligned and should be addressed independently. We further assess the current state of closed-loop planning in the field, revealing the limitations of learning-based methods in complex real-world scenarios and the value of simple rule-based priors such as centerline selection through lane graph search algorithms. More surprisingly, for the open-loop sub-task, we observe that the best results are achieved when using only this centerline as scene context (i.e., ignoring all information regarding the map and other agents). Combining these insights, we propose an extremely simple and efficient planner which outperforms an extensive set of competitors, winning the nuPlan planning challenge 2023.

**摘要:** NuPlan的发布标志着车辆运动规划研究的新纪元，提供了第一个需要精确的短期规划和长期自我预测的大规模真实世界数据集和评估方案。现有的系统很难同时满足这两个要求。事实上，我们发现，这些任务从根本上是错位的，应该独立处理。我们进一步评估了现场闭环规划的现状，揭示了基于学习的方法在复杂现实场景中的局限性，以及简单的基于规则的先验的价值，例如通过车道图搜索算法进行中心线选择。更令人惊讶的是，对于开环子任务，我们观察到，当仅使用这条中心线作为场景上下文时(即，忽略有关地图和其他代理的所有信息)，可以获得最佳结果。结合这些见解，我们提出了一个极其简单和高效的规划者，它的表现超过了一系列广泛的竞争对手，赢得了nuPlan 2023年规划挑战赛。

**[Paper URL](https://proceedings.mlr.press/v229/dauner23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/dauner23a/dauner23a.pdf)** 

# Learning Sequential Acquisition Policies for Robot-Assisted Feeding
**题目:** 学习机器人辅助喂食的顺序获取策略

**作者:** Priya Sundaresan, Jiajun Wu, Dorsa Sadigh

**Abstract:** A robot providing mealtime assistance must perform specialized maneuvers with various utensils in order to pick up and feed a range of food items. Beyond these dexterous low-level skills, an assistive robot must also plan these strategies in sequence over a long horizon to clear a plate and complete a meal. Previous methods in robot-assisted feeding introduce highly specialized primitives for food handling without a means to compose them together. Meanwhile, existing approaches to long-horizon manipulation lack the flexibility to embed highly specialized primitives into their frameworks. We propose Visual Action Planning OveR Sequences (VAPORS), a framework for long-horizon food acquisition. VAPORS learns a policy for high-level action selection by leveraging learned latent plate dynamics in simulation. To carry out sequential plans in the real world, VAPORS delegates action execution to visually parameterized primitives. We validate our approach on complex real-world acquisition trials involving noodle acquisition and bimanual scooping of jelly beans. Across 38 plates, VAPORS acquires much more efficiently than baselines, generalizes across realistic plate variations such as toppings and sauces, and qualitatively appeals to user feeding preferences in a survey conducted across 49 individuals. Code, datasets, videos, and supplementary materials can be found on our website: https://sites.google.com/view/vaporsbot.

**摘要:** 提供进餐帮助的机器人必须用各种器皿进行专门的动作，才能捡起和喂食一系列食物。除了这些灵巧的低级技能外，辅助机器人还必须在很长的时间内按顺序计划这些策略，以清理盘子并完成一顿饭。以前的机器人辅助喂食方法引入了高度专业化的基元来处理食物，而没有将它们组合在一起的手段。同时，现有的长期操纵方法缺乏将高度专门化的原语嵌入到其框架中的灵活性。我们提出了基于序列的视觉行动计划(VAPORS)，这是一个长期食物获取的框架。VAPORS通过在模拟中利用所学的潜在板块动力学来学习用于高级动作选择的策略。为了在现实世界中执行连续计划，Vapors将动作执行委托给可视的参数化基元。我们在复杂的现实世界收购试验中验证了我们的方法，这些试验涉及面条收购和双手挖取软糖豆。在一项对49个人进行的调查中，在38个盘子中，蒸汽的获取效率比基线高得多，概括了各种现实的盘子变化，如配料和酱料，并在质量上吸引了用户的喂养偏好。代码、数据集、视频和补充材料可在我们的网站上找到：https://sites.google.com/view/vaporsbot.

**[Paper URL](https://proceedings.mlr.press/v229/sundaresan23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sundaresan23b/sundaresan23b.pdf)** 

# Composable Part-Based Manipulation
**题目:** 基于可组合零件的操纵

**作者:** Weiyu Liu, Jiayuan Mao, Joy Hsu, Tucker Hermans, Animesh Garg, Jiajun Wu

**Abstract:** In this paper, we propose composable part-based manipulation (CPM), a novel approach that leverages object-part decomposition and part-part correspondences to improve learning and generalization of robotic manipulation skills. By considering the functional correspondences between object parts, we conceptualize functional actions, such as pouring and constrained placing, as combinations of different correspondence constraints. CPM comprises a collection of composable diffusion models, where each model captures a different inter-object correspondence. These diffusion models can generate parameters for manipulation skills based on the specific object parts. Leveraging part-based correspondences coupled with the task decomposition into distinct constraints enables strong generalization to novel objects and object categories. We validate our approach in both simulated and real-world scenarios, demonstrating its effectiveness in achieving robust and generalized manipulation capabilities.

**摘要:** 在本文中，我们提出了可组合的基于零件的操纵（CPM），这是一种利用对象零件分解和零件零件对应关系来改善机器人操纵技能的学习和概括的新型方法。通过考虑对象部分之间的功能对应关系，我们将功能动作（例如倾倒和约束放置）概念化为不同对应约束的组合。CPM包括一组可组合的扩散模型，其中每个模型捕获不同的对象间对应关系。这些扩散模型可以根据特定物体部分生成操纵技能的参数。利用基于零件的对应关系，再加上任务分解为不同的约束，能够对新颖的对象和对象类别进行强概括。我们在模拟和现实世界场景中验证了我们的方法，证明了其在实现稳健和广义操纵能力方面的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/liu23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23e/liu23e.pdf)** 

# Adv3D: Generating Safety-Critical 3D Objects through Closed-Loop Simulation
**题目:** Adv3 D：通过闭环模拟生成安全关键的3D对象

**作者:** Jay Sarva, Jingkang Wang, James Tu, Yuwen Xiong, Sivabalan Manivasagam, Raquel Urtasun

**Abstract:** Self-driving vehicles (SDVs) must be rigorously tested on a wide range of scenarios to ensure safe deployment. The industry typically relies on closed-loop simulation to evaluate how the SDV interacts on a corpus of synthetic and real scenarios and to verify good performance. However, they primarily only test the motion planning module of the system, and only consider behavior variations. It is key to evaluate the full autonomy system in closed-loop, and to understand how variations in sensor data based on scene appearance, such as the shape of actors, affect system performance. In this paper, we propose a framework, Adv3D, that takes real world scenarios and performs closed-loop sensor simulation to evaluate autonomy performance, and finds vehicle shapes that make the scenario more challenging, resulting in autonomy failures and uncomfortable SDV maneuvers. Unlike prior work that add contrived adversarial shapes to vehicle roof-tops or roadside to harm perception performance, we optimize a low-dimensional shape representation to modify the vehicle shape itself in a realistic manner to degrade full autonomy performance (e.g., perception, prediction, motion planning). Moreover, we find that the shape variations found with Adv3D optimized in closed-loop are much more effective than open-loop, demonstrating the importance of finding and testing scene appearance variations that affect full autonomy performance.

**摘要:** 自动驾驶车辆(SDVS)必须在广泛的场景下进行严格测试，以确保安全部署。该行业通常依靠闭环模拟来评估SDV如何在一系列合成和真实场景上进行交互，并验证良好的性能。然而，它们主要只测试系统的运动规划模块，并且只考虑行为变化。评价闭环系统中的全自主系统，了解传感器数据基于场景外观的变化，如演员的形状，是如何影响系统性能的关键。在本文中，我们提出了一个框架Adv3D，该框架采用真实世界的场景并进行闭环传感器仿真来评估自主性能，并找到使场景更具挑战性的车辆形状，从而导致自动驾驶失败和不舒服的SDV操纵。与以往在车顶或路边添加人为的对抗性形状以损害感知性能的工作不同，我们优化了低维形状表示，以现实的方式修改车辆形状本身，从而降低完全自主的性能(例如，感知、预测、运动规划)。此外，我们还发现，在开环的情况下，通过对Adv3D进行优化，发现的形状变化比开环的更有效，这说明了发现和测试影响完全自主性能的场景外观变化的重要性。

**[Paper URL](https://proceedings.mlr.press/v229/sarva23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sarva23a/sarva23a.pdf)** 

# FindThis: Language-Driven Object Disambiguation in Indoor Environments
**题目:** FindThis：室内环境中地理驱动的对象歧义消除

**作者:** Arjun Majumdar, Fei Xia, Brian Ichter, Dhruv Batra, Leonidas Guibas

**Abstract:** Natural language is naturally ambiguous. In this work, we consider interactions between a user and a mobile service robot tasked with locating a desired object, specified by a language utterance. We present a task FindThis, which addresses the problem of how to disambiguate and locate the particular object instance desired through a dialog with the user. To approach this problem we propose an algorithm, GoFind, which exploits visual attributes of the object that may be intrinsic (e.g., color, shape), or extrinsic (e.g., location, relationships to other entities), expressed in an open vocabulary. GoFind leverages the visual common sense learned by large language models to enable fine-grained object localization and attribute differentiation in a zero-shot manner. We also provide a new visio-linguistic dataset, 3D Objects in Context (3DOC), for evaluating agents on this task consisting of Google Scanned Objects placed in Habitat-Matterport 3D scenes. Finally, we validate our approach on a real robot operating in an unstructured physical office environment using complex fine-grained language instructions.

**摘要:** 自然语言自然是模棱两可的。在这项工作中，我们考虑了用户与移动服务机器人之间的交互，该移动服务机器人的任务是定位由语言话语指定的期望对象。我们提出了一个任务FindThis，它解决了如何通过与用户的对话来消除歧义和定位所需的特定对象实例的问题。为了解决这个问题，我们提出了一种算法，GoFind，它利用对象的视觉属性，这些视觉属性可以是以开放词汇表表示的内在的(例如，颜色、形状)，也可以是外在的(例如，位置、与其他实体的关系)。GoFind利用大型语言模型学习的视觉常识，以零距离方式实现细粒度对象本地化和属性区分。我们还提供了一个新的视觉语言数据集，3D上下文中的对象(3DOC)，用于评估这项任务中的代理，该任务由放置在Habite-Matterport 3D场景中的Google扫描的对象组成。最后，我们使用复杂的细粒度语言指令在一个运行在非结构化物理办公环境中的真实机器人上验证了我们的方法。

**[Paper URL](https://proceedings.mlr.press/v229/majumdar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/majumdar23a/majumdar23a.pdf)** 

# Action-Quantized Offline Reinforcement Learning for Robotic Skill Learning
**题目:** 用于机器人技能学习的时间量化离线强化学习

**作者:** Jianlan Luo, Perry Dong, Jeffrey Wu, Aviral Kumar, Xinyang Geng, Sergey Levine

**Abstract:** The offline reinforcement learning (RL) paradigm provides a general recipe to convert static behavior datasets into policies that can perform better than the policy that collected the data. While policy constraints, conservatism, and other methods for mitigating distributional shifts have made offline reinforcement learning more effective, the continuous action setting often necessitates various approximations for applying these techniques. Many of these challenges are greatly alleviated in discrete action settings, where offline RL constraints and regularizers can often be computed more precisely or even exactly. In this paper, we propose an adaptive scheme for action quantization. We use a VQ-VAE to learn state- conditioned action quantization, avoiding the exponential blowup that comes with naïve discretization of the action space. We show that several state-of-the-art offline RL methods such as IQL, CQL, and BRAC improve in performance on benchmarks when combined with our proposed discretization scheme. We further validate our approach on a set of challenging long-horizon complex robotic manipulation tasks in the Robomimic environment, where our discretized offline RL algorithms are able to improve upon their continuous counterparts by 2-3x. Our project page is at saqrl.github.io

**摘要:** 离线强化学习(RL)范例提供了将静态行为数据集转换为比收集数据的策略执行得更好的策略的一般方法。虽然策略约束、保守性和其他缓解分布变化的方法使得离线强化学习更有效，但连续动作设置通常需要应用这些技术的各种近似。在离散动作设置中，这些挑战中的许多都得到了极大的缓解，在离散动作设置中，离线RL约束和正则化通常可以被更准确地计算，甚至可以更准确地计算。本文提出了一种自适应动作量化方案。我们使用VQ-VAE来学习状态条件动作量化，避免了动作空间的天真离散化带来的指数爆炸。我们展示了几种最先进的离线RL方法，如IQL、CQL和BRAC，当与我们提出的离散化方案相结合时，在基准测试上的性能有所提高。在Robomimic环境中，我们进一步验证了我们的方法在一组具有挑战性的长期复杂机器人操作任务上的有效性，在该环境中，我们的离线RL算法能够比它们的连续同行提高2-3倍。我们的项目页面在saqrl.githorb.io

**[Paper URL](https://proceedings.mlr.press/v229/luo23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/luo23a/luo23a.pdf)** 

# Batch Differentiable Pose Refinement for In-The-Wild Camera/LiDAR Extrinsic Calibration
**题目:** 用于野外相机/LiDART外部校准的批量可区分姿态细化

**作者:** Lanke Frank Tarimo Fu, Maurice Fallon

**Abstract:** Accurate camera to LiDAR (Light Detection and Ranging) extrinsic calibration is important for robotic tasks carrying out tight sensor fusion — such as target tracking and odometry. Calibration is typically performed before deployment in controlled conditions using calibration targets, however, this limits scalability and subsequent recalibration. We propose a novel approach for target-free camera-LiDAR calibration using end-to-end direct alignment which doesn’t need calibration targets. Our batched formulation enhances sample efficiency during training and robustness at inference time. We present experimental results, on publicly available real-world data, demonstrating 1.6cm/$0.07^{\circ}$ median accuracy when transferred to unseen sensors from held-out data sequences. We also show state-of-the-art zero-shot transfer to unseen cameras, LiDARs, and environments.

**摘要:** 相机与LiDART（光检测和距离测量）的准确外部校准对于执行紧密传感器融合的机器人任务（例如目标跟踪和里程测量）非常重要。通常在受控条件下部署之前使用校准目标执行校准，然而，这限制了可扩展性和随后的重新校准。我们提出了一种新的无目标相机LiDART校准方法，使用端到端直接对准，不需要校准目标。我们的批量配方增强了训练期间的样本效率和推理时的稳健性。我们根据公开可用的现实世界数据展示了实验结果，证明了当从已发布的数据序列传输到不可见的传感器时，平均准确度为1.6厘米/0.07美元。我们还展示了对不可见相机、LiDART和环境的最先进零拍摄传输。

**[Paper URL](https://proceedings.mlr.press/v229/fu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/fu23a/fu23a.pdf)** 

# Fleet Active Learning: A Submodular Maximization Approach
**题目:** 舰队主动学习：子模块最大化方法

**作者:** Oguzhan Akcin, Orhan Unuvar, Onat Ure, Sandeep P. Chinchali

**Abstract:** In multi-robot systems, robots often gather data to improve the performance of their deep neural networks (DNNs) for perception and planning. Ideally, these robots should select the most informative samples from their local data distributions by employing active learning approaches. However, when the data collection is distributed among multiple robots, redundancy becomes an issue as different robots may select similar data points. To overcome this challenge, we propose a fleet active learning (FAL) framework in which robots collectively select informative data samples to enhance their DNN models. Our framework leverages submodular maximization techniques to prioritize the selection of samples with high information gain. Through an iterative algorithm, the robots coordinate their efforts to collectively select the most valuable samples while minimizing communication between robots. We provide a theoretical analysis of the performance of our proposed framework and show that it is able to approximate the NP-hard optimal solution. We demonstrate the effectiveness of our framework through experiments on real-world perception and classification datasets, which include autonomous driving datasets such as Berkeley DeepDrive. Our results show an improvement by up to $25.0 %$ in classification accuracy, $9.2 %$ in mean average precision and $48.5 %$ in the submodular objective value compared to a completely distributed baseline.

**摘要:** 在多机器人系统中，机器人经常收集数据以提高其深层神经网络(DNN)的感知和规划性能。理想情况下，这些机器人应该通过采用主动学习方法，从它们的本地数据分布中选择信息最丰富的样本。然而，当数据收集分布在多个机器人之间时，冗余成为一个问题，因为不同的机器人可能选择相似的数据点。为了克服这一挑战，我们提出了一种舰队主动学习(FAL)框架，在该框架中，机器人集体选择信息丰富的数据样本来增强其DNN模型。我们的框架利用子模最大化技术来优先选择具有高信息增益的样本。通过迭代算法，机器人协调他们的努力，共同选择最有价值的样本，同时最小化机器人之间的通信。我们对我们提出的框架的性能进行了理论分析，并证明它能够逼近NP-Hard最优解。我们通过在真实世界感知和分类数据集上的实验证明了我们的框架的有效性，其中包括伯克利DeepDrive等自动驾驶数据集。我们的结果表明，与完全分布式基线相比，分类准确率提高了25.0%美元，平均精度提高了9.2%美元，子模块目标值提高了48.5%美元。

**[Paper URL](https://proceedings.mlr.press/v229/akcin23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/akcin23a/akcin23a.pdf)** 

# Robust Reinforcement Learning in Continuous Control Tasks with Uncertainty Set Regularization
**题目:** 具有不确定集正规化的连续控制任务中的鲁棒强化学习

**作者:** Yuan Zhang, Jianhong Wang, Joschka Boedecker

**Abstract:** Reinforcement learning (RL) is recognized as lacking generalization and robustness under environmental perturbations, which excessively restricts its application for real-world robotics. Prior work claimed that adding regularization to the value function is equivalent to learning a robust policy under uncertain transitions. Although the regularization-robustness transformation is appealing for its simplicity and efficiency, it is still lacking in continuous control tasks. In this paper, we propose a new regularizer named Uncertainty Set Regularizer (USR), to formulate the uncertainty set on the parametric space of a transition function. To deal with unknown uncertainty sets, we further propose a novel adversarial approach to generate them based on the value function. We evaluate USR on the Real-world Reinforcement Learning (RWRL) benchmark and the Unitree A1 Robot, demonstrating improvements in the robust performance of perturbed testing environments and sim-to-real scenarios.

**摘要:** 强化学习（RL）被认为在环境扰动下缺乏普遍性和鲁棒性，这过度限制了其在现实世界机器人中的应用。之前的工作声称，向价值函数添加正规化相当于在不确定的转变下学习稳健的政策。尽管正规化-鲁棒性转换因其简单性和高效性而具有吸引力，但它仍然缺乏连续控制任务。在本文中，我们提出了一种新的规则化器，名为不确定性集规则化器（USR），用于在转移函数的参数空间上表达不确定性集。为了处理未知的不确定性集，我们进一步提出了一种新颖的对抗方法来基于价值函数生成它们。我们在现实世界强化学习（RWRL）基准测试和Unitree A1机器人上评估USR，展示了受干扰测试环境和模拟到真实场景的稳健性能的改进。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23d/zhang23d.pdf)** 

# Context-Aware Deep Reinforcement Learning for Autonomous Robotic Navigation in Unknown Area
**题目:** 上下文感知深度强化学习用于未知区域自主机器人导航

**作者:** Jingsong Liang, Zhichen Wang, Yuhong Cao, Jimmy Chiun, Mengqi Zhang, Guillaume Adrien Sartoretti

**Abstract:** Mapless navigation refers to a challenging task where a mobile robot must rapidly navigate to a predefined destination using its partial knowledge of the environment, which is updated online along the way, instead of a prior map of the environment. Inspired by the recent developments in deep reinforcement learning (DRL), we propose a learning-based framework for mapless navigation, which employs a context-aware policy network to achieve efficient decision-making (i.e., maximize the likelihood of finding the shortest route towards the target destination), especially in complex and large-scale environments. Specifically, our robot learns to form a context of its belief over the entire known area, which it uses to reason about long-term efficiency and sequence show-term movements. Additionally, we propose a graph rarefaction algorithm to enable more efficient decision-making in large-scale applications. We empirically demonstrate that our approach reduces average travel time by up to $61.4%$ and average planning time by up to $88.2%$ compared to benchmark planners (D*lite and BIT) on hundreds of test scenarios. We also validate our approach both in high-fidelity Gazebo simulations as well as on hardware, highlighting its promising applicability in the real world without further training/tuning.

**摘要:** 无人导航是指一项具有挑战性的任务，移动机器人必须使用其对环境的部分知识快速导航到预定的目的地，这些知识在一路上在线更新，而不是之前的环境地图。受深度强化学习(DRL)发展的启发，我们提出了一种基于学习的无障碍导航框架，该框架使用上下文感知策略网络来实现高效的决策(即最大化找到到达目标目的地的最短路径的可能性)，特别是在复杂和大规模的环境中。具体地说，我们的机器人学习在整个已知区域形成其信念的背景，它使用这一背景来推理长期效率和顺序显示的运动。此外，我们还提出了一种图稀疏算法，以便在大规模应用中更有效地进行决策。在数百个测试场景中，我们的经验表明，与基准计划者(D*LITE和BIT)相比，我们的方法将平均行程时间减少了61.4%美元，平均计划时间减少了88.2%美元。我们还在高保真Gazebo模拟和硬件上验证了我们的方法，强调了它在现实世界中的前景，而不需要进一步的培训/调整。

**[Paper URL](https://proceedings.mlr.press/v229/liang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liang23a/liang23a.pdf)** 

# Learning to Discern: Imitating Heterogeneous Human Demonstrations with Preference and Representation Learning
**题目:** 学会辨别：用偏好和代表学习模仿异类人类演示

**作者:** Sachit Kuhar, Shuo Cheng, Shivang Chopra, Matthew Bronars, Danfei Xu

**Abstract:** Practical Imitation Learning (IL) systems rely on large human demonstration datasets for successful policy learning. However, challenges lie in maintaining the quality of collected data and addressing the suboptimal nature of some demonstrations, which can compromise the overall dataset quality and hence the learning outcome. Furthermore, the intrinsic heterogeneity in human behavior can produce equally successful but disparate demonstrations, further exacerbating the challenge of discerning demonstration quality. To address these challenges, this paper introduces Learning to Discern (L2D), an offline imitation learning framework for learning from demonstrations with diverse quality and style. Given a small batch of demonstrations with sparse quality labels, we learn a latent representation for temporally embedded trajectory segments. Preference learning in this latent space trains a quality evaluator that generalizes to new demonstrators exhibiting different styles. Empirically, we show that L2D can effectively assess and learn from varying demonstrations, thereby leading to improved policy performance across a range of tasks in both simulations and on a physical robot.

**摘要:** 实用的模仿学习(IL)系统依赖于大量的人类演示数据集来成功地进行策略学习。然而，挑战在于保持收集的数据的质量，并解决一些演示的次优性质，这可能会损害整体数据集的质量，从而影响学习结果。此外，人类行为的内在异质性可以产生同样成功但不同的演示，进一步加剧了辨别演示质量的挑战。为了应对这些挑战，本文引入了学习辨别(L2D)，这是一个用于从不同质量和风格的演示中学习的离线模仿学习框架。给出一小批带有稀疏质量标签的演示，我们学习了时间嵌入轨迹段的潜在表示。在这个潜在空间中的偏好学习培养了一名质量评估者，他能概括出展示不同风格的新示威者。经验表明，L2D可以有效地评估和学习不同的演示，从而在模拟和物理机器人上的一系列任务上提高策略性能。

**[Paper URL](https://proceedings.mlr.press/v229/kuhar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kuhar23a/kuhar23a.pdf)** 

# Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter
**题目:** 图像引导机器人抓取：基于CLIP的杂波中参考抓取合成

**作者:** Georgios Tziafas, Yucheng XU, Arushi Goel, Mohammadreza Kasaei, Zhibin Li, Hamidreza Kasaei

**Abstract:** Robots operating in human-centric environments require the integration of visual grounding and grasping capabilities to effectively manipulate objects based on user instructions. This work focuses on the task of referring grasp synthesis, which predicts a grasp pose for an object referred through natural language in cluttered scenes. Existing approaches often employ multi-stage pipelines that first segment the referred object and then propose a suitable grasp, and are evaluated in private datasets or simulators that do not capture the complexity of natural indoor scenes. To address these limitations, we develop a challenging benchmark based on cluttered indoor scenes from OCID dataset, for which we generate referring expressions and connect them with 4-DoF grasp poses. Further, we propose a novel end-to-end model (CROG) that leverages the visual grounding capabilities of CLIP to learn grasp synthesis directly from image-text pairs. Our results show that vanilla integration of CLIP with pretrained models transfers poorly in our challenging benchmark, while CROG achieves significant improvements both in terms of grounding and grasping. Extensive robot experiments in both simulation and hardware demonstrate the effectiveness of our approach in challenging interactive object grasping scenarios that include clutter.

**摘要:** 在以人为中心的环境中运行的机器人需要整合视觉接地和抓取能力，以根据用户指令有效地操纵对象。这项工作主要关注参照抓取合成的任务，该任务预测在杂乱场景中通过自然语言参照的对象的抓握姿势。现有的方法通常采用多阶段流水线，首先分割参考对象，然后提出合适的抓取，并在没有捕捉到自然室内场景的复杂性的私有数据集或模拟器中进行评估。为了解决这些局限性，我们开发了一个基于OCID数据集的室内杂乱场景的挑战性基准，为其生成指代表达式并将其与4-DOF抓取姿势联系起来。此外，我们提出了一种新的端到端模型(CROG)，该模型利用CLIP的视觉基础能力直接从图文对学习抓取合成。我们的结果表明，CLIP与预先训练的模型的普通集成在我们具有挑战性的基准中传输效果很差，而CROG在接地和抓取方面都取得了显著的改善。机器人在仿真和硬件上的大量实验证明了我们的方法在挑战包括杂乱的交互式对象抓取场景中的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/tziafas23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tziafas23a/tziafas23a.pdf)** 

# Learning Reusable Manipulation Strategies
**题目:** 学习可重复使用的操纵策略

**作者:** Jiayuan Mao, Tomás Lozano-Pérez, Joshua B. Tenenbaum, Leslie Pack Kaelbling

**Abstract:** Humans demonstrate an impressive ability to acquire and generalize manipulation “tricks.” Even from a single demonstration, such as using soup ladles to reach for distant objects, we can apply this skill to new scenarios involving different object positions, sizes, and categories (e.g., forks and hammers). Additionally, we can flexibly combine various skills to devise long-term plans. In this paper, we present a framework that enables machines to acquire such manipulation skills, referred to as “mechanisms,” through a single demonstration and self-play. Our key insight lies in interpreting each demonstration as a sequence of changes in robot-object and object-object contact modes, which provides a scaffold for learning detailed samplers for continuous parameters. These learned mechanisms and samplers can be seamlessly integrated into standard task and motion planners, enabling their compositional use.

**摘要:** 人类在获得和推广操纵“技巧”方面表现出令人印象深刻的能力。即使是在单个演示中，例如使用汤勺去够远处的物体，我们也可以将这项技能应用于涉及不同物体位置、大小和类别的新场景（例如，叉子和锤子）。此外，我们还可以灵活地结合各种技能来制定长期计划。在本文中，我们提出了一个框架，该框架使机器能够通过单个演示和自我游戏获得此类操纵技能，称为“机制”。我们的主要见解在于将每个演示解释为机器人与对象和对象与对象接触模式的一系列变化，这为学习连续参数的详细采样器提供了一个框架。这些习得的机制和采样器可以无缝集成到标准任务和运动规划器中，从而实现它们的合成用途。

**[Paper URL](https://proceedings.mlr.press/v229/mao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mao23a/mao23a.pdf)** 

# Sample-Efficient Preference-based Reinforcement Learning with Dynamics Aware Rewards
**题目:** 具有动态感知奖励的样本高效的基于偏好的强化学习

**作者:** Katherine Metcalf, Miguel Sarabia, Natalie Mackraz, Barry-John Theobald

**Abstract:** Preference-based reinforcement learning (PbRL) aligns a robot behavior with human preferences via a reward function learned from binary feedback over agent behaviors. We show that encoding environment dynamics in the reward function improves the sample efficiency of PbRL by an order of magnitude. In our experiments we iterate between: (1) encoding environment dynamics in a state-action representation $z^{sa}$ via a self-supervised temporal consistency task, and (2) bootstrapping the preference-based reward function from $z^{sa}$, which results in faster policy learning and better final policy performance. For example, on quadruped-walk, walker-walk, and cheetah-run, with 50 preference labels we achieve the same performance as existing approaches with 500 preference labels, and we recover $83%$ and $66%$ of ground truth reward policy performance versus only $38%$ and $21%$ without environment dynamics. The performance gains demonstrate that explicitly encoding environment dynamics improves preference-learned reward functions.

**摘要:** 基于偏好的强化学习(PbRL)通过从二进制反馈中学习的奖励函数来使机器人的行为与人类的偏好一致。我们发现，奖励函数中的编码环境动态将PbRL的样本效率提高了一个数量级。在我们的实验中，我们迭代：(1)通过自我监督的时间一致性任务在状态-动作表示$z^{sa}$中编码环境动态，以及(2)从$z^{sa}$引导基于偏好的奖励函数，这导致更快的策略学习和更好的最终策略性能。例如，在四足行走、步行者行走和猎豹奔跑上，使用50个偏好标签，我们获得了与现有方法相同的性能，其中500个偏好标签，我们回收了$83%$和$66%$的基本事实奖励策略性能，而没有环境动力学的情况下，我们只回收了$38%$和$21%$。性能提升表明，显式编码环境动态改善了偏好学习的奖励函数。

**[Paper URL](https://proceedings.mlr.press/v229/metcalf23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/metcalf23a/metcalf23a.pdf)** 

# Im2Contact: Vision-Based Contact Localization Without Touch or Force Sensing
**题目:** Im2 Touch：基于视觉的接触定位，无需触摸或力传感

**作者:** Leon Kim, Yunshuang Li, Michael Posa, Dinesh Jayaraman

**Abstract:** Contacts play a critical role in most manipulation tasks. Robots today mainly use proximal touch/force sensors to sense contacts, but the information they provide must be calibrated and is inherently local, with practical applications relying either on extensive surface coverage or restrictive assumptions to resolve ambiguities. We propose a vision-based extrinsic contact localization task: with only a single RGB-D camera view of a robot workspace, identify when and where an object held by the robot contacts the rest of the environment. We show that careful task-attuned design is critical for a neural network trained in simulation to discover solutions that transfer well to a real robot. Our final approach im2contact demonstrates the promise of versatile general-purpose contact perception from vision alone, performing well for localizing various contact types (point, line, or planar; sticking, sliding, or rolling; single or multiple), and even under occlusions in its camera view. Video results can be found at: https://sites.google.com/view/im2contact/home

**摘要:** 联系人在大多数操作任务中扮演着关键角色。如今的机器人主要使用近端触摸/力传感器来感知接触，但它们提供的信息必须经过校准，并且本质上是局部的，实际应用要么依赖于广泛的表面覆盖，要么依赖于限制性假设来解决歧义。我们提出了一种基于视觉的外部接触定位任务：仅使用机器人工作空间的单个RGB-D摄像机视角，识别机器人持有的对象何时何地与环境的其余部分接触。我们表明，仔细的任务协调设计对于经过模拟训练的神经网络来说是至关重要的，以发现能够很好地转移到真实机器人上的解决方案。我们的最后一种方法im2Contact展示了仅从视觉就可以实现多功能通用接触感知的前景，在定位各种接触类型(点、线或平面；粘滞、滑动或滚动；单或多个)方面表现良好，甚至在相机视图中的遮挡情况下也是如此。视频结果可在以下网址找到：https://sites.google.com/view/im2contact/home

**[Paper URL](https://proceedings.mlr.press/v229/kim23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf)** 

# DROID: Learning from Offline Heterogeneous Demonstrations via Reward-Policy Distillation
**题目:** Droid：通过奖励政策提炼从线下异类演示中学习

**作者:** Sravan Jayanthi, Letian Chen, Nadya Balabanska, Van Duong, Erik Scarlatescu, Ezra Ameperosa, Zulfiqar Haider Zaidi, Daniel Martin, Taylor Keith Del Matto, Masahiro Ono, Matthew Gombolay

**Abstract:** Offline Learning from Demonstrations (OLfD) is valuable in domains where trial-and-error learning is infeasible or specifying a cost function is difficult, such as robotic surgery, autonomous driving, and path-finding for NASA’s Mars rovers. However, two key problems remain challenging in OLfD: 1) heterogeneity: demonstration data can be generated with diverse preferences and strategies, and 2) generalizability: the learned policy and reward must perform well beyond a limited training regime in unseen test settings. To overcome these challenges, we propose Dual Reward and policy Offline Inverse Distillation (DROID), where the key idea is to leverage diversity to improve generalization performance by decomposing common-task and individual-specific strategies and distilling knowledge in both the reward and policy spaces. We ground DROID in a novel and uniquely challenging Mars rover path-planning problem for NASA’s Mars Curiosity Rover. We also curate a novel dataset along 163 Sols (Martian days) and conduct a novel, empirical investigation to characterize heterogeneity in the dataset. We find DROID outperforms prior SOTA OLfD techniques, leading to a $26%$ improvement in modeling expert behaviors and $92%$ closer to the task objective of reaching the final destination. We also benchmark DROID on the OpenAI Gym Cartpole environment and find DROID achieves $55%$ (significantly) better performance modeling heterogeneous demonstrations.

**摘要:** 离线学习演示(OLfD)在试错式学习不可行或很难指定成本函数的领域很有价值，例如机器人手术、自动驾驶和NASA火星漫游车的寻路。然而，在OLFD中有两个关键问题仍然具有挑战性：1)异质性：可以生成具有不同偏好和策略的示范数据；2)概括性：学习到的政策和奖励必须在看不见的测试环境中的有限培训制度下表现得很好。为了克服这些挑战，我们提出了双重奖励和策略离线逆蒸馏(Droid)，其关键思想是通过分解共同任务和特定于个人的策略并在奖励和策略空间中提取知识来利用多样性来提高泛化性能。我们为NASA的火星好奇号火星车设计了一个新颖且具有独特挑战性的火星漫游车路径规划问题。我们还策划了一个沿163个索尔(火星日)的新数据集，并进行了一个新颖的、经验性的调查，以表征数据集中的异质性。我们发现，Droid的性能优于以前的Sota OLfD技术，导致建模专家行为的成本提高了26%$，离到达最终目的地的任务目标更近了92%$。我们还在OpenAI Gym Cartole环境上对Droid进行了基准测试，发现Droid在建模异类演示时的性能提高了55%$(显着)。

**[Paper URL](https://proceedings.mlr.press/v229/jayanthi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/jayanthi23a/jayanthi23a.pdf)** 

# SA6D: Self-Adaptive Few-Shot 6D Pose Estimator for Novel and Occluded Objects
**题目:** SA 6D：新颖和遮挡物体的自适应少镜头6D姿势估计器

**作者:** Ning Gao, Vien Anh Ngo, Hanna Ziesche, Gerhard Neumann

**Abstract:** To enable meaningful robotic manipulation of objects in the real-world, 6D pose estimation is one of the critical aspects. Most existing approaches have difficulties to extend predictions to scenarios where novel object instances are continuously introduced, especially with heavy occlusions. In this work, we propose a few-shot pose estimation (FSPE) approach called SA6D, which uses a self-adaptive segmentation module to identify the novel target object and construct a point cloud model of the target object using only a small number of cluttered reference images. Unlike existing methods, SA6D does not require object-centric reference images or any additional object information, making it a more generalizable and scalable solution across categories. We evaluate SA6D on real-world tabletop object datasets and demonstrate that SA6D outperforms existing FSPE methods, particularly in cluttered scenes with occlusions, while requiring fewer reference images.

**摘要:** 为了在现实世界中实现有意义的机器人操纵对象，6D姿态估计是关键方面之一。大多数现有方法都难以将预测扩展到不断引入新对象实例的场景，尤其是在严重遮挡的情况下。在这项工作中，我们提出了一种名为SA 6D的少镜头姿态估计（FSPE）方法，该方法使用自适应分割模块来识别新型目标对象，并仅使用少量杂乱的参考图像来构建目标对象的点云模型。与现有方法不同，SA 6D不需要以对象为中心的参考图像或任何额外的对象信息，使其成为跨类别更通用和可扩展的解决方案。我们在现实世界桌面对象数据集上评估SA 6D，并证明SA 6D优于现有的FSPE方法，特别是在具有遮挡的混乱场景中，同时需要更少的参考图像。

**[Paper URL](https://proceedings.mlr.press/v229/gao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gao23a/gao23a.pdf)** 

# Hierarchical Planning for Rope Manipulation using Knot Theory and a Learned Inverse Model
**题目:** 使用打结理论和习得逆模型的绳索操纵分层规划

**作者:** Matan Sudry, Tom Jurgenson, Aviv Tamar, Erez Karpas

**Abstract:** This work considers planning the manipulation of deformable 1-dimensional objects, such as ropes or cables, specifically to tie knots. We propose TWISTED: Tying With Inverse model and Search in Topological space Excluding Demos, a hierarchical planning approach which, at the high level, uses ideas from knot-theory to plan a sequence of rope configurations, while at the low level uses a neural-network inverse model to move between the configurations in the high-level plan. To train the neural network, we propose a self-supervised approach, where we learn from random movements of the rope. To focus the random movements on interesting configurations, such as knots, we propose a non-uniform sampling method tailored for this domain. In a simulation, we show that our approach can plan significantly faster and more accurately than baselines. We also show that our plans are robust to parameter changes in the physical simulation, suggesting future applications via sim2real.

**摘要:** 这项工作考虑规划可变形一维物体（例如绳索或电缆）的操纵，专门用于打结。我们建议扭曲：与逆模型捆绑在一起并在布局空间中搜索排除Demos，一种分层规划方法，在高层，使用打结理论的想法来规划绳索配置序列，而在低层，使用神经网络逆模型在高层计划中的配置之间移动。为了训练神经网络，我们提出了一种自我监督的方法，从绳索的随机运动中学习。为了将随机运动集中在有趣的配置（例如结）上，我们提出了一种专为该领域量身定制的非均匀采样方法。在模拟中，我们表明我们的方法可以比基线更快、更准确地进行规划。我们还表明，我们的计划对物理模拟中的参数变化具有鲁棒性，并通过sim 2real建议未来的应用。

**[Paper URL](https://proceedings.mlr.press/v229/sudry23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sudry23a/sudry23a.pdf)** 

# OVIR-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data
**题目:** DVIR-3D：无需3D数据训练的开放词汇3D实例检索

**作者:** Shiyang Lu, Haonan Chang, Eric Pu Jing, Abdeslam Boularias, Kostas Bekris

**Abstract:** This work presents OVIR-3D, a straightforward yet effective method for open-vocabulary 3D object instance retrieval without using any 3D data for training. Given a language query, the proposed method is able to return a ranked set of 3D object instance segments based on the feature similarity of the instance and the text query. This is achieved by a multi-view fusion of text-aligned 2D region proposals into 3D space, where the 2D region proposal network could leverage 2D datasets, which are more accessible and typically larger than 3D datasets. The proposed fusion process is efficient as it can be performed in real-time for most indoor 3D scenes and does not require additional training in 3D space. Experiments on public datasets and a real robot show the effectiveness of the method and its potential for applications in robot navigation and manipulation.

**摘要:** 这项工作提出了DVIR-3D，这是一种简单而有效的开放词汇表3D对象实例检索方法，无需使用任何3D数据进行训练。给定语言查询，所提出的方法能够根据实例和文本查询的特征相似性返回经过排序的3D对象实例片段集。这是通过将文本对齐的2D区域提案多视图融合到3D空间来实现的，其中2D区域提案网络可以利用2D数据集，这些数据集更容易访问，并且通常比3D数据集更大。提出的融合过程非常高效，因为它可以对大多数室内3D场景实时执行，并且不需要在3D空间中进行额外的训练。在公共数据集和真实机器人上的实验表明了该方法的有效性及其在机器人导航和操纵中的应用潜力。

**[Paper URL](https://proceedings.mlr.press/v229/lu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lu23a/lu23a.pdf)** 

# Efficient Sim-to-real Transfer of Contact-Rich Manipulation Skills with Online Admittance Residual Learning
**题目:** 利用在线准入剩余学习实现丰富接触式操纵技能的高效模拟到真实转移

**作者:** Xiang Zhang, Changhao Wang, Lingfeng Sun, Zheng Wu, Xinghao Zhu, Masayoshi Tomizuka

**Abstract:** Learning contact-rich manipulation skills is essential. Such skills require the robots to interact with the environment with feasible manipulation trajectories and suitable compliance control parameters to enable safe and stable contact. However, learning these skills is challenging due to data inefficiency in the real world and the sim-to-real gap in simulation. In this paper, we introduce a hybrid offline-online framework to learn robust manipulation skills. We employ model-free reinforcement learning for the offline phase to obtain the robot motion and compliance control parameters in simulation \RV{with domain randomization}. Subsequently, in the online phase, we learn the residual of the compliance control parameters to maximize robot performance-related criteria with force sensor measurements in real-time. To demonstrate the effectiveness and robustness of our approach, we provide comparative results against existing methods for assembly, pivoting, and screwing tasks.

**摘要:** 学习接触丰富的操纵技能至关重要。这些技能需要机器人通过可行的操纵轨迹和合适的合规控制参数与环境互动，以实现安全稳定的接触。然而，由于现实世界中的数据效率低下以及模拟中的简单与真实差距，学习这些技能具有挑战性。在本文中，我们引入了一个混合的离线-在线框架来学习强大的操纵技能。我们在离线阶段采用无模型强化学习，以在模拟\RV{具有域随机化}中获得机器人运动和顺从性控制参数。随后，在在线阶段，我们学习顺从性控制参数的剩余，以通过力传感器实时测量最大化机器人性能相关标准。为了证明我们方法的有效性和稳健性，我们提供了与现有组装、旋转和拧紧任务方法的比较结果。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23e/zhang23e.pdf)** 

# Tell Me Where to Go: A Composable Framework for Context-Aware Embodied Robot Navigation
**题目:** 告诉我去哪里：上下文感知的机器人导航的可组合框架

**作者:** Harel Biggie, Ajay Narasimha Mopidevi, Dusty Woods, Chris Heckman

**Abstract:** Humans have the remarkable ability to navigate through unfamiliar environments by solely relying on our prior knowledge and descriptions of the environment. For robots to perform the same type of navigation, they need to be able to associate natural language descriptions with their associated physical environment with a limited amount of prior knowledge. Recently, Large Language Models (LLMs) have been able to reason over billions of parameters and utilize them in multi-modal chat-based natural language responses. However, LLMs lack real-world awareness and their outputs are not always predictable. In this work, we develop a low-bandwidth framework that solves this lack of real-world generalization by creating an intermediate layer between an LLM and a robot navigation framework in the form of Python code. Our intermediate shoehorns the vast prior knowledge inherent in an LLM model into a series of input and output API instructions that a mobile robot can understand. We evaluate our method across four different environments and command classes on a mobile robot and highlight our framework’s ability to interpret contextual commands.

**摘要:** 人类仅依靠我们对环境的先验知识和描述，就具有在陌生环境中导航的非凡能力。要让机器人执行相同类型的导航，它们需要能够利用有限的先验知识将自然语言描述与相关的物理环境联系起来。最近，大型语言模型(LLM)已经能够推理超过数十亿个参数，并将它们用于基于多模式聊天的自然语言响应。然而，小岛屿发展中国家缺乏对现实世界的认识，其产出并不总是可预测的。在这项工作中，我们开发了一个低带宽框架，通过在LLM和以Python代码形式的机器人导航框架之间创建一个中间层来解决这种缺乏现实世界泛化的问题。我们的中间任务是将LLM模型中固有的大量先验知识转化为一系列移动机器人可以理解的输入和输出API指令。我们在移动机器人上的四个不同环境和命令类上对我们的方法进行了评估，并强调了我们的框架解释上下文命令的能力。

**[Paper URL](https://proceedings.mlr.press/v229/biggie23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/biggie23a/biggie23a.pdf)** 

# Dynamic Multi-Team Racing: Competitive Driving on 1/10-th Scale Vehicles via Learning in Simulation
**题目:** 动态多车队赛车：通过模拟学习在1/10比例车辆上进行竞技驾驶

**作者:** Peter Werner, Tim Seyde, Paul Drews, Thomas Matrai Balch, Igor Gilitschenski, Wilko Schwarting, Guy Rosman, Sertac Karaman, Daniela Rus

**Abstract:** Autonomous racing is a challenging task that requires vehicle handling at the dynamic limits of friction. While single-agent scenarios like Time Trials are solved competitively with classical model-based or model-free feedback control, multi-agent wheel-to-wheel racing poses several challenges including planning over unknown opponent intentions as well as negotiating interactions under dynamic constraints. We propose to address these challenges via a learning-based approach that effectively combines model-based techniques, massively parallel simulation, and self-play reinforcement learning to enable zero-shot sim-to-real transfer of highly dynamic policies. We deploy our algorithm in wheel-to-wheel multi-agent races on scale hardware to demonstrate the efficacy of our approach. Further details and videos can be found on the project website: https://sites.google.com/view/dynmutr/home.

**摘要:** 自动赛车是一项具有挑战性的任务，需要在摩擦的动态极限下进行车辆操纵。虽然像计时赛这样的单智能体场景是通过经典的基于模型或无模型的反馈控制来竞争性地解决的，但多智能体轮对轮比赛带来了几个挑战，包括对未知对手意图的规划以及在动态约束下协商交互。我们建议通过基于学习的方法来解决这些挑战，该方法有效地结合了基于模型的技术、大规模并行模拟和自玩强化学习，以实现高度动态策略的零触发即时转移。我们在大规模硬件上的轮到轮的多智能体竞赛中部署我们的算法，以证明我们方法的有效性。更多详细信息和视频可在项目网站上找到：https://sites.google.com/view/dynmutr/home。

**[Paper URL](https://proceedings.mlr.press/v229/werner23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/werner23a/werner23a.pdf)** 

# Stochastic Occupancy Grid Map Prediction in Dynamic Scenes
**题目:** 动态场景中随机占用网格地图预测

**作者:** Zhanteng Xie, Philip Dames

**Abstract:** This paper presents two variations of a novel stochastic prediction algorithm that enables mobile robots to accurately and robustly predict the future state of complex dynamic scenes. The proposed algorithm uses a variational autoencoder to predict a range of possible future states of the environment. The algorithm takes full advantage of the motion of the robot itself, the motion of dynamic objects, and the geometry of static objects in the scene to improve prediction accuracy. Three simulated and real-world datasets collected by different robot models are used to demonstrate that the proposed algorithm is able to achieve more accurate and robust prediction performance than other prediction algorithms. Furthermore, a predictive uncertainty-aware planner is proposed to demonstrate the effectiveness of the proposed predictor in simulation and real-world navigation experiments. Implementations are open source at https://github.com/TempleRAIL/SOGMP.

**摘要:** 本文提出了一种新型随机预测算法的两种变体，使移动机器人能够准确、稳健地预测复杂动态场景的未来状态。提出的算法使用变分自动编码器来预测环境的一系列可能的未来状态。该算法充分利用机器人本身的运动、动态物体的运动以及场景中静态物体的几何形状来提高预测准确性。使用不同机器人模型收集的三个模拟和现实世界数据集来证明所提出的算法能够实现比其他预测算法更准确和更稳健的预测性能。此外，提出了一种预测性不确定性感知规划器，以证明所提出的预测器在模拟和现实世界导航实验中的有效性。实现在https://github.com/TempleRAIL/SOGMP上开源。

**[Paper URL](https://proceedings.mlr.press/v229/xie23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xie23a/xie23a.pdf)** 

# A Bayesian approach to breaking things: efficiently predicting and repairing failure modes via sampling
**题目:** 打破事物的Bayesian方法：通过抽样有效预测和修复故障模式

**作者:** Charles Dawson, Chuchu Fan

**Abstract:** Before autonomous systems can be deployed in safety-critical applications, we must be able to understand and verify the safety of these systems. For cases where the risk or cost of real-world testing is prohibitive, we propose a simulation-based framework for a) predicting ways in which an autonomous system is likely to fail and b) automatically adjusting the system’s design to preemptively mitigate those failures. We frame this problem through the lens of approximate Bayesian inference and use differentiable simulation for efficient failure case prediction and repair. We apply our approach on a range of robotics and control problems, including optimizing search patterns for robot swarms and reducing the severity of outages in power transmission networks. Compared to optimization-based falsification techniques, our method predicts a more diverse, representative set of failure modes, and we also find that our use of differentiable simulation yields solutions that have up to 10x lower cost and requires up to 2x fewer iterations to converge relative to gradient-free techniques.

**摘要:** 在自主系统可以部署到安全关键型应用程序之前，我们必须能够了解和验证这些系统的安全性。对于真实世界测试的风险或成本令人望而却步的情况，我们提出了一个基于模拟的框架，用于a)预测自治系统可能失败的方式，以及b)自动调整系统的设计以抢先减轻这些失败。我们通过近似贝叶斯推理的镜头来框架这一问题，并使用可微模拟来有效地预测和修复故障案例。我们将我们的方法应用于一系列机器人和控制问题，包括优化机器人群的搜索模式和降低电力传输网络中的停电严重程度。与基于优化的证伪技术相比，我们的方法预测了一组更多样化、更具代表性的故障模式，并且我们还发现，使用可微模拟产生的解决方案与无梯度技术相比，成本降低了多达10倍，迭代收敛所需的迭代次数减少了2倍。

**[Paper URL](https://proceedings.mlr.press/v229/dawson23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/dawson23a/dawson23a.pdf)** 

# BridgeData V2: A Dataset for Robot Learning at Scale
**题目:** BridgeData V2：用于机器人大规模学习的数据集

**作者:** Homer Rich Walke, Kevin Black, Tony Z. Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, Abraham Lee, Kuan Fang, Chelsea Finn, Sergey Levine

**Abstract:** We introduce BridgeData V2, a large and diverse dataset of robotic manipulation behaviors designed to facilitate research in scalable robot learning. BridgeData V2 contains 53,896 trajectories collected across 24 environments on a publicly available low-cost robot. Unlike many existing robotic manipulation datasets, BridgeData V2 provides enough task and environment variability that skills learned from the data generalize across institutions, making the dataset a useful resource for a broad range of researchers. Additionally, the dataset is compatible with a wide variety of open-vocabulary, multi-task learning methods conditioned on goal images or natural language instructions. In our experiments,we apply 6 state-of-the-art imitation learning and offline reinforcement learning methods to the data and find that they succeed on a suite of tasks requiring varying amounts of generalization. We also demonstrate that the performance of these methods improves with more data and higher capacity models. By publicly sharing BridgeData V2 and our pre-trained models, we aim to accelerate research in scalable robot learning methods.

**摘要:** 我们介绍了BridgeData V2，这是一个关于机器人操作行为的大型且多样化的数据集，旨在促进可扩展机器人学习的研究。BridgeData V2包含在24个环境中收集的53,896个轨迹，这些轨迹是在一个公开可用的低成本机器人上收集的。与许多现有的机器人操作数据集不同，BridgeData V2提供了足够的任务和环境可变性，从数据中学到的技能可以在不同的机构中推广，使数据集成为广泛研究人员的有用资源。此外，该数据集与多种开放词汇、多任务学习方法兼容，这些方法以目标图像或自然语言指令为条件。在我们的实验中，我们将6种最先进的模仿学习和离线强化学习方法应用到数据中，发现它们在一组需要不同泛化程度的任务上取得了成功。我们还证明了这些方法的性能随着更多的数据和更高容量的模型而提高。通过公开分享BridgeData V2和我们预先训练的模型，我们的目标是加速可扩展机器人学习方法的研究。

**[Paper URL](https://proceedings.mlr.press/v229/walke23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/walke23a/walke23a.pdf)** 

# NOIR: Neural Signal Operated Intelligent Robots for Everyday Activities
**题目:** NOIR：用于日常活动的神经信号操作智能机器人

**作者:** Ruohan Zhang, Sharon Lee, Minjune Hwang, Ayano Hiranaka, Chen Wang, Wensi Ai, Jin Jie Ryan Tan, Shreya Gupta, Yilun Hao, Gabrael Levine, Ruohan Gao, Anthony Norcia, Li Fei-Fei, Jiajun Wu

**Abstract:** We present Neural Signal Operated Intelligent Robots (NOIR), a general-purpose, intelligent brain-robot interface system that enables humans to command robots to perform everyday activities through brain signals. Through this interface, humans communicate their intended objects of interest and actions to the robots using electroencephalography (EEG). Our novel system demonstrates success in an expansive array of 20 challenging, everyday household activities, including cooking, cleaning, personal care, and entertainment. The effectiveness of the system is improved by its synergistic integration of robot learning algorithms, allowing for NOIR to adapt to individual users and predict their intentions. Our work enhances the way humans interact with robots, replacing traditional channels of interaction with direct, neural communication.

**摘要:** 我们介绍了神经信号操作智能机器人（NOIR），这是一种通用的智能脑机器人接口系统，使人类能够通过大脑信号命令机器人执行日常活动。通过这个接口，人类使用脑电波（EEG）将他们感兴趣的预期对象和动作传达给机器人。我们的新颖系统在20项具有挑战性的日常家庭活动中取得了成功，包括烹饪、清洁、个人护理和娱乐。该系统的有效性通过机器人学习算法的协同集成而得到提高，使NOIR能够适应个人用户并预测他们的意图。我们的工作增强了人类与机器人互动的方式，用直接的神经通信取代传统的互动渠道。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23f.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23f/zhang23f.pdf)** 

# PolarNet: 3D Point Clouds for Language-Guided Robotic Manipulation
**题目:** PolarNet：用于图形引导机器人操纵的3D点云

**作者:** Shizhe Chen, Ricardo Garcia Pinel, Cordelia Schmid, Ivan Laptev

**Abstract:** The ability for robots to comprehend and execute manipulation tasks based on natural language instructions is a long-term goal in robotics. The dominant approaches for language-guided manipulation use 2D image representations, which face difficulties in combining multi-view cameras and inferring precise 3D positions and relationships. To address these limitations, we propose a 3D point cloud based policy called PolarNet for language-guided manipulation. It leverages carefully designed point cloud inputs, efficient point cloud encoders, and multimodal transformers to learn 3D point cloud representations and integrate them with language instructions for action prediction. PolarNet is shown to be effective and data efficient in a variety of experiments conducted on the RLBench benchmark. It outperforms state-of-the-art 2D and 3D approaches in both single-task and multi-task learning. It also achieves promising results on a real robot.

**摘要:** 机器人能够理解和执行基于自然语言指令的操纵任务是机器人技术的长期目标。语言引导操纵的主要方法使用2D图像表示，这在组合多视图相机和推断精确的3D位置和关系方面面临困难。为了解决这些限制，我们提出了一种名为PolarNet的基于3D点云的策略，用于语言引导的操纵。它利用精心设计的点云输入、高效的点云编码器和多模式转换器来学习3D点云表示，并将其与语言指令集成以进行动作预测。在对RL Bench基准进行的各种实验中，PolarNet被证明有效且数据高效。它在单任务和多任务学习方面都优于最先进的2D和3D方法。它还在真实机器人上取得了令人鼓舞的结果。

**[Paper URL](https://proceedings.mlr.press/v229/chen23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23b/chen23b.pdf)** 

# Stealthy Terrain-Aware Multi-Agent Active Search
**题目:** 隐形地形感知多智能体主动搜索

**作者:** Nikhil Angad Bakshi, Jeff Schneider

**Abstract:** Stealthy multi-agent active search is the problem of making efficient sequential data-collection decisions to identify an unknown number of sparsely located targets while adapting to new sensing information and concealing the search agents’ location from the targets. This problem is applicable to reconnaissance tasks wherein the safety of the search agents can be compromised as the targets may be adversarial. Prior work usually focuses either on adversarial search, where the risk of revealing the agents’ location to the targets is ignored or evasion strategies where efficient search is ignored. We present the Stealthy Terrain-Aware Reconnaissance (STAR) algorithm, a multi-objective parallelized Thompson sampling-based algorithm that relies on a strong topographical prior to reason over changing visibility risk over the course of the search. The STAR algorithm outperforms existing state-of-the-art multi-agent active search methods on both rate of recovery of targets as well as minimising risk even when subject to noisy observations, communication failures and an unknown number of targets.

**摘要:** 隐身多智能体主动搜索是指在适应新的感知信息并对搜索智能体的位置进行隐藏的同时，做出有效的顺序数据收集决策以识别未知数量的稀疏目标的问题。这个问题适用于侦察任务，其中搜索代理的安全可能会受到损害，因为目标可能是对抗性的。以前的工作通常集中在对抗性搜索，其中向目标泄露代理位置的风险被忽略，或者逃避策略，其中有效搜索被忽略。提出了隐身地形感知侦察算法(STAR)，这是一种基于多目标并行Thompson采样的算法，它依赖于强大的地形先于搜索过程中能见度风险的变化进行推理。STAR算法在目标恢复速度和风险最小化方面都优于现有的最先进的多智能体主动搜索方法，即使在受到噪声观测、通信故障和目标数量未知的情况下也是如此。

**[Paper URL](https://proceedings.mlr.press/v229/bakshi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/bakshi23a/bakshi23a.pdf)** 

# A Data-Efficient Visual-Audio Representation with Intuitive Fine-tuning for Voice-Controlled Robots
**题目:** 具有直观微调的语音控制机器人数据高效的视听表示

**作者:** Peixin Chang, Shuijing Liu, Tianchen Ji, Neeloy Chakraborty, Kaiwen Hong, Katherine Rose Driggs-Campbell

**Abstract:** A command-following robot that serves people in everyday life must continually improve itself in deployment domains with minimal help from its end users, instead of engineers. Previous methods are either difficult to continuously improve after the deployment or require a large number of new labels during fine-tuning. Motivated by (self-)supervised contrastive learning, we propose a novel representation that generates an intrinsic reward function for command-following robot tasks by associating images with sound commands. After the robot is deployed in a new domain, the representation can be updated intuitively and data-efficiently by non-experts without any hand-crafted reward functions. We demonstrate our approach on various sound types and robotic tasks, including navigation and manipulation with raw sensor inputs. In simulated and real-world experiments, we show that our system can continually self-improve in previously unseen scenarios given fewer new labeled data, while still achieving better performance over previous methods.

**摘要:** 一个在日常生活中为人们服务的跟随命令的机器人必须在最终用户而不是工程师的最小帮助下，在部署领域不断改进自己。以前的方法要么在部署后难以持续改进，要么在微调过程中需要大量的新标签。受(自)监督对比学习的启发，我们提出了一种新的表示方法，它通过将图像和声音命令相关联来生成命令跟随机器人任务的内在奖励函数。当机器人被部署到一个新的领域后，非专家可以直观地、数据高效地更新表示，而不需要任何手工制作的奖励函数。我们在各种声音类型和机器人任务中演示了我们的方法，包括导航和使用原始传感器输入的操作。在模拟和真实世界的实验中，我们的系统可以在较少的新标签数据的情况下，在以前未见的场景中不断地自我改进，同时仍然取得了比以前方法更好的性能。

**[Paper URL](https://proceedings.mlr.press/v229/chang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chang23a/chang23a.pdf)** 

# MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations
**题目:** MimicGen：使用人类演示的可扩展机器人学习的数据生成系统

**作者:** Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox

**Abstract:** Imitation learning from a large set of human demonstrations has proved to be an effective paradigm for building capable robot agents. However, the demonstrations can be extremely costly and time-consuming to collect. We introduce MimicGen, a system for automatically synthesizing large-scale, rich datasets from only a small number of human demonstrations by adapting them to new contexts. We use MimicGen to generate over 50K demonstrations across 18 tasks with diverse scene configurations, object instances, and robot arms from just  200 human demonstrations. We show that robot agents can be effectively trained on this generated dataset by imitation learning to achieve strong performance in long-horizon and high-precision tasks, such as multi-part assembly and coffee preparation, across broad initial state distributions. We further demonstrate that the effectiveness and utility of MimicGen data compare favorably to collecting additional human demonstrations, making it a powerful and economical approach towards scaling up robot learning. Datasets, simulation environments, videos, and more at https://mimicgen.github.io.

**摘要:** 事实证明，从大量的人类演示中进行模仿学习是建立有能力的机器人代理的有效范例。然而，收集演示可能非常昂贵和耗时。我们介绍了MimicGen，这是一个通过使大规模、丰富的数据集适应新的上下文来自动合成大规模、丰富的数据集的系统。我们使用MimicGen在18个任务中生成超过50K个演示，其中包含不同的场景配置、对象实例和机器人手臂，这些演示来自仅200个人类演示。我们表明，机器人智能体可以在这个生成的数据集上通过模仿学习得到有效的训练，从而在跨越广泛的初始状态分布的长时间和高精度任务中获得强大的性能，例如多部件装配和咖啡准备。我们进一步证明，与收集更多的人类演示相比，MimicGen数据的有效性和实用性更好，使其成为扩大机器人学习的一种强大而经济的方法。Https://mimicgen.github.io.上的数据集、模拟环境、视频等

**[Paper URL](https://proceedings.mlr.press/v229/mandlekar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mandlekar23a/mandlekar23a.pdf)** 

# Quantifying Assistive Robustness Via the Natural-Adversarial Frontier
**题目:** 通过自然对抗边界量化辅助稳健性

**作者:** Jerry Zhi-Yang He, Daniel S. Brown, Zackory Erickson, Anca Dragan

**Abstract:** Our ultimate goal is to build robust policies for robots that assist people. What makes this hard is that people can behave unexpectedly at test time, potentially interacting with the robot outside its training distribution and leading to failures. Even just measuring robustness is a challenge. Adversarial perturbations are the default, but they can paint the wrong picture: they can correspond to human motions that are unlikely to occur during natural interactions with people. A robot policy might fail under small adversarial perturbations but work under large natural perturbations. We propose that capturing robustness in these interactive settings requires constructing and analyzing the entire natural-adversarial frontier: the Pareto-frontier of human policies that are the best trade-offs between naturalness and low robot performance. We introduce RIGID, a method for constructing this frontier by training adversarial human policies that trade off between minimizing robot reward and acting human-like (as measured by a discriminator). On an Assistive Gym task, we use RIGID to analyze the performance of standard collaborative RL, as well as the performance of existing methods meant to increase robustness. We also compare the frontier RIGID identifies with the failures identified in expert adversarial interaction, and with naturally-occurring failures during user interaction. Overall, we find evidence that RIGID can provide a meaningful measure of robustness predictive of deployment performance, and uncover failure cases that are difficult to find manually.

**摘要:** 我们的最终目标是为机器人制定强有力的政策，帮助人类。使这一点变得困难的是，人们在测试时可能会出现意外行为，可能会与机器人在其训练分布之外进行交互，并导致失败。即使只是衡量健壮性也是一项挑战。对抗性的扰动是默认的，但它们可能描绘了错误的图景：它们可能对应于在与人的自然互动中不太可能发生的人类运动。机器人策略可能在小的对抗性扰动下失败，但在大的自然扰动下工作。我们认为，在这些交互环境中捕获健壮性需要构建和分析整个自然-对抗性前沿：人类策略的帕累托前沿，它是自然度和低机器人性能之间的最佳权衡。我们引入了刚性，一种通过训练对抗性人类策略来构建这一前沿的方法，该策略在最小化机器人奖励和表现得像人类(由鉴别器衡量)之间进行权衡。在辅助健身房任务上，我们使用Rigid来分析标准协作RL的性能，以及现有方法的性能，以增加健壮性。我们还将前沿刚性识别与专家对抗性交互中识别的失败进行比较，并与用户交互过程中自然发生的失败进行比较。总体而言，我们发现有证据表明，刚性可以提供部署性能预测的健壮性的有意义的衡量标准，并发现手动难以发现的故障案例。

**[Paper URL](https://proceedings.mlr.press/v229/he23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/he23a/he23a.pdf)** 

# Dynamic Handover: Throw and Catch with Bimanual Hands
**题目:** 动态交接：双手投掷和接球

**作者:** Binghao Huang, Yuanpei Chen, Tianyu Wang, Yuzhe Qin, Yaodong Yang, Nikolay Atanasov, Xiaolong Wang

**Abstract:** Humans throw and catch objects all the time. However, such a seemingly common skill introduces a lot of challenges for robots to achieve: The robots need to operate such dynamic actions at high-speed, collaborate precisely, and interact with diverse objects. In this paper, we design a system with two multi-finger hands attached to robot arms to solve this problem. We train our system using Multi-Agent Reinforcement Learning in simulation and perform Sim2Real transfer to deploy on the real robots. To overcome the Sim2Real gap, we provide multiple novel algorithm designs including learning a trajectory prediction model for the object. Such a model can help the robot catcher has a real-time estimation of where the object will be heading, and then react accordingly. We conduct our experiments with multiple objects in the real-world system, and show significant improvements over multiple baselines. Our project page is available at https://binghao-huang.github.io/dynamic_handover/

**摘要:** 人类一直在投掷和捕捉物体。然而，这种看似常见的技能给机器人带来了很多挑战：机器人需要高速操作这种动态动作、精确协作并与不同对象互动。本文中，我们设计了一个将两个多指手连接到机器人手臂上的系统来解决这个问题。我们在模拟中使用多智能体强化学习来训练我们的系统，并执行Sim2Real传输以部署在真实机器人上。为了克服Sim2Real差距，我们提供了多种新颖的算法设计，包括学习对象的轨迹预测模型。这样的模型可以帮助机器人捕捉器实时估计物体的前进方向，然后做出相应的反应。我们在现实世界系统中对多个对象进行了实验，并在多个基线上显示出显着的改进。我们的项目页面可访问https://binghao-huang.github.io/dynamic_handover/

**[Paper URL](https://proceedings.mlr.press/v229/huang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/huang23d/huang23d.pdf)** 

# Cross-Dataset Sensor Alignment: Making Visual 3D Object Detector Generalizable
**题目:** 跨数据集传感器对齐：使视觉3D对象检测器可通用化

**作者:** Liangtao Zheng, Yicheng Liu, Yue Wang, Hang Zhao

**Abstract:** While camera-based 3D object detection has evolved rapidly, these models are susceptible to overfitting to specific sensor setups. For example, in autonomous driving, most datasets are collected using a single sensor configuration. This paper evaluates the generalization capability of camera-based 3D object detectors, including adapting detectors from one dataset to another and training detectors with multiple datasets. We observe that merely aggregating datasets yields drastic performance drops, contrary to the expected improvements associated with increased training data. To close the gap, we introduce an efficient technique for aligning disparate sensor configurations — a combination of camera intrinsic synchronization, camera extrinsic correction, and ego frame alignment, which collectively enhance cross-dataset performance remarkably. Compared with single dataset baselines, we achieve 42.3 mAP improvement on KITTI, 23.2 mAP improvement on Lyft, 18.5 mAP improvement on nuScenes, 17.3 mAP improvement on KITTI-360, 8.4 mAP improvement on Argoverse2 and 3.9 mAP improvement on Waymo. We hope this comprehensive study can facilitate research on generalizable 3D object detection and associated tasks.

**摘要:** 虽然基于摄像头的3D对象检测已经迅速发展，但这些模型很容易过度适应特定的传感器设置。例如，在自动驾驶中，大多数数据集是使用单一传感器配置收集的。本文评估了基于摄像机的三维目标检测器的泛化能力，包括从一个数据集到另一个数据集的检测器自适应和用多个数据集训练检测器。我们观察到，仅仅聚集数据集会导致性能急剧下降，这与增加训练数据所带来的预期改善相反。为了缩小这一差距，我们引入了一种有效的技术来对齐不同的传感器配置-相机内部同步、相机外部校正和EGO帧对准的组合，这些技术共同显著提高了跨数据集性能。与单一数据集基线相比，Kitti改进了42.3幅地图，Lyft改进了23.2幅地图，nuScenes改进了18.5幅地图，Kitti-360改进了17.3幅地图，Argoverse2改进了8.4幅地图，Waymo改进了3.9幅地图。我们希望这一综合性的研究能够促进可推广的3D目标检测及其相关任务的研究。

**[Paper URL](https://proceedings.mlr.press/v229/zheng23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zheng23a/zheng23a.pdf)** 

# REBOOT: Reuse Data for Bootstrapping Efficient Real-World Dexterous Manipulation
**题目:** REBOOT：重用数据以引导有效的现实世界灵巧操纵

**作者:** Zheyuan Hu, Aaron Rovinsky, Jianlan Luo, Vikash Kumar, Abhishek Gupta, Sergey Levine

**Abstract:** Dexterous manipulation tasks involving contact-rich interactions pose a significant challenge for both model-based control systems and imitation learning algorithms. The complexity arises from the need for multi-fingered robotic hands to dynamically establish and break contacts, balance forces on the non-prehensile object, and control a high number of degrees of freedom. Reinforcement learning (RL) offers a promising approach due to its general applicability and capacity to autonomously acquire optimal manipulation strategies. However, its real-world application is often hindered by the necessity to generate a large number of samples, reset the environment, and obtain reward signals. In this work, we introduce an efficient system for learning dexterous manipulation skills with RL to alleviate these challenges. The main idea of our approach is the integration of recent advancements in sample-efficient RL and replay buffer bootstrapping. This unique combination allows us to utilize data from different tasks or objects as a starting point for training new tasks, significantly improving learning efficiency. Additionally, our system completes the real-world training cycle by incorporating learned resets via an imitation-based pickup policy and learned reward functions, to eliminate the need for manual reset and reward engineering. We show the benefits of reusing past data as replay buffer initialization for new tasks, for instance, the fast acquisitions of intricate manipulation skills in the real world on a four-fingered robotic hand. https://sites.google.com/view/reboot-dexterous

**摘要:** 涉及大量接触交互的灵活操作任务对基于模型的控制系统和模拟学习算法都是一个巨大的挑战。这种复杂性源于需要多指机械手来动态地建立和断开接触、平衡不可抓握物体上的力以及控制大量的自由度。强化学习(RL)由于其广泛的适用性和自主获取最优操作策略的能力，提供了一种很有前途的方法。然而，由于需要生成大量样本、重置环境和获取奖励信号，它在现实世界中的应用经常受到阻碍。在这项工作中，我们介绍了一个有效的系统来学习灵活的操作技能与RL来缓解这些挑战。我们的方法的主要思想是集成了样本效率RL和重放缓冲区自举方面的最新进展。这种独特的组合允许我们利用来自不同任务或对象的数据作为训练新任务的起点，显著提高学习效率。此外，我们的系统通过基于模仿的拾取策略和学习奖励功能整合了学习重置，从而完成了真实世界的训练周期，从而消除了手动重置和奖励工程的需要。我们展示了重复使用过去的数据作为新任务的重放缓冲区初始化的好处，例如，在四指机械手上快速获得现实世界中复杂的操作技能。Https://sites.google.com/view/reboot-dexterous

**[Paper URL](https://proceedings.mlr.press/v229/hu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/hu23a/hu23a.pdf)** 

# Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs
**题目:** 具有开放词汇3D场景图的上下文感知实体基础

**作者:** Haonan Chang, Kowndinya Boyalakuntla, Shiyang Lu, Siwei Cai, Eric Pu Jing, Shreesh Keskar, Shijie Geng, Adeeb Abbas, Lifeng Zhou, Kostas Bekris, Abdeslam Boularias

**Abstract:** We present an Open-Vocabulary 3D Scene Graph (OVSG), a formal framework for grounding a variety of entities, such as object instances, agents, and regions, with free-form text-based queries. Unlike conventional semantic-based object localization approaches, our system facilitates context-aware entity localization, allowing for queries such as “pick up a cup on a kitchen table" or “navigate to a sofa on which someone is sitting". In contrast to existing research on 3D scene graphs, OVSG supports free-form text input and open-vocabulary querying. Through a series of comparative experiments using the ScanNet dataset and a self-collected dataset, we demonstrate that our proposed approach significantly surpasses the performance of previous semantic-based localization techniques. Moreover, we highlight the practical application of OVSG in real-world robot navigation and manipulation experiments. The code and dataset used for evaluation will be made available upon publication.

**摘要:** 我们提出了一个开放词汇3D场景图（DVSG），这是一个形式框架，用于通过自由形式的基于文本的查询来建立各种实体（例如对象实例、代理和区域）的基础。与传统的基于语义的对象定位方法不同，我们的系统促进了上下文感知实体定位，允许诸如“拿起厨房桌子上的杯子”或“导航到有人坐的沙发”等查询。与现有的3D场景图研究相反，DVSG支持自由形式的文本输入和开放词汇查询。通过使用ScanNet数据集和自收集数据集的一系列比较实验，我们证明我们提出的方法显着优于之前基于语义的定位技术的性能。此外，我们还强调了DVSG在现实世界机器人导航和操纵实验中的实际应用。用于评估的代码和数据集将在发布后提供。

**[Paper URL](https://proceedings.mlr.press/v229/chang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chang23b/chang23b.pdf)** 

# HomeRobot: Open-Vocabulary Mobile Manipulation
**题目:** HomeRobot：开放词汇移动操纵

**作者:** Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin S. Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander Clegg, John M. Turner, Zsolt Kira, Manolis Savva, Angel X. Chang, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, Chris Paxton

**Abstract:** HomeRobot (noun): An affordable compliant robot that navigates homes and manipulates a wide range of objects in order to complete everyday tasks. Open-Vocabulary Mobile Manipulation (OVMM) is the problem of picking any object in any unseen environment, and placing it in a commanded location. This is a foundational challenge for robots to be useful assistants in human environments, because it involves tackling sub-problems from across robotics: perception, language understanding, navigation, and manipulation are all essential to OVMM. In addition, integration of the solutions to these sub-problems poses its own substantial challenges. To drive research in this area, we introduce the HomeRobot OVMM benchmark, where an agent navigates household environments to grasp novel objects and place them on target receptacles. HomeRobot has two components: a simulation component, which uses a large and diverse curated object set in new, high-quality multi-room home environments; and a real-world component, providing a software stack for the low-cost Hello Robot Stretch to encourage replication of real-world experiments across labs. We implement both reinforcement learning and heuristic (model-based) baselines and show evidence of sim-to-real transfer. Our baselines achieve a $20%$ success rate in the real world; our experiments identify ways future research work improve performance. See videos on our website: https://home-robot-ovmm.github.io/.

**摘要:** HomeRobot(名词)：一种负担得起的、合规的机器人，它能在家里导航，并操纵各种物体，以完成日常任务。开放式词汇移动操作(OVMM)是在任何看不见的环境中挑选任何对象，并将其放置在命令位置的问题。这是机器人在人类环境中成为有用助手的一个基础性挑战，因为它涉及处理来自机器人学的子问题：感知、语言理解、导航和操作都是OVMM的关键。此外，将这些子问题的解决办法结合起来本身也构成了重大挑战。为了推动这一领域的研究，我们引入了HomeRobot OVMM基准，在该基准中，代理导航家庭环境以抓住新对象并将它们放置在目标容器上。HomeRobot有两个组件：一个模拟组件，它在新的、高质量的多房间家庭环境中使用一个大型且多样化的精选对象集；以及一个真实组件，它为低成本的Hello Robot延伸提供一个软件堆栈，以鼓励在实验室之间复制真实世界的实验。我们实现了强化学习和启发式(基于模型)基线，并展示了从模拟到真实的迁移证据。我们的基准在现实世界中实现了$20%$的成功率；我们的实验确定了未来研究工作改进性能的方法。在我们的网站上观看视频：https://home-robot-ovmm.github.io/.

**[Paper URL](https://proceedings.mlr.press/v229/yenamandra23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yenamandra23a/yenamandra23a.pdf)** 

# PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play
**题目:** PlayFusion：通过从注释游戏中传播来获取技能

**作者:** Lili Chen, Shikhar Bahl, Deepak Pathak

**Abstract:** Learning from unstructured and uncurated data has become the dominant paradigm for generative approaches in language or vision. Such unstructured and unguided behavior data, commonly known as play, is also easier to collect in robotics but much more difficult to learn from due to its inherently multimodal, noisy, and suboptimal nature. In this paper, we study this problem of learning goal-directed skill policies from unstructured play data which is labeled with language in hindsight. Specifically, we leverage advances in diffusion models to learn a multi-task diffusion model to extract robotic skills from play data. Using a conditional denoising diffusion process in the space of states and actions, we can gracefully handle the complexity and multimodality of play data and generate diverse and interesting robot behaviors. To make diffusion models more useful for skill learning, we encourage robotic agents to acquire a vocabulary of skills by introducing discrete bottlenecks into the conditional behavior generation process. In our experiments, we demonstrate the effectiveness of our approach across a wide variety of environments in both simulation and the real world. Video results available at https://play-fusion.github.io.

**摘要:** 从非结构化和非精选数据中学习已经成为语言或视觉中生成方法的主导范例。这种无结构和无指导的行为数据，通常被称为游戏，在机器人中也更容易收集，但由于其固有的多模式、噪声和次优性质，学习起来要困难得多。本文研究了从事后带有语言标签的非结构化游戏数据中学习目标导向技能策略的问题。具体地说，我们利用扩散模型的进步来学习多任务扩散模型，以从游戏数据中提取机器人技能。利用状态空间和动作空间中的条件去噪扩散过程，可以很好地处理游戏数据的复杂性和多发性，并产生多样化和有趣的机器人行为。为了使扩散模型对技能学习更有用，我们鼓励机器人代理通过在条件行为生成过程中引入离散瓶颈来获取技能词汇。在我们的实验中，我们展示了我们的方法在模拟和真实世界中的各种环境中的有效性。视频结果可在https://play-fusion.github.io.上查看

**[Paper URL](https://proceedings.mlr.press/v229/chen23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23c/chen23c.pdf)** 

# Shelving, Stacking, Hanging: Relational Pose Diffusion for Multi-modal Rearrangement
**题目:** 搁置、堆叠、悬挂：多模式重新排列的关系位姿扩散

**作者:** Anthony Simeonov, Ankit Goyal, Lucas Manuelli, Yen-Chen Lin, Alina Sarmiento, Alberto Rodriguez Garcia, Pulkit Agrawal, Dieter Fox

**Abstract:** We propose a system for rearranging objects in a scene to achieve a desired object-scene placing relationship, such as a book inserted in an open slot of a bookshelf. The pipeline generalizes to novel geometries, poses, and layouts of both scenes and objects, and is trained from demonstrations to operate directly on 3D point clouds. Our system overcomes challenges associated with the existence of many geometrically-similar rearrangement solutions for a given scene. By leveraging an iterative pose de-noising training procedure, we can fit multi-modal demonstration data and produce multi-modal outputs while remaining precise and accurate. We also show the advantages of conditioning on relevant local geometric features while ignoring irrelevant global structure that harms both generalization and precision. We demonstrate our approach on three distinct rearrangement tasks that require handling multi-modality and generalization over object shape and pose in both simulation and the real world. Project website, code, and videos: https://anthonysimeonov.github.io/rpdiff-multi-modal

**摘要:** 我们提出了一种系统，用于重新排列场景中的对象以实现期望的对象-场景放置关系，例如将一本书插入书架的开放槽中。该管道概括为场景和对象的新几何图形、姿势和布局，并通过演示训练直接在3D点云上操作。我们的系统克服了与给定场景的许多几何相似的重排解决方案的存在相关的挑战。通过利用迭代的姿势去噪训练过程，我们可以拟合多模式演示数据并产生多模式输出，同时保持精确和准确。我们还展示了条件相关局部几何特征的优势，而忽略了不相关的全局结构，这既损害了泛化，又损害了精度。我们在三个不同的重排任务上演示了我们的方法，这三个任务需要在仿真和真实世界中处理对象形状和姿势的多模态和泛化。项目网站、代码和视频：https://anthonysimeonov.github.io/rpdiff-multi-modal

**[Paper URL](https://proceedings.mlr.press/v229/simeonov23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/simeonov23a/simeonov23a.pdf)** 

# Learning Efficient Abstract Planning Models that Choose What to Predict
**题目:** 学习选择预测内容的高效抽象规划模型

**作者:** Nishanth Kumar, Willie McClinton, Rohan Chitnis, Tom Silver, Tomás Lozano-Pérez, Leslie Pack Kaelbling

**Abstract:** An effective approach to solving long-horizon tasks in robotics domains with continuous state and action spaces is bilevel planning, wherein a high-level search over an abstraction of an environment is used to guide low-level decision-making. Recent work has shown how to enable such bilevel planning by learning abstract models in the form of symbolic operators and neural samplers. In this work, we show that existing symbolic operator learning approaches fall short in many robotics domains where a robot’s actions tend to cause a large number of irrelevant changes in the abstract state. This is primarily because they attempt to learn operators that exactly predict all observed changes in the abstract state. To overcome this issue, we propose to learn operators that ‘choose what to predict’ by only modelling changes necessary for abstract planning to achieve specified goals. Experimentally, we show that our approach learns operators that lead to efficient planning across 10 different hybrid robotics domains, including 4 from the challenging BEHAVIOR-100 benchmark, while generalizing to novel initial states, goals, and objects.

**摘要:** 在具有连续状态和动作空间的机器人领域中，解决长时间任务的一种有效方法是双层规划，其中使用对环境抽象的高层搜索来指导低层决策。最近的工作表明，如何通过学习符号运算符和神经采样器形式的抽象模型来实现这种双层规划。在这项工作中，我们证明了现有的符号算子学习方法在许多机器人领域中的不足，在这些领域中，机器人的动作往往会导致抽象状态的大量无关变化。这主要是因为他们试图学习准确预测抽象状态中所有观察到的变化的运算符。为了解决这个问题，我们建议学习运算符，通过只对抽象规划实现特定目标所需的更改进行建模，来“选择要预测的内容”。实验表明，我们的方法学习了一些算子，这些算子能够在10个不同的混合机器人领域进行有效的规划，其中包括4个来自具有挑战性的行为-100基准的领域，同时概括为新的初始状态、目标和对象。

**[Paper URL](https://proceedings.mlr.press/v229/kumar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kumar23a/kumar23a.pdf)** 

# DYNAMO-GRASP: DYNAMics-aware Optimization for GRASP Point Detection in Suction Grippers
**题目:** DYNAMO-GRASP：抽吸夹持器中GRASP点检测的动态感知优化

**作者:** Boling Yang, Soofiyan Atar, Markus Grotz, Byron Boots, Joshua Smith

**Abstract:** In this research, we introduce a novel approach to the challenge of suction grasp point detection. Our method, exploiting the strengths of physics-based simulation and data-driven modeling, accounts for object dynamics during the grasping process, markedly enhancing the robot’s capability to handle previously unseen objects and scenarios in real-world settings. We benchmark DYNAMO-GRASP against established approaches via comprehensive evaluations in both simulated and real-world environments. DYNAMO-GRASP delivers improved grasping performance with greater consistency in both simulated and real-world settings. Remarkably, in real-world tests with challenging scenarios, our method demonstrates a success rate improvement of up to $48%$ over SOTA methods. Demonstrating a strong ability to adapt to complex and unexpected object dynamics, our method offers robust generalization to real-world challenges. The results of this research set the stage for more reliable and resilient robotic manipulation in intricate real-world situations. Experiment videos, dataset, model, and code are available at: https://sites.google.com/view/dynamo-grasp.

**摘要:** 在这项研究中，我们介绍了一种新的方法来解决吸力抓取点检测的挑战。我们的方法利用了基于物理的模拟和数据驱动建模的优势，考虑了抓取过程中的对象动力学，显著增强了机器人在现实世界环境中处理以前未见过的对象和场景的能力。我们通过在模拟和真实环境中进行全面评估，对照已建立的方法对Dynamo-Graff进行基准测试。Dynamo-Grave在模拟和真实环境中都提供了更好的抓取性能和更大的一致性。值得注意的是，在具有挑战性场景的真实世界测试中，我们的方法比SOTA方法的成功率提高了高达48%$。我们的方法显示了强大的适应复杂和意外对象动态的能力，为现实世界的挑战提供了健壮的泛化。这项研究的结果为在复杂的现实世界中进行更可靠和更具弹性的机器人操作奠定了基础。有关实验视频、数据集、模型和代码，请访问：https://sites.google.com/view/dynamo-grasp.

**[Paper URL](https://proceedings.mlr.press/v229/yang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23a/yang23a.pdf)** 

# HYDRA: Hybrid Robot Actions for Imitation Learning
**题目:** Hyspel：用于模仿学习的混合机器人动作

**作者:** Suneel Belkhale, Yuchen Cui, Dorsa Sadigh

**Abstract:** Imitation Learning (IL) is a sample efficient paradigm for robot learning using expert demonstrations. However, policies learned through IL suffer from state distribution shift at test time, due to compounding errors in action prediction which lead to previously unseen states. Choosing an action representation for the policy that minimizes this distribution shift is critical in imitation learning. Prior work propose using temporal action abstractions to reduce compounding errors, but they often sacrifice policy dexterity or require domain-specific knowledge. To address these trade-offs, we introduce HYDRA, a method that leverages a hybrid action space with two levels of action abstractions: sparse high-level waypoints and dense low-level actions. HYDRA dynamically switches between action abstractions at test time to enable both coarse and fine-grained control of a robot. In addition, HYDRA employs action relabeling to increase the consistency of actions in the dataset, further reducing distribution shift. HYDRA outperforms prior imitation learning methods by $30-40%$ on seven challenging simulation and real world environments, involving long-horizon tasks in the real world like making coffee and toasting bread. Videos are found on our website: https://tinyurl.com/3mc6793z

**摘要:** 模仿学习(IL)是利用专家演示进行机器人学习的一种有效范例。然而，通过IL学习的策略在测试时会出现状态分布变化，这是由于操作预测中的复合错误导致了以前看不到的状态。在模仿学习中，为策略选择一个最大限度地减少这种分布变化的动作表示是至关重要的。以前的工作建议使用时态动作抽象来减少复合错误，但它们经常牺牲策略的灵活性或需要特定于领域的知识。为了解决这些权衡，我们引入了Hydra，这是一种利用具有两个操作抽象级别的混合操作空间的方法：稀疏的高级路点和密集的低级别操作。九头蛇在测试时动态地在动作抽象之间切换，以实现对机器人的粗粒度和细粒度控制。此外，Hydra使用动作重新标记来提高数据集中动作的一致性，从而进一步减少分布偏移。在七个具有挑战性的模拟和现实世界环境中，九头蛇的表现比以前的模仿学习方法高出30%-40%$，这些环境涉及现实世界中的长期任务，如煮咖啡和烤面包。视频可在我们的网站上找到：https://tinyurl.com/3mc6793z

**[Paper URL](https://proceedings.mlr.press/v229/belkhale23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/belkhale23a/belkhale23a.pdf)** 

# Embodied Lifelong Learning for Task and Motion Planning
**题目:** 任务和动作规划的终身学习

**作者:** Jorge Mendez-Mendez, Leslie Pack Kaelbling, Tomás Lozano-Pérez

**Abstract:** A robot deployed in a home over long stretches of time faces a true lifelong learning problem. As it seeks to provide assistance to its users, the robot should leverage any accumulated experience to improve its own knowledge and proficiency. We formalize this setting with a novel formulation of lifelong learning for task and motion planning (TAMP), which endows our learner with the compositionality of TAMP systems. Exploiting the modularity of TAMP, we develop a mixture of generative models that produces candidate continuous parameters for a planner. Whereas most existing lifelong learning approaches determine a priori how data is shared across various models, our approach learns shared and non-shared models and determines which to use online during planning based on auxiliary tasks that serve as a proxy for each model’s understanding of a state. Our method exhibits substantial improvements (over time and compared to baselines) in planning success on 2D and BEHAVIOR domains.

**摘要:** 长期部署在家中的机器人面临着真正的终身学习问题。当机器人寻求为用户提供帮助时，它应该利用任何积累的经验来提高自己的知识和熟练程度。我们通过任务和动作规划终身学习（TAMP）的新颖公式来正式化这种设置，它赋予我们的学习者TAMP系统的组成性。利用TAMP的模块性，我们开发了生成模型的混合，为计划者生成候选连续参数。尽管大多数现有的终身学习方法先验地确定数据如何在各种模型之间共享，但我们的方法学习共享和非共享模型，并根据辅助任务来确定在规划期间在线使用哪个模型，这些辅助任务充当每个模型对状态的理解的代理。我们的方法在2D和行为领域的规划成功方面表现出了重大改进（随着时间的推移以及与基线相比）。

**[Paper URL](https://proceedings.mlr.press/v229/mendez-mendez23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mendez-mendez23a/mendez-mendez23a.pdf)** 

# 4D-Former: Multimodal 4D Panoptic Segmentation
**题目:** 4D成型器：多模式4D全景分割

**作者:** Ali Athar, Enxu Li, Sergio Casas, Raquel Urtasun

**Abstract:** 4D panoptic segmentation is a challenging but practically useful task that requires every point in a LiDAR point-cloud sequence to be assigned a semantic class label, and individual objects to be segmented and tracked over time. Existing approaches utilize only LiDAR inputs which convey limited information in regions with point sparsity. This problem can, however, be mitigated by utilizing RGB camera images which offer appearance-based information that can reinforce the geometry-based LiDAR features. Motivated by this, we propose 4D-Former: a novel method for 4D panoptic segmentation which leverages both LiDAR and image modalities, and predicts semantic masks as well as temporally consistent object masks for the input point-cloud sequence. We encode semantic classes and objects using a set of concise queries which absorb feature information from both data modalities. Additionally, we propose a learned mechanism to associate object tracks over time which reasons over both appearance and spatial location. We apply 4D-Former to the nuScenes and SemanticKITTI datasets where it achieves state-of-the-art results.

**摘要:** 4D全景分割是一项具有挑战性但实际有用的任务，它需要为LiDAR点云序列中的每个点分配一个语义类标签，并随着时间的推移对单个对象进行分割和跟踪。现有方法仅利用在点稀疏区域中传递有限信息的LiDAR输入。然而，这个问题可以通过利用RGB相机图像来缓解，RGB相机图像提供了基于外观的信息，可以加强基于几何的LiDAR特征。受此启发，我们提出了一种新的4D全景分割方法，该方法利用LiDAR和图像通道，预测输入点云序列的语义掩模和时间一致的目标掩模。我们使用一组简明的查询来编码语义类和对象，这些查询吸收了这两种数据模式的特征信息。此外，我们提出了一种学习机制来关联对象轨迹随时间的变化，这是由于外观和空间位置的原因。我们将4D-Former应用于nuScenes和SemancKITTI数据集，在那里它获得了最先进的结果。

**[Paper URL](https://proceedings.mlr.press/v229/athar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/athar23a/athar23a.pdf)** 

# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
**题目:** RT-2：视觉-语言-动作模型将网络知识转移到机器人控制

**作者:** Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi, Grecia Salazar, Michael S. Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J. Joshi, Alex Irpan, Brian Ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen, Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, Kehang Han

**Abstract:** We study how vision-language models trained on Internet-scale data can be incorporated directly into end-to-end robotic control to boost generalization and enable emergent semantic reasoning. Our goal is to enable a single end-to-end trained model to both learn to map robot observations to actions and enjoy the benefits of large-scale pretraining on language and vision-language data from the web. To this end, we propose to co-fine-tune state-of-the-art vision-language models on both robotic trajectory data and Internet-scale vision-language tasks, such as visual question answering. In contrast to other approaches, we propose a simple, general recipe to achieve this goal: in order to fit both natural language responses and robotic actions into the same format, we express the actions as text tokens and incorporate them directly into the training set of the model in the same way as natural language tokens. We refer to such category of models as vision-language-action models (VLA) and instantiate an example of such a model, which we call RT-2. Our extensive evaluation (6k evaluation trials) shows that our approach leads to performant robotic policies and enables RT-2 to obtain a range of emergent capabilities from Internet-scale training. This includes significantly improved generalization to novel objects, the ability to interpret commands not present in the robot training data (such as placing an object onto a particular number or icon), and the ability to perform rudimentary reasoning in response to user commands (such as picking up the smallest or largest object, or the one closest to another object). We further show that incorporating chain of thought reasoning allows RT-2 to perform multi-stage semantic reasoning, for example figuring out which object to pick up for use as an improvised hammer (a rock), or which type of drink is best suited for someone who is tired (an energy drink).

**摘要:** 我们研究了如何将在互联网规模的数据上训练的视觉语言模型直接结合到端到端的机器人控制中，以促进泛化和实现紧急语义推理。我们的目标是使单一的端到端训练模型既能学习将机器人的观察映射到动作，又能从网络上享受大规模语言和视觉语言数据预训练的好处。为此，我们建议在机器人轨迹数据和互联网规模的视觉语言任务(如视觉问题回答)上共同微调最先进的视觉语言模型。与其他方法不同，我们提出了一个简单、通用的方法来实现这一目标：为了使自然语言响应和机器人动作符合相同的格式，我们将动作表示为文本令牌，并以与自然语言令牌相同的方式将它们直接合并到模型的训练集中。我们将这样的模型称为视觉-语言-动作模型(VLA)，并实例化了这样一个模型的例子，我们称之为RT-2。我们的广泛评估(6k评估试验)表明，我们的方法导致了性能良好的机器人策略，并使RT-2能够从互联网规模的训练中获得一系列应急能力。这包括显著改进的对新对象的泛化、解释机器人训练数据中不存在的命令的能力(例如将对象放置在特定数量或图标上)、以及响应于用户命令执行基本推理的能力(例如拾取最小或最大的对象，或最接近另一对象的对象)。我们进一步证明，结合思维链推理允许RT-2执行多阶段语义推理，例如，找出拿起哪个对象用作临时锤子(岩石)，或者哪种类型的饮料最适合疲惫的人(能量饮料)。

**[Paper URL](https://proceedings.mlr.press/v229/zitkovich23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zitkovich23a/zitkovich23a.pdf)** 

# Seeing-Eye Quadruped Navigation with Force Responsive Locomotion Control
**题目:** 具有力响应运动控制的直视四足导航

**作者:** David DeFazio, Eisuke Hirota, Shiqi Zhang

**Abstract:** Seeing-eye robots are very useful tools for guiding visually impaired people, potentially producing a huge societal impact given the low availability and high cost of real guide dogs. Although a few seeing-eye robot systems have already been demonstrated, none considered external tugs from humans, which frequently occur in a real guide dog setting. In this paper, we simultaneously train a locomotion controller that is robust to external tugging forces via Reinforcement Learning (RL), and an external force estimator via supervised learning. The controller ensures stable walking, and the force estimator enables the robot to respond to the external forces from the human. These forces are used to guide the robot to the global goal, which is unknown to the robot, while the robot guides the human around nearby obstacles via a local planner. Experimental results in simulation and on hardware show that our controller is robust to external forces, and our seeing-eye system can accurately detect force direction. We demonstrate our full seeing-eye robot system on a real quadruped robot with a blindfolded human.

**摘要:** 导盲犬机器人是指导视障人士的非常有用的工具，鉴于真实导盲犬的可获得性低且成本高，这可能会产生巨大的社会影响。尽管一些导盲犬机器人系统已经被演示，但没有一个被认为是来自人类的外部拖拽，这种情况经常发生在真实的导盲犬环境中。在本文中，我们同时通过强化学习(RL)训练一个对外部牵引力具有鲁棒性的运动控制器，并通过有监督学习训练一个外力估计器。控制器确保了稳定的行走，力估计器使机器人能够响应来自人类的外力。这些力被用来引导机器人到达机器人未知的全局目标，而机器人通过当地的规划器引导人类绕过附近的障碍物。仿真和硬件实验结果表明，该控制器对外力具有较强的鲁棒性，视眼系统能够准确地检测力的方向。我们在一个真实的四足机器人上演示了我们的完整视眼机器人系统，其中一个人被蒙住了眼睛。

**[Paper URL](https://proceedings.mlr.press/v229/defazio23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/defazio23a/defazio23a.pdf)** 

# Waypoint-Based Imitation Learning for Robotic Manipulation
**题目:** 基于路点的机器人操纵模仿学习

**作者:** Lucy Xiaoyang Shi, Archit Sharma, Tony Z. Zhao, Chelsea Finn

**Abstract:** While imitation learning methods have seen a resurgent interest for robotic manipulation, the well-known problem of compounding errors continues to afflict behavioral cloning (BC). Waypoints can help address this problem by reducing the horizon of the learning problem for BC, and thus, the errors compounded over time. However, waypoint labeling is underspecified, and requires additional human supervision. Can we generate waypoints automatically without any additional human supervision? Our key insight is that if a trajectory segment can be approximated by linear motion, the endpoints can be used as waypoints. We propose Automatic Waypoint Extraction (AWE) for imitation learning, a preprocessing module to decompose a demonstration into a minimal set of waypoints which when interpolated linearly can approximate the trajectory up to a specified error threshold. AWE can be combined with any BC algorithm, and we find that AWE can increase the success rate of state-of-the-art algorithms by up to $25%$ in simulation and by $4-28%$ on real-world bimanual manipulation tasks, reducing the decision making horizon by up to a factor of 10. Videos and code are available at https://lucys0.github.io/awe/.

**摘要:** 虽然模仿学习方法重新引起了人们对机器人操作的兴趣，但众所周知的合成错误问题仍在困扰着行为克隆(BC)。路点可以通过减少BC学习问题的范围来帮助解决这个问题，因此，随着时间的推移，错误会变得更加复杂。然而，路点标签没有明确规定，需要额外的人工监督。我们可以在没有任何额外人工监督的情况下自动生成路点吗？我们的主要见解是，如果轨迹段可以用直线运动来近似，那么端点可以用作路点。我们提出了用于模仿学习的自动路点提取(AWE)，这是一个预处理模块，将演示分解成一个最小的路点集，当线性内插时，这些路点可以逼近轨迹，直到指定的误差阈值。AWE可以与任何BC算法相结合，我们发现AWE可以将最先进的算法的成功率在模拟中提高高达25%$，在现实世界的双手操作任务中提高$4-28%$，将决策范围缩短高达10倍。视频和代码可在https://lucys0.github.io/awe/.上找到

**[Paper URL](https://proceedings.mlr.press/v229/shi23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shi23b/shi23b.pdf)** 

# Multi-Resolution Sensing for Real-Time Control with Vision-Language Models
**题目:** 利用视觉语言模型进行实时控制的多分辨率传感

**作者:** Saumya Saxena, Mohit Sharma, Oliver Kroemer

**Abstract:** Leveraging sensing modalities across diverse spatial and temporal resolutions can improve performance of robotic manipulation tasks. Multi-spatial resolution sensing provides hierarchical information captured at different spatial scales and enables both coarse and precise motions. Simultaneously multi-temporal resolution sensing enables the agent to exhibit high reactivity and real-time control. In this work, we propose a framework for learning generalizable language-conditioned multi-task policies that utilize sensing at different spatial and temporal resolutions using networks of varying capacities to effectively perform real time control of precise and reactive tasks. We leverage off-the-shelf pretrained vision-language models to operate on low-frequency global features along with small non-pretrained models to adapt to high frequency local feedback. Through extensive experiments in 3 domains (coarse, precise and dynamic manipulation tasks), we show that our approach significantly improves ($2\times$ on average) over recent multi-task baselines. Further, our approach generalizes well to visual and geometric variations in target objects and to varying interaction forces.

**摘要:** 在不同的空间和时间分辨率上利用传感模式可以提高机器人操作任务的性能。多空间分辨率传感提供了在不同空间尺度上捕获的分层信息，并支持粗略和精确的运动。同时，多时相分辨率感知使智能体具有较高的反应性和实时控制能力。在这项工作中，我们提出了一个学习泛化语言条件的多任务策略的框架，该策略利用不同空间和时间分辨率的感知，使用不同能力的网络来有效地执行对精确和反应性任务的实时控制。我们利用现成的预先训练的视觉语言模型对低频率的全局特征以及小的非预先训练的模型进行操作，以适应高频局部反馈。通过在3个领域(粗略、精确和动态操作任务)上的广泛实验，我们的方法比最近的多任务基线显著提高(平均2倍$)。此外，我们的方法很好地适用于目标对象中的视觉和几何变化以及不同的相互作用力。

**[Paper URL](https://proceedings.mlr.press/v229/saxena23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/saxena23a/saxena23a.pdf)** 

# SCALE: Causal Learning and Discovery of Robot Manipulation Skills using Simulation
**题目:** SCALE：使用模拟进行因果学习和机器人操纵技能的发现

**作者:** Tabitha Edith Lee, Shivam Vats, Siddharth Girdhar, Oliver Kroemer

**Abstract:** We propose SCALE, an approach for discovering and learning a diverse set of interpretable robot skills from a limited dataset. Rather than learning a single skill which may fail to capture all the modes in the data, we first identify the different modes via causal reasoning and learn a separate skill for each of them. Our main insight is to associate each mode with a unique set of causally relevant context variables that are discovered by performing causal interventions in simulation. This enables data partitioning based on the causal processes that generated the data, and then compressed skills that ignore the irrelevant variables can be trained. We model each robot skill as a Regional Compressed Option, which extends the options framework by associating a causal process and its relevant variables with the option. Modeled as the skill Data Generating Region, each causal process is local in nature and hence valid over only a subset of the context space. We demonstrate our approach for two representative manipulation tasks: block stacking and peg-in-hole insertion under uncertainty. Our experiments show that our approach yields diverse skills that are compact, robust to domain shifts, and suitable for sim-to-real transfer.

**摘要:** 我们提出了Scale，一种从有限的数据集中发现和学习各种可解释的机器人技能的方法。我们不是学习一种可能无法捕获数据中所有模式的单一技能，而是首先通过因果推理识别不同的模式，并为每种模式学习一种单独的技能。我们的主要见解是将每种模式与一组唯一的因果相关上下文变量相关联，这些变量是通过在模拟中执行因果干预而发现的。这实现了基于生成数据的因果过程的数据分区，然后可以训练忽略不相关变量的压缩技能。我们将每个机器人技能建模为区域压缩期权，通过将因果过程及其相关变量与期权相关联来扩展期权框架。每个因果过程被建模为技能数据生成区域，本质上是局部的，因此仅在上下文空间的子集上有效。我们在两个典型的操作任务中演示了我们的方法：块堆叠和不确定情况下的插销孔。我们的实验表明，我们的方法产生了不同的技能，这些技能紧凑，对域转换具有健壮性，并且适合于从模拟到真实的转换。

**[Paper URL](https://proceedings.mlr.press/v229/lee23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lee23b/lee23b.pdf)** 

# Learning Robot Manipulation from Cross-Morphology Demonstration
**题目:** 从跨形态演示中学习机器人操纵

**作者:** Gautam Salhotra, I-Chun Arthur Liu, Gaurav S. Sukhatme

**Abstract:** Some Learning from Demonstrations (LfD) methods handle small mismatches in the action spaces of the teacher and student. Here we address the casewhere the teacher’s morphology is substantially different from that of the student. Our framework, Morphological Adaptation in Imitation Learning (MAIL), bridges this gap allowing us to train an agent from demonstrations by other agents with significantly different morphologies. MAIL learns from suboptimal demonstrations, so long as they provide some guidance towards a desired solution. We demonstrate MAIL on manipulation tasks with rigid and deformable objects including 3D cloth manipulation interacting with rigid obstacles. We train a visual control policy for a robot with one end-effector using demonstrations from a simulated agent with two end-effectors. MAIL shows up to $24%$ improvement in a normalized performance metric over LfD and non-LfD baselines. It is deployed to a real Franka Panda robot, handles multiple variations in properties for objects (size, rotation, translation), and cloth-specific properties (color, thickness, size, material).

**摘要:** 一些从演示中学习(LFD)的方法处理教师和学生的动作空间中的微小不匹配。在这里，我们讨论教师的形态与学生的形态有很大不同的情况。我们的框架，模仿学习中的形态适应(Mail)，弥合了这一差距，允许我们从其他具有显著不同形态的代理的演示中训练代理。Mail可以从不太理想的演示中学习，只要它们为所需的解决方案提供一些指导。我们演示了使用刚性和可变形对象的Mail on操纵任务，包括与刚性障碍物交互的3D布料操纵。我们使用具有两个末端执行器的模拟智能体的演示来训练具有一个末端执行器的机器人的视觉控制策略。MAIL显示，与LFD和非LFD基准相比，归一化性能指标最高可提高24%$。它被部署在一个真正的Franka Panda机器人上，处理对象属性(大小、旋转、平移)和布料特定属性(颜色、厚度、大小、材质)的多种变化。

**[Paper URL](https://proceedings.mlr.press/v229/salhotra23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/salhotra23a/salhotra23a.pdf)** 

# Synthesizing Navigation Abstractions for Planning with Portable Manipulation Skills
**题目:** 利用便携式操纵技能合成导航抽象以进行规划

**作者:** Eric Rosen, Steven James, Sergio Orozco, Vedant Gupta, Max Merlin, Stefanie Tellex, George Konidaris

**Abstract:** We address the problem of efficiently learning high-level abstractions for task-level robot planning. Existing approaches require large amounts of data and fail to generalize learned abstractions to new environments. To address this, we propose to exploit the independence between spatial and non-spatial state variables in the preconditions of manipulation and navigation skills, mirroring the manipulation-navigation split in robotics research. Given a collection of portable manipulation abstractions (i.e., object-centric manipulation skills paired with matching symbolic representations), we derive an algorithm to automatically generate navigation abstractions that support mobile manipulation planning in a novel environment. We apply our approach to simulated data in AI2Thor and on real robot hardware with a coffee preparation task, efficiently generating plannable representations for mobile manipulators in just a few minutes of robot time, significantly outperforming state-of-the-art baselines.

**摘要:** 我们解决了有效地学习任务级机器人规划的高层抽象的问题。现有方法需要大量数据，并且无法将学习到的抽象推广到新环境。为了解决这一问题，我们提出在操作和导航技能的前提下，利用空间和非空间状态变量之间的独立性，反映了机器人学研究中的操作-导航分离。给出了一组可移植的操作抽象(即，以对象为中心的操作技能和匹配的符号表示)，我们推导了一个算法来自动生成导航抽象，以支持在新环境中的移动操作规划。我们将我们的方法应用于AI2Thor和具有咖啡准备任务的真实机器人硬件上的模拟数据，在几分钟的机器人时间内高效地为移动机械手生成可规划的表示，远远超过最先进的基线。

**[Paper URL](https://proceedings.mlr.press/v229/rosen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rosen23a/rosen23a.pdf)** 

# Transforming a Quadruped into a Guide Robot for the Visually Impaired: Formalizing Wayfinding, Interaction Modeling, and Safety Mechanism
**题目:** 将四足动物改造成视觉障碍者的向导机器人：正式化寻路、交互建模和安全机制

**作者:** J. Taery Kim, Wenhao Yu, Yash Kothari, Bruce Walker, Jie Tan, Greg Turk, Sehoon Ha

**Abstract:** This paper explores the principles for transforming a quadrupedal robot into a guide robot for individuals with visual impairments. A guide robot has great potential to resolve the limited availability of guide animals that are accessible to only two to three percent of the potential blind or visually impaired (BVI) users. To build a successful guide robot, our paper explores three key topics: (1) formalizing the navigation mechanism of a guide dog and a human, (2) developing a data-driven model of their interaction, and (3) improving user safety. First, we formalize the wayfinding task of the human-guide robot team using Markov Decision Processes based on the literature and interviews. Then we collect real human-robot interaction data from three visually impaired and six sighted people and develop an interaction model called the "Delayed Harness" to effectively simulate the navigation behaviors of the team. Additionally, we introduce an action shielding mechanism to enhance user safety by predicting and filtering out dangerous actions. We evaluate the developed interaction model and the safety mechanism in simulation, which greatly reduce the prediction errors and the number of collisions, respectively. We also demonstrate the integrated system on an AlienGo robot with a rigid harness, by guiding users over 100+ meter trajectories.

**摘要:** 本文探讨了将四足机器人改造成为视障人士服务的引导机器人的原理。导游机器人具有巨大的潜力，可以解决导游动物有限的问题，而导游动物只有2%到3%的潜在盲人或视障(BVI)用户可以接触到。为了构建一个成功的导游机器人，本文探讨了三个关键问题：(1)形式化导盲犬和人类的导航机制；(2)建立导盲犬和人类交互的数据驱动模型；(3)提高用户安全性。首先，在文献和访谈的基础上，利用马尔可夫决策过程对人工导引机器人团队的寻路任务进行形式化描述。然后，我们收集了3名视障人士和6名视障人士的真实人-机器人交互数据，并开发了一个名为延迟线束的交互模型，以有效地模拟团队的导航行为。此外，我们还引入了动作屏蔽机制，通过预测和过滤危险动作来增强用户的安全性。我们在仿真中对所提出的交互模型和安全机制进行了评估，分别大大降低了预测误差和碰撞次数。我们还在带有刚性线束的AlienGo机器人上演示了集成系统，引导用户超过100米的轨迹。

**[Paper URL](https://proceedings.mlr.press/v229/kim23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kim23c/kim23c.pdf)** 

# A Bayesian Approach to Robust Inverse Reinforcement Learning
**题目:** 鲁棒反向强化学习的Bayesian方法

**作者:** Ran Wei, Siliang Zeng, Chenliang Li, Alfredo Garcia, Anthony D McDonald, Mingyi Hong

**Abstract:** We consider a Bayesian approach to offline model-based inverse reinforcement learning (IRL). The proposed framework differs from existing offline model-based IRL approaches by performing simultaneous estimation of the expert’s reward function and subjective model of environment dynamics. We make use of a class of prior distributions which parameterizes how accurate the expert’s model of the environment is to develop efficient algorithms to estimate the expert’s reward and subjective dynamics in high-dimensional settings. Our analysis reveals a novel insight that the estimated policy exhibits robust performance when the expert is believed (a priori) to have a highly accurate model of the environment. We verify this observation in the MuJoCo environments and show that our algorithms outperform state-of-the-art offline IRL algorithms.

**摘要:** 我们考虑了基于离线模型的反向强化学习（IRL）的Bayesian方法。所提出的框架与现有的基于离线模型的IRL方法不同，它同时执行专家的奖励函数和环境动态的主观模型的估计。我们利用一类先验分布，它参数化了专家环境模型的准确性，以开发有效的算法来估计专家在多维环境中的回报和主观动态。我们的分析揭示了一个新颖的见解，即当专家被认为（先验）拥有高度准确的环境模型时，估计的策略表现出稳健的性能。我们在MuJoCo环境中验证了这一观察结果，并表明我们的算法优于最先进的离线IRL算法。

**[Paper URL](https://proceedings.mlr.press/v229/wei23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wei23a/wei23a.pdf)** 

# ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation
**题目:** ChainedDistuser：统一机器人操纵的轨迹扩散和关键位姿预测

**作者:** Zhou Xian, Nikolaos Gkanatsios, Theophile Gervet, Tsung-Wei Ke, Katerina Fragkiadaki

**Abstract:** We present ChainedDiffuser, a policy architecture that unifies action keypose prediction and trajectory diffusion generation for learning robot manipulation from demonstrations. Our main innovation is to use a global transformer-based action predictor to predict actions at keyframes, a task that requires multi- modal semantic scene understanding, and to use a local trajectory diffuser to predict trajectory segments that connect predicted macro-actions. ChainedDiffuser sets a new record on established manipulation benchmarks, and outperforms both state-of-the-art keypose (macro-action) prediction models that use motion plan- ners for trajectory prediction, and trajectory diffusion policies that do not predict keyframe macro-actions. We conduct experiments in both simulated and real-world environments and demonstrate ChainedDiffuser’s ability to solve a wide range of manipulation tasks involving interactions with diverse objects.

**摘要:** 我们介绍了ChainedDivuser，这是一种政策架构，它将动作关键位数预测和轨迹扩散生成统一起来，用于从演示中学习机器人操纵。我们的主要创新是使用基于全局变换器的动作预测器来预测关键帧处的动作，这是一项需要多模式语义场景理解的任务，并使用局部轨迹扩散器来预测连接预测的宏动作的轨迹片段。ChainedDivuser在既定操纵基准上创下了新纪录，并且优于使用运动计划进行轨迹预测的最先进的关键姿势（宏动作）预测模型和不预测关键帧宏动作的轨迹扩散策略。我们在模拟和现实世界环境中进行实验，并展示ChainedDistuser解决涉及与不同对象交互的广泛操纵任务的能力。

**[Paper URL](https://proceedings.mlr.press/v229/xian23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xian23a/xian23a.pdf)** 

# IIFL: Implicit Interactive Fleet Learning from Heterogeneous Human Supervisors
**题目:** IIFL：来自异类人类主管的隐性互动舰队学习

**作者:** Gaurav Datta, Ryan Hoque, Anrui Gu, Eugen Solowjow, Ken Goldberg

**Abstract:** Imitation learning has been applied to a range of robotic tasks, but can struggle when robots encounter edge cases that are not represented in the training data (i.e., distribution shift). Interactive fleet learning (IFL) mitigates distribution shift by allowing robots to access remote human supervisors during task execution and learn from them over time, but different supervisors may demonstrate the task in different ways. Recent work proposes Implicit Behavior Cloning (IBC), which is able to represent multimodal demonstrations using energy-based models (EBMs). In this work, we propose Implicit Interactive Fleet Learning (IIFL), an algorithm that builds on IBC for interactive imitation learning from multiple heterogeneous human supervisors. A key insight in IIFL is a novel approach for uncertainty quantification in EBMs using Jeffreys divergence. While IIFL is more computationally expensive than explicit methods, results suggest that IIFL achieves a 2.8x higher success rate in simulation experiments and a 4.5x higher return on human effort in a physical block pushing task over (Explicit) IFL, IBC, and other baselines.

**摘要:** 模仿学习已被应用于一系列机器人任务，但当机器人遇到训练数据中未表示的边缘情况(即分布偏移)时，可能会遇到困难。交互式舰队学习(IFL)通过允许机器人在任务执行期间访问远程人类主管并随着时间的推移向他们学习来缓解分布偏移，但不同的主管可能会以不同的方式演示任务。最近的工作提出了隐式行为克隆(IBC)，它能够使用基于能量的模型(EBM)来表示多模式演示。在这项工作中，我们提出了隐式交互舰队学习(IIFL)，这是一种建立在IBC基础上的算法，用于从多个异质人类监督者那里进行交互模仿学习。IIFL的一个关键见解是使用Jeffreys散度对EBM中的不确定性进行量化的新方法。虽然IIFL在计算上比显式方法更昂贵，但结果表明，在模拟实验中，IIFL在(显式)IFL、IBC和其他基线上实现了2.8倍的成功率和4.5倍的人工回报。

**[Paper URL](https://proceedings.mlr.press/v229/datta23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/datta23a/datta23a.pdf)** 

# CAT: Closed-loop Adversarial Training for Safe End-to-End Driving
**题目:** CAT：安全端到端驾驶的闭环对抗培训

**作者:** Linrui Zhang, Zhenghao Peng, Quanyi Li, Bolei Zhou

**Abstract:** Driving safety is a top priority for autonomous vehicles. Orthogonal to prior work handling accident-prone traffic events by algorithm designs at the policy level, we investigate a Closed-loop Adversarial Training (CAT) framework for safe end-to-end driving in this paper through the lens of environment augmentation. CAT aims to continuously improve the safety of driving agents by training the agent on safety-critical scenarios that are dynamically generated over time. A novel resampling technique is developed to turn log-replay real-world driving scenarios into safety-critical ones via probabilistic factorization, where the adversarial traffic generation is modeled as the multiplication of standard motion prediction sub-problems. Consequently, CAT can launch more efficient physical attacks compared to existing safety-critical scenario generation methods and yields a significantly less computational cost in the iterative learning pipeline. We incorporate CAT into the MetaDrive simulator and validate our approach on hundreds of driving scenarios imported from real-world driving datasets. Experimental results demonstrate that CAT can effectively generate adversarial scenarios countering the agent being trained. After training, the agent can achieve superior driving safety in both log-replay and safety-critical traffic scenarios on the held-out test set. Code and data are available at: https://metadriverse.github.io/cat

**摘要:** 驾驶安全是自动驾驶汽车的首要任务。与以往处理事故多发交通事件的算法设计在策略层面上的工作正交，本文从环境增强的角度研究了一种端到端安全驾驶的闭环对抗性训练(CAT)框架。CAT旨在通过对驾驶代理进行安全关键场景培训，不断提高驾驶代理的安全性，这些场景是随着时间的推移动态生成的。提出了一种新的重采样技术，通过概率因式分解将日志重放的真实驾驶场景转化为安全关键场景，其中对抗性流量的生成被建模为标准运动预测子问题的乘法。因此，与现有的安全关键场景生成方法相比，CAT可以发起更有效的物理攻击，并在迭代学习管道中产生显著更低的计算成本。我们将CAT集成到MetaDrive模拟器中，并在从真实驾驶数据集导入的数百个驾驶场景上验证了我们的方法。实验结果表明，CAT能够有效地生成对抗被训练智能体的对抗性场景。经过培训后，该代理可以在日志重放和安全关键交通场景中在坚持测试集上实现卓越的驾驶安全。代码和数据可在以下网址获得：https://metadriverse.github.io/cat

**[Paper URL](https://proceedings.mlr.press/v229/zhang23g.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23g/zhang23g.pdf)** 

# Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control
**题目:** 神经图控制屏障函数引导的分布式避碰多智能体控制

**作者:** Songyuan Zhang, Kunal Garg, Chuchu Fan

**Abstract:** We consider the problem of designing distributed collision-avoidance multi-agent control in large-scale environments with potentially moving obstacles, where a large number of agents are required to maintain safety using only local information and reach their goals. This paper addresses the problem of collision avoidance, scalability, and generalizability by introducing graph control barrier functions (GCBFs) for distributed control. The newly introduced GCBF is based on the well-established CBF theory for safety guarantees but utilizes a graph structure for scalable and generalizable decentralized control. We use graph neural networks to learn both neural a GCBF certificate and distributed control. We also extend the framework from handling state-based models to directly taking point clouds from LiDAR for more practical robotics settings. We demonstrated the efficacy of GCBF in a variety of numerical experiments, where the number, density, and traveling distance of agents, as well as the number of unseen and uncontrolled obstacles increase. Empirical results show that GCBF outperforms leading methods such as MAPPO and multi-agent distributed CBF (MDCBF). Trained with only $16$ agents, GCBF can achieve up to $3$ times improvement of success rate (agents reach goals and never encountered in any collisions) on $<500$ agents, and still maintain more than $50%$ success rates for $>\!1000$ agents when other methods completely fail.

**摘要:** 我们考虑了在具有潜在移动障碍物的大规模环境中的分布式避碰多智能体控制设计问题，在这种环境中，需要大量的智能体仅使用局部信息来维护安全并达到他们的目标。通过引入用于分布式控制的图控制屏障函数(GCBF)来解决冲突避免、可伸缩性和泛化问题。新引入的GCBF是基于公认的CBF安全保证理论，但利用图结构来实现可扩展和可推广的分散控制。我们使用图神经网络来学习神经网络和分布式控制。我们还将该框架从处理基于状态的模型扩展到直接从LiDAR获取点云，以实现更实用的机器人设置。我们在各种数值实验中展示了GCBF的有效性，在这些数值实验中，智能体的数量、密度和移动距离以及看不见和无法控制的障碍物的数量增加。实证结果表明，GCBF优于MAPPO和多智能体分布式CBF(MDCBF)等主流方法。仅用$16$代理进行训练，GCBF可以使$<500$代理的成功率(代理达到目标且在任何冲突中从未遇到)提高高达$3$倍，并且当其他方法完全失败时，$>1000$代理仍然保持$50%$以上的成功率。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23h.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23h/zhang23h.pdf)** 

# STERLING: Self-Supervised Terrain Representation Learning from Unconstrained Robot Experience
**题目:** 斯特林：从不受约束的机器人体验中进行自我监督的地形表示学习

**作者:** Haresh Karnan, Elvin Yang, Daniel Farkash, Garrett Warnell, Joydeep Biswas, Peter Stone

**Abstract:** Terrain awareness, i.e., the ability to identify and distinguish different types of terrain, is a critical ability that robots must have to succeed at autonomous off-road navigation. Current approaches that provide robots with this awareness either rely on labeled data which is expensive to collect, engineered features and cost functions that may not generalize, or expert human demonstrations which may not be available. Towards endowing robots with terrain awareness without these limitations, we introduce Self-supervised TErrain Representation LearnING (STERLING), a novel approach for learning terrain representations that relies solely on easy-to-collect, unconstrained (e.g., non-expert), and unlabelled robot experience, with no additional constraints on data collection. STERLING employs a novel multi-modal self-supervision objective through non-contrastive representation learning to learn relevant terrain representations for terrain-aware navigation. Through physical robot experiments in off-road environments, we evaluate STERLING features on the task of preference-aligned visual navigation and find that STERLING features perform on par with fully-supervised approaches and outperform other state-of-the-art methods with respect to preference alignment. Additionally, we perform a large-scale experiment of autonomously hiking a 3-mile long trail which STERLING completes successfully with only two manual interventions, demonstrating its robustness to real-world off-road conditions.

**摘要:** 地形感知，即识别和区分不同类型地形的能力，是机器人在自主越野导航中必须具备的关键能力。目前为机器人提供这种感知的方法要么依赖于收集成本高昂的标签数据，要么依赖于可能无法推广的工程特征和成本函数，或者依赖于可能无法获得的专家人类演示。为了赋予机器人不受这些限制的地形感知能力，我们引入了自监督地形表示学习(Sterling)，这是一种新的学习地形表示的方法，它完全依赖于易于收集、不受限制(例如，非专家)和未标记的机器人经验，不需要额外的数据收集限制。Sterling采用了一种新颖的多模式自我监控目标，通过非对比表示学习学习相关的地形表示，以实现地形感知导航。通过在越野环境下的物理机器人实验，我们对Sterling特征在偏好对齐视觉导航任务中的性能进行了评估，发现Sterling特征在偏好对齐方面的表现与完全监督方法相当，并优于其他最先进的方法。此外，我们进行了一项大规模的自主徒步旅行3英里长的小径的实验，Sterling只需两次人工干预就成功完成了这一实验，展示了其对现实世界越野条件的稳健性。

**[Paper URL](https://proceedings.mlr.press/v229/karnan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/karnan23a/karnan23a.pdf)** 

# Towards General Single-Utensil Food Acquisition with Human-Informed Actions
**题目:** 以人为本的行动实现一般单一器皿食品采购

**作者:** Ethan Kroll Gordon, Amal Nanavati, Ramya Challa, Bernie Hao Zhu, Taylor Annette Kessler Faulkner, Siddhartha Srinivasa

**Abstract:** Food acquisition with common general-purpose utensils is a necessary component of robot applications like in-home assistive feeding. Learning acquisition policies in this space is difficult in part because any model will need to contend with extensive state and actions spaces. Food is extremely diverse and generally difficult to simulate, and acquisition actions like skewers, scoops, wiggles, and twirls can be parameterized in myriad ways. However, food’s visual diversity can belie a degree of physical homogeneity, and many foods allow flexibility in how they are acquired. Due to these facts, our key insight is that a small subset of actions is sufficient to acquire a wide variety of food items. In this work, we present a methodology for identifying such a subset from limited human trajectory data. We first develop an over-parameterized action space of robot acquisition trajectories that capture the variety of human food acquisition technique. By mapping human trajectories into this space and clustering, we construct a discrete set of 11 actions. We demonstrate that this set is capable of acquiring a variety of food items with $\geq80%$ success rate, a rate that users have said is sufficient for in-home robot-assisted feeding. Furthermore, since this set is so small, we also show that we can use online learning to determine a sufficiently optimal action for a previously-unseen food item over the course of a single meal.

**摘要:** 使用通用餐具获取食物是机器人应用(如家庭辅助喂养)的必要组成部分。在这一领域学习收购政策是困难的，部分原因是任何模型都需要应对广泛的状态和操作空间。食物是非常多样的，通常很难模拟，而像串、勺子、摆动和旋转这样的获取动作可以通过各种方式进行参数化。然而，食物的视觉多样性可能掩盖了一定程度的物理同质性，而且许多食物允许灵活地获取它们。由于这些事实，我们的关键洞察力是，一小部分行动足以获得各种各样的食物。在这项工作中，我们提出了一种从有限的人体轨迹数据中识别这样一个子集的方法。我们首先开发了一种捕捉人类食物获取技术多样性的机器人获取轨迹的超参数动作空间。通过将人的轨迹映射到这个空间并进行聚类，我们构建了一个由11个动作组成的离散集合。我们证明，这套设备能够获取各种食物，成功率为80%，用户曾表示，这个成功率足以满足家庭机器人辅助喂养的要求。此外，由于这个集合如此之小，我们还表明，我们可以使用在线学习来确定在一顿饭的过程中针对以前从未见过的食物的充分最佳行动。

**[Paper URL](https://proceedings.mlr.press/v229/gordon23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gordon23a/gordon23a.pdf)** 

# ScalableMap: Scalable Map Learning for Online Long-Range Vectorized HD Map Construction
**题目:** ScalableMap：可扩展地图学习，用于在线远程载体化高清地图构建

**作者:** Jingyi Yu, Zizhao Zhang, Shengfu Xia, Jizhang Sang

**Abstract:** We propose a novel end-to-end pipeline for online long-range vectorized high-definition (HD) map construction using on-board camera sensors. The vectorized representation of HD maps, employing polylines and polygons to represent map elements, is widely used by downstream tasks. However, previous schemes designed with reference to dynamic object detection overlook the structural constraints within linear map elements, resulting in performance degradation in long-range scenarios. In this paper, we exploit the properties of map elements to improve the performance of map construction. We extract more accurate bird’s eye view (BEV) features guided by their linear structure, and then propose a hierarchical sparse map representation to further leverage the scalability of vectorized map elements, and design a progressive decoding mechanism and a supervision strategy based on this representation. Our approach, ScalableMap, demonstrates superior performance on the nuScenes dataset, especially in long-range scenarios, surpassing previous state-of-the-art model by 6.5 mAP while achieving 18.3 FPS.

**摘要:** 我们提出了一种新型的端到端管道，用于使用车载摄像头传感器在线构建远程矢量化高清地图。高清地图的矢量化表示，使用折线和多边形来表示地图元素，被下游任务广泛使用。然而，以往参考动态目标检测设计的方案忽略了线性地图元素内部的结构约束，导致在远程场景下性能下降。在本文中，我们利用地图元素的性质来提高地图构建的性能。根据鸟瞰地图的线性结构，提取出更准确的鸟瞰特征，提出了一种层次稀疏地图表示法，进一步利用矢量化地图元素的可扩展性，设计了一种渐进译码机制和基于该表示法的监督策略。我们的方法ScalableMap在nuScenes数据集上展示了卓越的性能，特别是在远程场景中，比以前最先进的模型高出6.5MAP，同时实现了18.3FPS。

**[Paper URL](https://proceedings.mlr.press/v229/yu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yu23b/yu23b.pdf)** 

# Tuning Legged Locomotion Controllers via Safe Bayesian Optimization
**题目:** 通过安全Bayesian优化调整腿部运动控制器

**作者:** Daniel Widmer, Dongho Kang, Bhavya Sukhija, Jonas Hübotter, Andreas Krause, Stelian Coros

**Abstract:** This paper presents a data-driven strategy to streamline the deployment of model-based controllers in legged robotic hardware platforms. Our approach leverages a model-free safe learning algorithm to automate the tuning of control gains, addressing the mismatch between the simplified model used in the control formulation and the real system. This method substantially mitigates the risk of hazardous interactions with the robot by sample-efficiently optimizing parameters within a probably safe region. Additionally, we extend the applicability of our approach to incorporate the different gait parameters as contexts, leading to a safe, sample-efficient exploration algorithm capable of tuning a motion controller for diverse gait patterns. We validate our method through simulation and hardware experiments, where we demonstrate that the algorithm obtains superior performance on tuning a model-based motion controller for multiple gaits safely.

**摘要:** 本文提出了一种数据驱动策略，以简化腿机器人硬件平台中基于模型的控制器的部署。我们的方法利用无模型安全学习算法来自动调整控制收益，解决控制公式中使用的简化模型与真实系统之间的不匹配问题。该方法通过在可能的安全区域内以样本效率优化参数，大大降低了与机器人危险互动的风险。此外，我们扩展了我们方法的适用性，将不同的步态参数作为上下文结合起来，从而产生了一种安全、样本高效的探索算法，能够针对不同的步态模式调整运动控制器。我们通过模拟和硬件实验验证了我们的方法，其中我们证明该算法在安全地针对多个步态调整基于模型的运动控制器方面获得了卓越的性能。

**[Paper URL](https://proceedings.mlr.press/v229/widmer23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/widmer23a/widmer23a.pdf)** 

# TraCo: Learning Virtual Traffic Coordinator for Cooperation with Multi-Agent Reinforcement Learning
**题目:** TraCo：学习虚拟交通协调员，与多智能体强化学习合作

**作者:** Weiwei Liu, Wei Jing, lingping Gao, Ke Guo, Gang Xu, Yong Liu

**Abstract:** Multi-agent reinforcement learning (MARL) has emerged as a popular technique in diverse domains due to its ability to automate system controller design and facilitate continuous intelligence learning. For instance, traffic flow is often trained with MARL to enable intelligent simulations for autonomous driving. However, The existing MARL algorithm only characterizes the relative degree of each agent’s contribution to the team, and cannot express the contribution that the team needs from the agent. Especially in the field of autonomous driving, the team changes over time, and the agent needs to act directly according to the needs of the team. To address these limitations, we propose an innovative method inspired by realistic traffic coordinators called the Traffic Coordinator Network (TraCo). Our approach leverages a combination of cross-attention and counterfactual advantage function, allowing us to extract distinctive characteristics of domain agents and accurately quantify the contribution that a team needs from an agent. Through experiments conducted on four traffic tasks, we demonstrate that our method outperforms existing approaches, yielding superior performance. Furthermore, our approach enables the emergence of rich and diverse social behaviors among vehicles within the traffic flow.

**摘要:** 多智能体强化学习(MAIL)以其自动化系统控制器设计和促进持续智能学习的能力而在不同领域成为一种流行的技术。例如，交通流经常使用Marl进行训练，以实现自动驾驶的智能模拟。然而，现有的MAIL算法只刻画了每个智能体对团队贡献的相对程度，不能表达团队需要智能体做出的贡献。特别是在自动驾驶领域，团队会随着时间的推移而变化，代理需要根据团队的需求直接行动。为了解决这些局限性，我们提出了一种受现实交通协调者启发的创新方法，称为交通协调器网络(TRACO)。我们的方法利用了交叉注意和反事实优势函数的组合，允许我们提取领域代理的独特特征，并准确地量化团队需要从代理那里做出的贡献。通过在四个流量任务上的实验，我们证明了我们的方法优于现有的方法，产生了更好的性能。此外，我们的方法能够在交通流中的车辆之间出现丰富和多样化的社会行为。

**[Paper URL](https://proceedings.mlr.press/v229/liu23f.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23f/liu23f.pdf)** 

# Enabling Efficient, Reliable Real-World Reinforcement Learning with Approximate Physics-Based Models
**题目:** 利用基于物理的近似模型实现高效、可靠的现实世界强化学习

**作者:** Tyler Westenbroek, Jacob Levy, David Fridovich-Keil

**Abstract:** We focus on developing efficient and reliable policy optimization strategies for robot learning with real-world data.  In recent years, policy gradient methods have emerged as a promising paradigm for training control policies in simulation.  However, these approaches often remain too data inefficient or unreliable to train on real robotic hardware. In this paper we introduce a novel policy gradient-based policy optimization framework which systematically leverages a (possibly highly simplified) first-principles model and enables learning precise control policies with limited amounts of real-world data. Our approach $1)$ uses the derivatives of the model to produce sample-efficient estimates of the policy gradient and $2)$ uses the model to design a low-level tracking controller, which is embedded in the policy class. Theoretical analysis provides insight into how the presence of this feedback controller addresses overcomes key limitations of stand-alone policy gradient methods, while hardware experiments with a small car and quadruped demonstrate that our approach can learn precise control strategies reliably and with only minutes of real-world data.

**摘要:** 我们专注于利用真实世界的数据为机器人学习开发高效和可靠的策略优化策略。近年来，策略梯度方法已成为仿真中训练控制策略的一种很有前途的范例。然而，这些方法往往过于数据低效或不可靠，无法在真正的机器人硬件上进行训练。在本文中，我们介绍了一种新的基于策略梯度的策略优化框架，该框架系统地利用了(可能高度简化的)第一原理模型，并且能够利用有限的真实世界数据来学习精确的控制策略。我们的方法$1)$使用该模型的导数来产生策略梯度的样本有效估计，并且$2)$使用该模型来设计嵌入在策略类中的低级跟踪控制器。理论分析揭示了这种反馈控制器的存在如何克服独立策略梯度方法的关键限制，而小型汽车和四足动物的硬件实验表明，我们的方法可以可靠地学习精确的控制策略，并且只需几分钟的真实世界数据。

**[Paper URL](https://proceedings.mlr.press/v229/westenbroek23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/westenbroek23a/westenbroek23a.pdf)** 

# Large Language Models as General Pattern Machines
**题目:** 作为通用模式机的大型语言模型

**作者:** Suvir Mirchandani, Fei Xia, Pete Florence, Brian Ichter, Danny Driess, Montserrat Gonzalez Arenas, Kanishka Rao, Dorsa Sadigh, Andy Zeng

**Abstract:** We observe that pre-trained large language models (LLMs) are capable of autoregressively completing complex token sequences–from arbitrary ones procedurally generated by probabilistic context-free grammars (PCFG), to more rich spatial patterns found in the Abstraction and Reasoning Corpus (ARC), a general AI benchmark, prompted in the style of ASCII art. Surprisingly, pattern completion proficiency can be partially retained even when the sequences are expressed using tokens randomly sampled from the vocabulary. These results suggest that without any additional training, LLMs can serve as general sequence modelers, driven by in-context learning. In this work, we investigate how these zero-shot capabilities may be applied to problems in robotics–from extrapolating sequences of numbers that represent states over time to complete simple motions, to least-to-most prompting of reward-conditioned trajectories that can discover and represent closed-loop policies (e.g., a stabilizing controller for CartPole). While difficult to deploy today for real systems due to latency, context size limitations, and compute costs, the approach of using LLMs to drive low-level control may provide an exciting glimpse into how the patterns among words could be transferred to actions.

**摘要:** 我们观察到，预先训练的大型语言模型(LLM)能够自动回归地完成复杂的标记序列-从由概率上下文无关文法(PCFG)程序生成的任意序列，到以ASCII ART风格提示的通用人工智能基准语料库(ARC)中发现的更丰富的空间模式。令人惊讶的是，即使使用从词汇表中随机抽样的令牌来表示序列，模式补全的熟练程度也可以部分保持。这些结果表明，在没有任何额外训练的情况下，LLMS可以作为一般的序列建模器，由情境学习驱动。在这项工作中，我们研究了如何将这些零射击能力应用于机器人学中的问题-从外推表示随时间变化的状态的数字序列以完成简单的运动，到从最少到最多地提示可以发现和表示闭环系统策略的奖励条件轨迹(例如，CartPole的稳定控制器)。尽管由于延迟、上下文大小限制和计算成本的原因，目前很难为实际系统部署，但使用LLMS驱动低级控制的方法可能会提供一个令人兴奋的一瞥，了解如何将单词之间的模式转换为动作。

**[Paper URL](https://proceedings.mlr.press/v229/mirchandani23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mirchandani23a/mirchandani23a.pdf)** 

# One-shot Imitation Learning via Interaction Warping
**题目:** 通过交互扭曲进行一次性模仿学习

**作者:** Ondrej Biza, Skye Thompson, Kishore Reddy Pagidi, Abhinav Kumar, Elise van der Pol, Robin Walters, Thomas Kipf, Jan-Willem van de Meent, Lawson L. S. Wong, Robert Platt

**Abstract:** Learning robot policies from few demonstrations is crucial in open-ended applications. We propose a new method, Interaction Warping, for one-shot learning SE(3) robotic manipulation policies. We infer the 3D mesh of each object in the environment using shape warping, a technique for aligning point clouds across object instances. Then, we represent manipulation actions as keypoints on objects, which can be warped with the shape of the object. We show successful one-shot imitation learning on three simulated and real-world object re-arrangement tasks. We also demonstrate the ability of our method to predict object meshes and robot grasps in the wild. Webpage: https://shapewarping.github.io.

**摘要:** 在开放式应用程序中，从少数演示中学习机器人政策至关重要。我们提出了一种新方法“交互扭曲”，用于一次性学习SE（3）机器人操纵策略。我们使用形状扭曲来推断环境中每个对象的3D网格，这是一种用于在对象实例之间对齐点云的技术。然后，我们将操纵动作表示为对象上的关键点，这些关键点可以随着对象的形状而变形。我们在三个模拟和现实世界的对象重新排列任务上展示了成功的一次性模仿学习。我们还展示了我们的方法预测对象网格和机器人野外抓取的能力。网页：https://shapewarping.github.io。

**[Paper URL](https://proceedings.mlr.press/v229/biza23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/biza23a/biza23a.pdf)** 

# Learning to See Physical Properties with Active Sensing Motor Policies
**题目:** 学习通过主动传感电机策略查看物理特性

**作者:** Gabriel B. Margolis, Xiang Fu, Yandong Ji, Pulkit Agrawal

**Abstract:** To plan efficient robot locomotion, we must use the information about a terrain’s physics that can be inferred from color images. To this end, we train a visual perception module that predicts terrain properties using labels from a small amount of real-world proprioceptive locomotion. To ensure label precision, we introduce Active Sensing Motor Policies (ASMP). These policies are trained to prefer motor skills that facilitate accurately estimating the environment’s physics, like swiping a foot to observe friction. The estimated labels supervise a vision model that infers physical properties directly from color images and can be reused for different tasks. Leveraging a pretrained vision backbone, we demonstrate robust generalization in image space, enabling path planning from overhead imagery despite using only ground camera images for training.

**摘要:** 为了规划高效的机器人移动，我们必须使用可以从彩色图像推断的地形物理信息。为此，我们训练了一个视觉感知模块，该模块使用来自少量现实世界的主体感受运动的标签来预测地形属性。为了确保标签精度，我们引入了主动传感电机政策（ASMP）。这些政策经过培训，更喜欢有助于准确估计环境物理的运动技能，例如滑动脚来观察摩擦力。估计的标签监督视觉模型，该模型直接从彩色图像中推断物理性质，并可重复用于不同的任务。利用预先训练的视觉主干，我们在图像空间中展示了鲁棒的概括性，即使仅使用地面摄像机图像进行训练，也可以根据头顶图像进行路径规划。

**[Paper URL](https://proceedings.mlr.press/v229/margolis23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/margolis23a/margolis23a.pdf)** 

# General In-hand Object Rotation with Vision and Touch
**题目:** 通过视觉和触摸进行一般手持物体旋转

**作者:** Haozhi Qi, Brent Yi, Sudharshan Suresh, Mike Lambeta, Yi Ma, Roberto Calandra, Jitendra Malik

**Abstract:** We introduce Rotateit, a system that enables fingertip-based object rotation along multiple axes by leveraging multimodal sensory inputs. Our system is trained in simulation, where it has access to ground-truth object shapes and physical properties. Then we distill it to operate on realistic yet noisy simulated visuotactile and proprioceptive sensory inputs. These multimodal inputs are fused via a visuotactile transformer, enabling online inference of object shapes and physical properties during deployment. We show significant performance improvements over prior methods and highlight the importance of visual and tactile sensing.

**摘要:** 我们引入Rotateit，这是一个通过利用多模式感官输入来实现基于指尖的物体沿着多个轴旋转的系统。我们的系统经过模拟训练，可以访问地面真实物体形状和物理属性。然后我们对其进行提取，以处理现实但有噪音的模拟视觉触觉和主体感觉输入。这些多模式输入通过视觉Transformer融合，从而在部署期间在线推断对象形状和物理属性。与先前的方法相比，我们表现出了显着的性能改进，并强调了视觉和触觉传感的重要性。

**[Paper URL](https://proceedings.mlr.press/v229/qi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/qi23a/qi23a.pdf)** 

# Imitating Task and Motion Planning with Visuomotor Transformers
**题目:** 使用可视化变形金刚模仿任务和运动规划

**作者:** Murtaza Dalal, Ajay Mandlekar, Caelan Reed Garrett, Ankur Handa, Ruslan Salakhutdinov, Dieter Fox

**Abstract:** Imitation learning is a powerful tool for training robot manipulation policies, allowing them to learn from expert demonstrations without manual programming or trial-and-error. However, common methods of data collection, such as human supervision, scale poorly, as they are time-consuming and labor-intensive. In contrast, Task and Motion Planning (TAMP) can autonomously generate large-scale datasets of diverse demonstrations. In this work, we show that the combination of large-scale datasets generated by TAMP supervisors and flexible Transformer models to fit them is a powerful paradigm for robot manipulation. We present a novel imitation learning system called OPTIMUS that trains large-scale visuomotor Transformer policies by imitating a TAMP agent. We conduct a thorough study of the design decisions required to imitate TAMP and demonstrate that OPTIMUS can solve a wide variety of challenging vision-based manipulation tasks with over 70 different objects, ranging from long-horizon pick-and-place tasks, to shelf and articulated object manipulation, achieving $70$ to $80%$ success rates. Video results and code at https://mihdalal.github.io/optimus/

**摘要:** 模仿学习是训练机器人操作策略的强大工具，允许他们从专家演示中学习，而无需手动编程或反复试验。然而，常见的数据收集方法，如人工监督，可伸缩性差，因为它们耗时和劳动密集型。相比之下，任务和运动规划(TAMP)可以自动生成不同演示的大规模数据集。在这项工作中，我们展示了由夯实监督员生成的大规模数据集与灵活的变压器模型相结合来适应它们是一种强大的机器人操作范例。我们提出了一个新的模拟学习系统，称为擎天柱，它通过模拟一个捣固剂来训练大规模的视觉运动转换器策略。我们对模拟夯实所需的设计决策进行了深入的研究，并展示了擎天柱可以针对70多个不同的对象解决各种具有挑战性的基于视觉的操纵任务，从长视距拾取和放置任务到货架和铰接式对象操纵，实现了70%到80%的成功率。Https://mihdalal.github.io/optimus/上的视频结果和代码

**[Paper URL](https://proceedings.mlr.press/v229/dalal23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/dalal23a/dalal23a.pdf)** 

# Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks
**题目:** 好奇心驱动的关节运动和操纵任务学习

**作者:** Clemens Schwarke, Victor Klemm, Matthijs van der Boon, Marko Bjelonic, Marco Hutter

**Abstract:** Learning complex locomotion and manipulation tasks presents significant challenges, often requiring extensive engineering of, e.g., reward functions or curricula to provide meaningful feedback to the Reinforcement Learning (RL) algorithm. This paper proposes an intrinsically motivated RL approach to reduce task-specific engineering. The desired task is encoded in a single sparse reward, i.e., a reward of “+1" is given if the task is achieved. Intrinsic motivation enables learning by guiding exploration toward the sparse reward signal. Specifically, we adapt the idea of Random Network Distillation (RND) to the robotics domain to learn holistic motion control policies involving simultaneous locomotion and manipulation. We investigate opening doors as an exemplary task for robotic ap- plications. A second task involving package manipulation from a table to a bin highlights the generalization capabilities of the presented approach. Finally, the resulting RL policies are executed in real-world experiments on a wheeled-legged robot in biped mode. We experienced no failure in our experiments, which consisted of opening push doors (over 15 times in a row) and manipulating packages (over 5 times in a row).

**摘要:** 学习复杂的运动和操纵任务是一个巨大的挑战，通常需要对奖励函数或课程进行广泛的工程设计，以向强化学习(RL)算法提供有意义的反馈。本文提出了一种内在激励的RL方法，以减少特定于任务的工程。期望的任务被编码在单个稀疏奖励中，即，如果任务完成，则给予“+1”奖励。内在动机通过引导对稀疏奖励信号的探索而使学习成为可能。具体地说，我们将随机网络蒸馏(RND)的思想应用到机器人领域，以学习涉及同时运动和操作的整体运动控制策略。我们将开门作为机器人应用的一个典型任务进行研究。第二个任务涉及从表到箱的包操作，突出了所提出的方法的泛化能力。最后，将得到的RL策略在两足模式下的轮腿机器人上进行了真实世界的实验。在我们的实验中，我们没有经历过失败，包括打开推门(连续15次以上)和操纵包裹(连续5次以上)。

**[Paper URL](https://proceedings.mlr.press/v229/schwarke23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/schwarke23a/schwarke23a.pdf)** 

# Towards Scalable Coverage-Based Testing of Autonomous Vehicles
**题目:** 迈向可扩展的基于覆盖率的自动驾驶车辆测试

**作者:** James Tu, Simon Suo, Chris Zhang, Kelvin Wong, Raquel Urtasun

**Abstract:** To deploy autonomous vehicles(AVs) in the real world, developers must understand the conditions in which the system can operate safely. To do this in a scalable manner, AVs are often tested in simulation on parameterized scenarios. In this context, it’s important to build a testing framework that partitions the scenario parameter space into safe, unsafe, and unknown regions. Existing approaches rely on discretizing continuous parameter spaces into bins, which scales poorly to high-dimensional spaces and cannot describe regions with arbitrary shape. In this work, we introduce a problem formulation which avoids discretization – by modeling the probability of meeting safety requirements everywhere, the parameter space can be paritioned using a probability threshold. Based on our formulation, we propose GUARD as a testing framework which leverages Gaussian Processes to model probability and levelset algorithms to efficiently generate tests. Moreover, we introduce a set of novel evaluation metrics for coverage-based testing frameworks to capture the key objectives of testing. In our evaluation suite of diverse high-dimensional scenarios, GUARD significantly outperforms existing approaches. By proposing an efficient, accurate, and scalable testing framework, our work is a step towards safely deploying autonomous vehicles at scale.

**摘要:** 要在现实世界中部署自动驾驶汽车(AV)，开发人员必须了解系统可以安全运行的条件。为了以可扩展的方式做到这一点，AV通常在参数化场景的模拟中进行测试。在这种情况下，重要的是构建一个测试框架，将场景参数空间划分为安全、不安全和未知区域。现有的方法依赖于将连续的参数空间离散成箱，这种方法对高维空间的伸缩性很差，并且不能描述任意形状的区域。在这项工作中，我们引入了一种避免离散化的问题描述-通过对到处满足安全要求的概率进行建模，可以使用概率阈值来划分参数空间。基于我们的公式，我们提出了Guard作为一个测试框架，它利用高斯过程对概率和水平集算法进行建模，从而有效地生成测试。此外，我们还为基于覆盖的测试框架引入了一套新的评估指标，以捕捉测试的关键目标。在我们对各种高维场景的评估套件中，Guard的性能明显优于现有方法。通过提出一个高效、准确和可扩展的测试框架，我们的工作是朝着安全地大规模部署自动驾驶汽车迈出了一步。

**[Paper URL](https://proceedings.mlr.press/v229/tu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tu23a/tu23a.pdf)** 

# PLEX: Making the Most of the Available Data for Robotic Manipulation Pretraining
**题目:** PFLEX：充分利用可用数据进行机器人操纵预训练

**作者:** Garrett Thomas, Ching-An Cheng, Ricky Loynd, Felipe Vieira Frujeri, Vibhav Vineet, Mihai Jalobeanu, Andrey Kolobov

**Abstract:** A rich representation is key to general robotic manipulation, but existing approaches to representation learning require large amounts of multimodal demonstrations. In this work we propose PLEX, a transformer-based architecture that learns from a small amount of task-agnostic visuomotor trajectories and a much larger amount of task-conditioned object manipulation videos – a type of data available in quantity. PLEX uses visuomotor trajectories to induce a latent feature space and to learn task-agnostic manipulation routines, while diverse video-only demonstrations teach PLEX how to plan in the induced latent feature space for a wide variety of tasks. Experiments showcase PLEX’s generalization on Meta-World and SOTA performance in challenging Robosuite environments. In particular, using relative positional encoding in PLEX’s transformers greatly helps in low-data regimes of learning from human-collected demonstrations.

**摘要:** 丰富的表示是一般机器人操作的关键，但现有的表示学习方法需要大量的多模式演示。在这项工作中，我们提出了PFLEX，这是一种基于转换器的架构，可以从少量任务不可知的可视化轨迹和大量任务条件下的对象操纵视频中学习--一种大量可用的数据类型。PFLEX使用视觉轨迹来诱导潜在特征空间并学习任务不可知的操纵例程，而各种纯视频演示则教PFLEX如何在诱导的潜在特征空间中为各种任务进行规划。实验展示了PFLEX在具有挑战性的机器人套件环境中对元世界和SOTA性能的概括。特别是，在PFLEX的转换器中使用相对位置编码对于从人类收集的演示中学习的低数据环境非常有帮助。

**[Paper URL](https://proceedings.mlr.press/v229/thomas23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/thomas23a/thomas23a.pdf)** 

# Learning Lyapunov-Stable Polynomial Dynamical Systems Through Imitation
**题目:** 通过模仿学习Lyapunov稳定的多项动力系统

**作者:** Amin Abyaneh, Hsiu-Chin Lin

**Abstract:** Imitation learning is a paradigm to address complex motion planning problems by learning a policy to imitate an expert’s behavior. However, relying solely on the expert’s data might lead to unsafe actions when the robot deviates from the demonstrated trajectories. Stability guarantees have previously been provided utilizing nonlinear dynamical systems, acting as high-level motion planners, in conjunction with the Lyapunov stability theorem. Yet, these methods are prone to inaccurate policies, high computational cost, sample inefficiency, or quasi stability when replicating complex and highly nonlinear trajectories. To mitigate this problem, we present an approach for learning a globally stable nonlinear dynamical system as a motion planning policy. We model the nonlinear dynamical system as a parametric polynomial and learn the polynomial’s coefficients jointly with a Lyapunov candidate. To showcase its success, we compare our method against the state of the art in simulation and conduct real-world experiments with the Kinova Gen3 Lite manipulator arm. Our experiments demonstrate the sample efficiency and reproduction accuracy of our method for various expert trajectories, while remaining stable in the face of perturbations.

**摘要:** 模仿学习是一种通过学习模仿专家行为的策略来解决复杂运动规划问题的范例。然而，当机器人偏离演示的轨迹时，单纯依赖专家的数据可能会导致不安全的行为。以前已经利用非线性动力系统作为高级运动规划器，结合李亚普诺夫稳定性定理来提供稳定性保证。然而，这些方法在复制复杂和高度非线性的轨迹时，容易出现策略不准确、计算成本高、样本效率低或准稳定的问题。为了缓解这一问题，我们提出了一种学习全局稳定的非线性动力系统的方法作为运动规划策略。我们将非线性动力系统建模为参数多项式，并与Lyapunov候选者联合学习多项式的系数。为了展示它的成功，我们将我们的方法与最先进的模拟技术进行了比较，并使用Kinova Gen3 Lite操作臂进行了真实世界的实验。我们的实验证明了我们的方法对于各种专家轨迹的样本效率和再现精度，同时在面对扰动时保持稳定。

**[Paper URL](https://proceedings.mlr.press/v229/abyaneh23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/abyaneh23a/abyaneh23a.pdf)** 

# MUTEX: Learning Unified Policies from Multimodal Task Specifications
**题目:** MUTEK：从多模式任务规范中学习统一策略

**作者:** Rutav Shah, Roberto Martín-Martín, Yuke Zhu

**Abstract:** Humans use different modalities, such as speech, text, images, videos, etc., to communicate their intent and goals with teammates. For robots to become better assistants, we aim to endow them with the ability to follow instructions and understand tasks specified by their human partners. Most robotic policy learning methods have focused on one single modality of task specification while ignoring the rich cross-modal information. We present MUTEX, a unified approach to policy learning from multimodal task specifications. It trains a transformer-based architecture to facilitate cross-modal reasoning, combining masked modeling and cross-modal matching objectives in a two-stage training procedure. After training, MUTEX can follow a task specification in any of the six learned modalities (video demonstrations, goal images, text goal descriptions, text instructions, speech goal descriptions, and speech instructions) or a combination of them. We systematically evaluate the benefits of MUTEX in a newly designed dataset with 100 tasks in simulation and 50 tasks in the real world, annotated with multiple instances of task specifications in different modalities, and observe improved performance over methods trained specifically for any single modality. More information at https://ut-austin-rpl.github.io/MUTEX/

**摘要:** 人类使用不同的方式，如语音、文本、图像、视频等，与队友交流他们的意图和目标。为了让机器人成为更好的助手，我们的目标是赋予它们遵循指令和理解人类伴侣指定的任务的能力。大多数机器人策略学习方法都集中在单一通道的任务指定上，而忽略了丰富的跨通道信息。我们提出了MUTEX，一种从多通道任务规范中学习策略的统一方法。它训练基于变压器的体系结构以促进跨模式推理，在两个阶段的训练过程中结合掩蔽建模和跨模式匹配目标。培训后，MUTEX可以遵循六种学习模式(视频演示、目标图像、文本目标描述、文本说明、演讲目标描述和演讲说明)中的任何一种或它们的组合中的任务规范。我们在一个新设计的数据集中系统地评估了MUTEX的好处，该数据集中有100个模拟任务和50个真实世界的任务，用不同模式下的任务规范的多个实例进行标注，并观察到与专门针对任何单一模式训练的方法相比，性能有所改善。欲了解更多信息，请访问https://ut-austin-rpl.github.io/MUTEX/。

**[Paper URL](https://proceedings.mlr.press/v229/shah23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shah23b/shah23b.pdf)** 

# Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning
**题目:** 使用大型语言模型导航：作为规划启发式的语义猜测

**作者:** Dhruv Shah, Michael Robert Equi, Błażej Osiński, Fei Xia, Brian Ichter, Sergey Levine

**Abstract:** Navigation in unfamiliar environments presents a major challenge for robots: while mapping and planning techniques can be used to build up a representation of the world, quickly discovering a path to a desired goal in unfamiliar settings with such methods often requires lengthy mapping and exploration. Humans can rapidly navigate new environments, particularly indoor environments that are laid out logically, by leveraging semantics — e.g., a kitchen often adjoins a living room, an exit sign indicates the way out, and so forth. Language models can provide robots with such knowledge, but directly using language models to instruct a robot how to reach some destination can also be impractical: while language models might produce a narrative about how to reach some goal, because they are not grounded in real-world observations, this narrative might be arbitrarily wrong. Therefore, in this paper we study how the “semantic guesswork” produced by language models can be utilized as a guiding heuristic for planning algorithms. Our method, Language Frontier Guide (LFG), uses the language model to bias exploration of novel real-world environments by incorporating the semantic knowledge stored in language models as a search heuristic for planning with either topological or metric maps. We evaluate LFG in challenging real-world environments and simulated benchmarks, outperforming uninformed exploration and other ways of using language models.

**摘要:** 在陌生环境中导航对机器人来说是一个重大挑战：虽然地图和规划技术可以用来建立世界的表征，但用这种方法在不熟悉的环境中快速发现通往期望目标的路径通常需要漫长的地图绘制和探索。人类可以通过利用语义快速导航新环境，特别是逻辑布局的室内环境--例如，厨房通常与客厅相邻，出口标志指示出口，等等。语言模型可以为机器人提供这样的知识，但直接使用语言模型来指导机器人如何到达某个目的地也是不切实际的：尽管语言模型可能会产生关于如何达到某个目标的叙事，因为它们不是基于现实世界的观察，但这种叙事可能是完全错误的。因此，在本文中，我们研究如何将语言模型产生的“语义猜测”用作规划算法的启发式指导。我们的方法Language Frontier Guide(LFG)使用语言模型来引导对新的现实世界环境的探索，方法是将存储在语言模型中的语义知识作为搜索启发式，用于使用拓扑或度量地图进行规划。我们在具有挑战性的真实世界环境和模拟基准中评估LFG，表现优于未经知情的探索和其他使用语言模型的方式。

**[Paper URL](https://proceedings.mlr.press/v229/shah23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/shah23c/shah23c.pdf)** 

# A Data-efficient Neural ODE Framework for Optimal Control of Soft Manipulators
**题目:** 软机械手最优控制的数据高效神经ODE框架

**作者:** Mohammadreza Kasaei, Keyhan Kouhkiloui Babarahmati, Zhibin Li, Mohsen Khadem

**Abstract:** This paper introduces a novel approach for modeling continuous forward kinematic models of soft continuum robots by employing Augmented Neural ODE (ANODE), a cutting-edge family of deep neural network models. To the best of our knowledge, this is the first application of ANODE in modeling soft continuum robots. This formulation introduces auxiliary dimensions, allowing the system’s states to evolve in the augmented space which provides a richer set of dynamics that the model can learn, increasing the flexibility and accuracy of the model. Our methodology achieves exceptional sample efficiency, training the continuous forward kinematic model using only 25 scattered data points. Additionally, we design and implement a fully parallel Model Predictive Path Integral (MPPI)-based controller running on a GPU, which efficiently manages a non-convex objective function. Through a set of experiments, we showed that the proposed framework (ANODE+MPPI) significantly outperforms state-of-the-art learning-based methods such as FNN and RNN in unseen-before scenarios and marginally outperforms them in seen-before scenarios.

**摘要:** 介绍了一种利用增广神经网络(ANODE)这一前沿的深度神经网络模型来建立柔性连续体机器人连续正运动学模型的新方法。据我们所知，这是首次将阳极应用于软连续体机器人的建模。这个公式引入了辅助维度，允许系统的状态在扩展空间中演变，这提供了模型可以学习的更丰富的动力学集，增加了模型的灵活性和准确性。我们的方法实现了出色的样本效率，只使用25个散乱的数据点来训练连续正向运动学模型。此外，我们还设计并实现了一种完全并行的基于模型预测路径积分(MPPI)的控制器，该控制器运行在GPU上，可以有效地管理非凸目标函数。通过一组实验，我们证明了所提出的框架(阳极+MPPI)在前所未见的场景中的性能显著优于基于学习的方法(如FNN和RNN)，在未见过的场景中的性能略优于它们。

**[Paper URL](https://proceedings.mlr.press/v229/kasaei23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kasaei23a/kasaei23a.pdf)** 

# Language Conditioned Traffic Generation
**题目:** 语言条件流量生成

**作者:** Shuhan Tan, Boris Ivanovic, Xinshuo Weng, Marco Pavone, Philipp Kraehenbuehl

**Abstract:** Simulation forms the backbone of modern self-driving development. Simulators help develop, test, and improve driving systems without putting humans, vehicles, or their environment at risk. However, simulators face a major challenge: They rely on realistic, scalable, yet interesting content. While recent advances in rendering and scene reconstruction make great strides in creating static scene assets, modeling their layout, dynamics, and behaviors remains challenging. In this work, we turn to language as a source of supervision for dynamic traffic scene generation. Our model, LCTGen, combines a large language model with a transformer-based decoder architecture that selects likely map locations from a dataset of maps, and produces an initial traffic distribution, as well as the dynamics of each vehicle. LCTGen outperforms prior work in both unconditional and conditional traffic scene generation in terms of realism and fidelity.

**摘要:** 模拟是现代自动驾驶开发的支柱。模拟器有助于开发、测试和改进驾驶系统，而不会将人类、车辆或其环境置于危险之中。然而，模拟器面临着一个重大挑战：它们依赖于现实、可扩展且有趣的内容。虽然渲染和场景重建方面的最新进展在创建静态场景资产方面取得了长足的进步，但对其布局、动态和行为进行建模仍然具有挑战性。在这项工作中，我们将语言作为动态交通场景生成的监督来源。我们的模型LCTGen将大型语言模型与基于转换器的解码器架构相结合，该架构从地图数据集中选择可能的地图位置，并生成初始交通分布以及每辆车辆的动态。LCTGen在无条件和有条件交通场景生成方面都优于之前的工作。

**[Paper URL](https://proceedings.mlr.press/v229/tan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tan23a/tan23a.pdf)** 

# CALAMARI: Contact-Aware and Language conditioned spatial Action MApping for contact-RIch manipulation
**题目:** CALAMARI：接触感知和语言条件化的空间动作MAPING用于接触Ritch操纵

**作者:** Youngsun Wi, Mark Van der Merwe, Pete Florence, Andy Zeng, Nima Fazeli

**Abstract:** Making contact with purpose is a central part of robot manipulation and remains essential for many household tasks – from sweeping dust into a dustpan, to wiping tables; from erasing whiteboards, to applying paint. In this work, we investigate learning language-conditioned, vision-based manipulation policies wherein the action representation is in fact, contact itself – predicting contact formations at which tools grasped by the robot should meet an observable surface. Our approach, Contact-Aware and Language conditioned spatial Action MApping for contact-RIch manipulation (CALAMARI), exhibits several advantages including (i) benefiting from existing visual-language models for pretrained spatial features, grounding instructions to behaviors, and for sim2real transfer; and (ii) factorizing perception and control over a natural boundary (i.e. contact) into two modules that synergize with each other, whereby action predictions can be aligned per pixel with image observations, and low-level controllers can optimize motion trajectories that maintain contact while avoiding penetration. Experiments show that CALAMARI outperforms existing state-of-the-art model architectures for a broad range of contact-rich tasks, and pushes new ground on embodiment-agnostic generalization to unseen objects with varying elasticity, geometry, and colors in both simulated and real-world settings.

**摘要:** 与目标的接触是机器人操作的核心部分，对于许多家务活来说，这仍然是必不可少的--从把灰尘扫进垃圾桶，到擦拭桌子；从擦白板到涂抹油漆。在这项工作中，我们研究了学习语言制约的、基于视觉的操作策略，其中动作表示实际上是接触本身-预测机器人抓住的工具应该与可观察表面相遇的接触队形。我们的方法，接触感知和语言条件空间动作映射，用于接触丰富的操作(Calamari)，展示了几个优点，包括：(I)受益于现有的视觉语言模型，用于预先训练的空间特征，对行为的基础指令，以及用于简单真实的转移；以及(Ii)将对自然边界(即接触)的感知和控制分解为两个相互协同的模块，从而可以将动作预测与图像观察对齐，并且低级控制器可以优化运动轨迹，在保持接触的同时避免渗透。实验表明，Calamari在广泛的接触丰富任务方面的表现优于现有的最先进的模型架构，并将与实施例无关的泛化推向了在模拟和真实世界设置中具有不同弹性、几何形状和颜色的不可见对象。

**[Paper URL](https://proceedings.mlr.press/v229/wi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wi23a/wi23a.pdf)** 

# Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities
**题目:** 通过能力感知和沟通推广异类多机器人策略

**作者:** Pierce Howell, Max Rudolph, Reza Joseph Torbati, Kevin Fu, Harish Ravichandar

**Abstract:** Recent advances in multi-agent reinforcement learning (MARL) are enabling impressive coordination in heterogeneous multi-robot teams. However, existing approaches often overlook the challenge of generalizing learned policies to teams of new compositions, sizes, and robots. While such generalization might not be important in teams of virtual agents that can retrain policies on-demand, it is pivotal in multi-robot systems that are deployed in the real-world and must readily adapt to inevitable changes. As such, multi-robot policies must remain robust to team changes – an ability we call adaptive teaming. In this work, we investigate if awareness and communication of robot capabilities can provide such generalization by conducting detailed experiments involving an established multi-robot test bed. We demonstrate that shared decentralized policies, that enable robots to be both aware of and communicate their capabilities, can achieve adaptive teaming by implicitly capturing the fundamental relationship between collective capabilities and effective coordination. Videos of trained policies can be viewed at https://sites.google.com/view/cap-comm .

**摘要:** 多智能体强化学习(MAIL)的最新进展使异质多机器人团队能够进行令人印象深刻的协调。然而，现有的方法往往忽略了将学习的策略推广到新组成、新规模和新机器人的团队的挑战。虽然这种泛化在可以按需重新培训策略的虚拟代理团队中可能并不重要，但在部署在现实世界中并且必须随时适应不可避免的变化的多机器人系统中是关键的。因此，多机器人策略必须保持对团队变化的健壮--我们称之为自适应团队能力。在这项工作中，我们通过在建立的多机器人试验台上进行详细的实验，调查对机器人能力的感知和沟通是否可以提供这种推广。我们证明了共享的分散策略，使机器人能够感知和交流它们的能力，可以通过隐含地捕捉集体能力和有效协调之间的基本关系来实现自适应组队。培训政策的视频可以在https://sites.google.com/view/cap-comm上观看。

**[Paper URL](https://proceedings.mlr.press/v229/howell23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/howell23a/howell23a.pdf)** 

# CAJun: Continuous Adaptive Jumping using a Learned Centroidal Controller
**题目:** CAJun：使用习得的重心控制器进行连续自适应跳跃

**作者:** Yuxiang Yang, Guanya Shi, Xiangyun Meng, Wenhao Yu, Tingnan Zhang, Jie Tan, Byron Boots

**Abstract:** We present CAJun, a novel hierarchical learning and control framework that enables legged robots to jump continuously with adaptive jumping distances. CAJun consists of a high-level centroidal policy and a low-level leg controller. In particular, we use reinforcement learning (RL) to train the centroidal policy, which specifies the gait timing, base velocity, and swing foot position for the leg controller. The leg controller optimizes motor commands for the swing and stance legs according to the gait timing to track the swing foot target and base velocity commands. Additionally, we reformulate the stance leg optimizer in the leg controller to speed up policy training by an order of magnitude. Our system combines the versatility of learning with the robustness of optimal control. We show that after 20 minutes of training on a single GPU, CAJun can achieve continuous, long jumps with adaptive distances on a Go1 robot with small sim-to-real gaps. Moreover, the robot can jump across gaps with a maximum width of 70cm, which is over $40%$ wider than existing methods.

**摘要:** 提出了一种新颖的分层学习和控制框架CAJUN，它使腿部机器人能够以自适应的跳跃距离连续跳跃。CAJUN由高级质心策略和低级LEG控制器组成。特别是，我们使用强化学习(RL)来训练质心策略，该策略指定腿部控制器的步态定时、基本速度和摆动脚位置。腿控制器根据步态定时优化摆动和站立腿的马达命令，以跟踪摆动脚目标和基本速度命令。此外，我们在腿部控制器中重新制定了立场腿部优化器，以将策略训练速度提高一个数量级。我们的系统结合了学习的多功能性和最优控制的稳健性。结果表明，在单个GPU上训练20分钟后，Cajun可以在GO1机器人上实现连续的、具有自适应距离的跳远，并且模拟与真实的差距很小。此外，该机器人可以跳过最大宽度为70厘米的缝隙，比现有方法宽40%美元以上。

**[Paper URL](https://proceedings.mlr.press/v229/yang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23b/yang23b.pdf)** 

# Multi-Predictor Fusion: Combining Learning-based and Rule-based Trajectory Predictors
**题目:** 多预测器融合：结合基于学习和基于规则的轨迹预测器

**作者:** Sushant Veer, Apoorva Sharma, Marco Pavone

**Abstract:** Trajectory prediction modules are key enablers for safe and efficient planning of autonomous vehicles (AVs), particularly in highly interactive traffic scenarios. Recently, learning-based trajectory predictors have experienced considerable success in providing state-of-the-art performance due to their ability to learn multimodal behaviors of other agents from data. In this paper, we present an algorithm called multi-predictor fusion (MPF) that augments the performance of learning-based predictors by imbuing them with motion planners that are tasked with satisfying logic-based rules. MPF probabilistically combines learning- and rule-based predictors by mixing trajectories from both standalone predictors in accordance with a belief distribution that reflects the online performance of each predictor. In our results, we show that MPF outperforms the two standalone predictors on various metrics and delivers the most consistent performance.

**摘要:** 轨迹预测模块是安全有效规划自动驾驶车辆（AV）的关键推动因素，特别是在高度交互的交通场景中。最近，基于学习的轨迹预测器在提供最先进的性能方面取得了相当大的成功，因为它们能够从数据中学习其他智能体的多模式行为。在本文中，我们提出了一种名为多预测器融合（MPF）的算法，该算法通过为基于学习的预测器注入负责满足基于逻辑的规则的运动规划器来增强基于学习的预测器的性能。MPF通过根据反映每个预测器在线表现的信念分布混合来自两个独立预测器的轨迹，概率地组合了基于学习的预测器和基于规则的预测器。在我们的结果中，我们表明MPF在各种指标上优于两个独立预测器，并提供最一致的性能。

**[Paper URL](https://proceedings.mlr.press/v229/veer23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/veer23a/veer23a.pdf)** 

# Neural Field Dynamics Model for Granular Object Piles Manipulation
**题目:** 颗粒物桩操纵的神经场动力学模型

**作者:** Shangjie Xue, Shuo Cheng, Pujith Kachana, Danfei Xu

**Abstract:** We present a learning-based dynamics model for granular material manipulation. Drawing inspiration from computer graphics’ Eulerian approach, our method adopts a fully convolutional neural network that operates on a density field-based representation of object piles, allowing it to exploit the spatial locality of inter-object interactions through the convolution operations. This approach greatly improves the learning and computation efficiency compared to existing latent or particle-based methods and sidesteps the need for state estimation, making it directly applicable to real-world settings. Furthermore, our differentiable action rendering module makes the model fully differentiable and can be directly integrated with a gradient-based algorithm for curvilinear trajectory optimization. We evaluate our model with a wide array of piles manipulation tasks both in simulation and real-world experiments and demonstrate that it significantly exceeds existing methods in both accuracy and computation efficiency. More details can be found at https://sites.google.com/view/nfd-corl23/

**摘要:** 提出了一种基于学习的颗粒物料操纵动力学模型。受计算机图形学欧拉方法的启发，我们的方法采用了基于密度场的对象堆表示的完全卷积神经网络，允许它通过卷积运算来利用对象间交互的空间局部性。与现有的潜在或基于粒子的方法相比，该方法极大地提高了学习和计算效率，并避免了对状态估计的需要，使其直接适用于真实世界的设置。此外，我们的可微动作渲染模块使模型完全可微，并且可以直接与基于梯度的曲线轨迹优化算法集成。我们在模拟和真实世界的实验中对我们的模型进行了评估，结果表明，该模型在精度和计算效率上都明显优于现有的方法。欲了解更多详情，请访问https://sites.google.com/view/nfd-corl23/。

**[Paper URL](https://proceedings.mlr.press/v229/xue23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xue23a/xue23a.pdf)** 

# AR2-D2: Training a Robot Without a Robot
**题目:** AR 2-D2：在没有机器人的情况下训练机器人

**作者:** Jiafei Duan, Yi Ru Wang, Mohit Shridhar, Dieter Fox, Ranjay Krishna

**Abstract:** Diligently gathered human demonstrations serve as the unsung heroes empowering the progression of robot learning. Today, demonstrations are collected by training people to use specialized controllers, which (tele-)operate robots to manipulate a small number of objects. By contrast, we introduce AR2-D2: a system for collecting demonstrations which (1) does not require people with specialized training, (2) does not require any real robots during data collection, and therefore, (3) enables manipulation of diverse objects with a real robot. AR2-D2 is a framework in the form of an iOS app that people can use to record a video of themselves manipulating any object while simultaneously capturing essential data modalities for training a real robot. We show that data collected via our system enables the training of behavior cloning agents in manipulating real objects. Our experiments further show that training with our AR data is as effective as training with real-world robot demonstrations. Moreover, our user study indicates that users find AR2-D2 intuitive to use and require no training in contrast to four other frequently employed methods for collecting robot demonstrations.

**摘要:** 勤奋收集的人类示范充当了无名英雄，推动了机器人学习的进步。今天，通过培训人们使用专门的控制器来收集演示，这些控制器(远程)操作机器人来操纵少量物体。相反，我们介绍了AR2-D2：一种用于收集演示的系统，它(1)不需要经过专门培训的人，(2)在数据收集期间不需要任何真实的机器人，因此，(3)能够使用真实的机器人操纵不同的对象。AR2-D2是一个iOS应用程序形式的框架，人们可以用它来记录自己操作任何对象的视频，同时捕捉训练真实机器人所需的必要数据。我们表明，通过我们的系统收集的数据能够训练行为克隆代理操纵真实对象。我们的实验进一步表明，使用我们的AR数据进行训练与使用真实世界的机器人演示进行训练一样有效。此外，我们的用户研究表明，与其他四种经常使用的收集机器人演示的方法相比，用户发现AR2-D2使用起来很直观，不需要培训。

**[Paper URL](https://proceedings.mlr.press/v229/duan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/duan23a/duan23a.pdf)** 

# Affordance-Driven Next-Best-View Planning for Robotic Grasping
**题目:** 经济实惠驱动的机器人抓取的下一个最佳视图规划

**作者:** Xuechao Zhang, Dong Wang, Sun Han, Weichuang Li, Bin Zhao, Zhigang Wang, Xiaoming Duan, Chongrong Fang, Xuelong Li, Jianping He

**Abstract:** Grasping occluded objects in cluttered environments is an essential component in complex robotic manipulation tasks. In this paper, we introduce an AffordanCE-driven Next-Best-View planning policy (ACE-NBV) that tries to find a feasible grasp for target object via continuously observing scenes from new viewpoints. This policy is motivated by the observation that the grasp affordances of an occluded object can be better-measured under the view when the view-direction are the same as the grasp view. Specifically, our method leverages the paradigm of novel view imagery to predict the grasps affordances under previously unobserved view, and select next observation view based on the highest imagined grasp quality of the target object. The experimental results in simulation and on a real robot demonstrate the effectiveness of the proposed affordance-driven next-best-view planning policy. Project page: https://sszxc.net/ace-nbv/.

**摘要:** 在混乱的环境中抓取被遮挡的物体是复杂机器人操纵任务的重要组成部分。在本文中，我们介绍了一种由AffordanCE驱动的下一个最佳视图规划政策（ACE-NBV），该政策试图通过从新的角度持续观察场景来找到对目标对象的可行把握。该政策的动机是这样一种观察：当视图方向与抓取视图相同时，可以在视图下更好地测量被遮挡物体的抓取可供性。具体来说，我们的方法利用新视图图像的范式来预测之前未观察到的视图下的抓取可供性，并根据目标对象的最高想象抓取质量选择下一个观察视图。模拟和真实机器人上的实验结果证明了所提出的负担能力驱动的次佳视图规划政策的有效性。项目页面：https://sszxc.net/ace-nbv/。

**[Paper URL](https://proceedings.mlr.press/v229/zhang23i.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23i/zhang23i.pdf)** 

# PairwiseNet: Pairwise Collision Distance Learning for High-dof Robot Systems
**题目:** PairwiseNet：高自由度机器人系统的成对碰撞距离学习

**作者:** Jihwan Kim, Frank C. Park

**Abstract:** Motion planning for robot manipulation systems operating in complex environments remains a challenging problem. It requires the evaluation of both the collision distance and its derivative. Owing to its computational complexity, recent studies have attempted to utilize data-driven approaches to learn the collision distance. However, their performance degrades significantly for complicated high-dof systems, such as multi-arm robots. Additionally, the model must be retrained every time the environment undergoes even slight changes. In this paper, we propose PairwiseNet, a model that estimates the minimum distance between two geometric shapes and overcomes many of the limitations of current models. By dividing the problem of global collision distance learning into smaller pairwise sub-problems, PairwiseNet can be used to efficiently calculate the global collision distance. PairwiseNet can be deployed without further modifications or training for any system comprised of the same shape elements (as those in the training dataset). Experiments with multi-arm manipulation systems of various dof indicate that our model achieves significant performance improvements concerning several performance metrics, especially the false positive rate with the collision-free guaranteed threshold. Results further demonstrate that our single trained PairwiseNet model is applicable to all multi-arm systems used in the evaluation. The code is available at https://github.com/kjh6526/PairwiseNet.

**摘要:** 在复杂环境中运行的机器人操作系统的运动规划仍然是一个具有挑战性的问题。它需要计算碰撞距离及其导数。由于其计算的复杂性，最近的研究试图利用数据驱动的方法来学习碰撞距离。然而，对于复杂的高自由度系统，如多臂机器人，它们的性能会显著下降。此外，每次环境发生微小变化时，都必须对模型进行重新培训。在本文中，我们提出了Pairise Net，这是一个估计两个几何形状之间的最小距离的模型，克服了现有模型的许多局限性。通过将全局碰撞距离学习问题分解为更小的两两子问题，PairwieNet可用于高效地计算全局碰撞距离。对于由相同形状元素(与训练数据集中的那些形状元素)组成的任何系统，无需进一步修改或训练即可部署PairwieNet。在不同DOF的多臂操作系统上的实验表明，我们的模型在几个性能指标上都有显著的提高，特别是在无碰撞保证阈值下的误检率。结果进一步表明，我们的单一训练的PairwieNet模型适用于评估中使用的所有多臂系统。代码可在https://github.com/kjh6526/PairwiseNet.上获得

**[Paper URL](https://proceedings.mlr.press/v229/kim23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kim23d/kim23d.pdf)** 

# Fighting Uncertainty with Gradients: Offline Reinforcement Learning via Diffusion Score Matching
**题目:** 与学生对抗不确定性：通过扩散分数匹配的离线强化学习

**作者:** H.J. Terry Suh, Glen Chou, Hongkai Dai, Lujie Yang, Abhishek Gupta, Russ Tedrake

**Abstract:** Gradient-based methods enable efficient search capabilities in high dimensions. However, in order to apply them effectively in offline optimization paradigms such as offline Reinforcement Learning (RL) or Imitation Learning (IL), we require a more careful consideration of how uncertainty estimation interplays with first-order methods that attempt to minimize them. We study smoothed distance to data as an uncertainty metric, and claim that it has two beneficial properties: (i) it allows gradient-based methods that attempt to minimize uncertainty to drive iterates to data as smoothing is annealed, and (ii) it facilitates analysis of model bias with Lipschitz constants. As distance to data can be expensive to compute online, we consider settings where we need amortize this computation. Instead of learning the distance however, we propose to learn its gradients directly as an oracle for first-order optimizers. We show these gradients can be efficiently learned with score-matching techniques by leveraging the equivalence between distance to data and data likelihood. Using this insight, we propose Score-Guided Planning (SGP), a planning algorithm for offline RL that utilizes score-matching to enable first-order planning in high-dimensional problems, where zeroth-order methods were unable to scale, and ensembles were unable to overcome local minima. Website: https://sites.google.com/view/score-guided-planning/home

**摘要:** 基于梯度的方法能够在高维中实现高效的搜索能力。然而，为了将它们有效地应用于离线优化范例，如离线强化学习(RL)或模仿学习(IL)，我们需要更仔细地考虑不确定性估计如何与试图将其最小化的一阶方法相互作用。我们将数据的平滑距离作为一种不确定性度量来研究，并声称它有两个有益的性质：(I)它允许基于梯度的方法试图最小化不确定性，以在平滑处理后驱动对数据的迭代；(Ii)它便于使用Lipschitz常量分析模型偏差。由于在线计算到数据的距离可能成本高昂，因此我们考虑需要摊销此计算的设置。然而，我们不是学习距离，而是建议直接学习它的梯度，作为一阶优化器的预言。我们证明，通过利用数据距离和数据可能性之间的等价性，这些梯度可以通过分数匹配技术有效地学习。基于这一认识，我们提出了一种用于离线RL的规划算法SGP，该算法利用分数匹配来实现高维问题的一阶规划，在高维问题中，零阶方法无法缩放，并且集成无法克服局部极小。网址：https://sites.google.com/view/score-guided-planning/home

**[Paper URL](https://proceedings.mlr.press/v229/suh23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/suh23a/suh23a.pdf)** 

# Generative Skill Chaining: Long-Horizon Skill Planning with Diffusion Models
**题目:** 生成技能链：采用扩散模型的长期技能规划

**作者:** Utkarsh Aashu Mishra, Shangjie Xue, Yongxin Chen, Danfei Xu

**Abstract:** Long-horizon tasks, usually characterized by complex subtask dependencies, present a significant challenge in manipulation planning. Skill chaining is a practical approach to solving unseen tasks by combining learned skill priors. However, such methods are myopic if sequenced greedily and face scalability issues with search-based planning strategy. To address these challenges, we introduce Generative Skill Chaining (GSC), a probabilistic framework that learns skill-centric diffusion models and composes their learned distributions to generate long-horizon plans during inference. GSC samples from all skill models in parallel to efficiently solve unseen tasks while enforcing geometric constraints. We evaluate the method on various long-horizon tasks and demonstrate its capability in reasoning about action dependencies, constraint handling, and generalization, along with its ability to replan in the face of perturbations. We show results in simulation and on real robot to validate the efficiency and scalability of GSC, highlighting its potential for advancing long-horizon task planning. More details are available at: https://generative-skill-chaining.github.io/

**摘要:** 长期任务通常以复杂的子任务依赖关系为特征，这给操纵规划带来了巨大的挑战。技能链是一种通过结合已学技能优先顺序来解决未知任务的实用方法。然而，如果贪婪地排序，这种方法是短视的，并且面临基于搜索的规划策略的可扩展性问题。为了应对这些挑战，我们引入了生成技能链(GSC)，这是一个概率框架，它学习以技能为中心的扩散模型，并组成它们的学习分布，以在推理过程中生成长期计划。GSC并行地从所有技能模型中采样，以在实施几何约束的同时高效地解决看不见的任务。我们在不同的长时间任务上对该方法进行了评估，并展示了它在动作依赖推理、约束处理和泛化方面的能力，以及它在面对扰动时重新规划的能力。我们在仿真和真实机器人上的结果验证了GSC的有效性和可扩展性，突出了它在推进长期任务规划方面的潜力。欲了解更多详情，请访问：https://generative-skill-chaining.github.io/。

**[Paper URL](https://proceedings.mlr.press/v229/mishra23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mishra23a/mishra23a.pdf)** 

# Online Learning for Obstacle Avoidance
**题目:** 避障在线学习

**作者:** David Snyder, Meghan Booker, Nathaniel Simon, Wenhan Xia, Daniel Suo, Elad Hazan, Anirudha Majumdar

**Abstract:** We approach the fundamental problem of obstacle avoidance for robotic systems via the lens of online learning. In contrast to prior work that either assumes worst-case realizations of uncertainty in the environment or a stationary stochastic model of uncertainty, we propose a method that is efficient to implement and provably grants instance-optimality with respect to perturbations of trajectories generated from an open-loop planner (in the sense of minimizing worst-case regret). The resulting policy adapts online to realizations of uncertainty and provably compares well with the best obstacle avoidance policy in hindsight from a rich class of policies. The method is validated in simulation on a dynamical system environment and compared to baseline open-loop planning and robust Hamilton-Jacobi reachability techniques. Further, it is implemented on a hardware example where a quadruped robot traverses a dense obstacle field and encounters input disturbances due to time delays, model uncertainty, and dynamics nonlinearities.

**摘要:** 我们通过在线学习的视角来探讨机器人系统避障的基本问题。与以前的工作不同，我们要么假设环境中不确定性的最坏情况实现，要么假设不确定的平稳随机模型，相反，我们提出了一种方法，该方法对于开环规划器产生的轨迹的扰动(在最小化最坏情况的遗憾的意义上)是有效的并且可证明地授予实例最优性。由此产生的政策在网上适应了不确定性的实现，并被证明与事后看来最好的避障政策相比，这是一种丰富的政策类别。在动态系统环境下对该方法进行了仿真验证，并与基线开环规划和稳健的哈密顿-雅可比可达性方法进行了比较。进一步，在一个硬件例子上实现了它，其中一个四足机器人穿越密集的障碍场，并遇到由于时滞、模型不确定性和动力学非线性引起的输入扰动。

**[Paper URL](https://proceedings.mlr.press/v229/snyder23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/snyder23a/snyder23a.pdf)** 

# Polybot: Training One Policy Across Robots While Embracing Variability
**题目:** Polybot：在接受可变性的同时训练跨机器人的一项政策

**作者:** Jonathan Heewon Yang, Dorsa Sadigh, Chelsea Finn

**Abstract:** Reusing large datasets is crucial to scale vision-based robotic manipulators to everyday scenarios due to the high cost of collecting robotic datasets. However, robotic platforms possess varying control schemes, camera viewpoints, kinematic configurations, and end-effector morphologies, posing significant challenges when transferring manipulation skills from one platform to another. To tackle this problem, we propose a set of key design decisions to train a single policy for deployment on multiple robotic platforms. Our framework first aligns the observation and action spaces of our policy across embodiments via utilizing wrist cameras and a unified, but modular codebase. To bridge the remaining domain shift, we align our policy’s internal representations across embodiments via contrastive learning. We evaluate our method on a dataset collected over 60 hours spanning 6 tasks and 3 robots with varying joint configurations and sizes: the WidowX 250S, Franka Emika Panda, and Sawyer. Our results demonstrate significant improvements in success rate and sample efficiency for our policy when using new task data collected on a different robot, validating our proposed design decisions. More details and videos can be found on our project website: https://sites.google.com/view/cradle-multirobot

**摘要:** 由于收集机器人数据集的成本很高，因此重用大数据集对于将基于视觉的机器人操作器扩展到日常场景至关重要。然而，机器人平台具有不同的控制方案、摄像机视点、运动学配置和末端执行器形态，这给将操作技能从一个平台转移到另一个平台带来了巨大的挑战。为了解决这个问题，我们提出了一组关键的设计决策，以训练单个策略，以便在多个机器人平台上部署。我们的框架首先通过使用腕式摄像头和统一但模块化的代码库，跨实施例对齐我们政策的观察和行动空间。为了弥合剩余的域转移，我们通过对比学习跨实施例对齐我们的策略的内部表示。我们在一个超过60小时的数据集上对我们的方法进行了评估，收集了6个任务和3个关节配置和大小不同的机器人：WidowX 250S、Franka Emika Panda和Sawyer。当使用在不同机器人上收集的新任务数据时，我们的结果表明我们的策略在成功率和样本效率方面有了显著的改善，验证了我们提出的设计决策。更多细节和视频可在我们的项目网站上找到：https://sites.google.com/view/cradle-multirobot

**[Paper URL](https://proceedings.mlr.press/v229/yang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23c/yang23c.pdf)** 

# RoboPianist: Dexterous Piano Playing with Deep Reinforcement Learning
**题目:** 机器人钢琴家：采用深度强化学习的灵巧钢琴演奏

**作者:** Kevin Zakka, Philipp Wu, Laura Smith, Nimrod Gileadi, Taylor Howell, Xue Bin Peng, Sumeet Singh, Yuval Tassa, Pete Florence, Andy Zeng, Pieter Abbeel

**Abstract:** Replicating human-like dexterity in robot hands represents one of the largest open problems in robotics. Reinforcement learning is a promising approach that has achieved impressive progress in the last few years; however, the class of problems it has typically addressed corresponds to a rather narrow definition of dexterity as compared to human capabilities. To address this gap, we investigate piano-playing, a skill that challenges even the human limits of dexterity, as a means to test high-dimensional control, and which requires high spatial and temporal precision, and complex finger coordination and planning. We introduce RoboPianist, a system that enables simulated anthropomorphic hands to learn an extensive repertoire of 150 piano pieces where traditional model-based optimization struggles. We additionally introduce an open-sourced environment, benchmark of tasks, interpretable evaluation metrics, and open challenges for future study. Our website featuring videos, code, and datasets is available at https://kzakka.com/robopianist/

**摘要:** 在机器人手中复制类似人类的灵巧性是机器人学中最大的悬而未决的问题之一。强化学习是一种很有前途的方法，在过去几年中取得了令人印象深刻的进展；然而，它通常解决的问题类别对应于与人类能力相比对灵活性的一个相当狭窄的定义。为了解决这一差距，我们调查了钢琴演奏，这项技能甚至挑战了人类灵巧的极限，作为测试高维控制的一种手段，它需要高空间和时间精度，以及复杂的手指协调和规划。我们介绍了RoboPianist，这是一个使模拟的拟人手能够学习150首钢琴曲目的系统，传统的基于模型的优化在这些曲目中举步维艰。此外，我们还介绍了开源环境、任务基准、可解释的评估指标以及未来研究的开放挑战。我们的网站提供了视频、代码和数据集，网址为https://kzakka.com/robopianist/

**[Paper URL](https://proceedings.mlr.press/v229/zakka23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zakka23a/zakka23a.pdf)** 

# Revisiting Depth-guided Methods for Monocular 3D Object Detection by Hierarchical Balanced Depth
**题目:** 通过分层平衡深度重新审视深度引导单目3D对象检测方法

**作者:** Yi-Rong Chen, Ching-Yu Tseng, Yi-Syuan Liou, Tsung-Han Wu, Winston H. Hsu

**Abstract:** Monocular 3D object detection has seen significant advancements with the incorporation of depth information. However, there remains a considerable performance gap compared to LiDAR-based methods, largely due to inaccurate depth estimation. We argue that this issue stems from the commonly used pixel-wise depth map loss, which inherently creates the imbalance of loss weighting between near and distant objects. To address these challenges, we propose MonoHBD (Monocular Hierarchical Balanced Depth), a comprehensive solution with the hierarchical mechanism. We introduce the Hierarchical Depth Map (HDM) structure that incorporates depth bins and depth offsets to enhance the localization accuracy for objects. Leveraging RoIAlign, our Balanced Depth Extractor (BDE) module captures both scene-level depth relationships and object-specific depth characteristics while considering the geometry properties through the inclusion of camera calibration parameters. Furthermore, we propose a novel depth map loss that regularizes object-level depth features to mitigate imbalanced loss propagation. Our model reaches state-of-the-art results on the KITTI 3D object detection benchmark while supporting real-time detection. Excessive ablation studies are also conducted to prove the efficacy of our proposed modules.

**摘要:** 随着深度信息的加入，单目3D目标检测已经取得了显著的进步。然而，与基于LiDAR的方法相比，仍有相当大的性能差距，这主要是由于深度估计不准确所致。我们认为，这个问题源于通常使用的像素深度图损失，这内在地造成了近物体和远物体之间损失权重的不平衡。为了应对这些挑战，我们提出了MonoHBD(单目分层平衡深度)，这是一种具有分层机制的综合解决方案。我们引入了层次深度图(HDM)结构，该结构结合了深度框和深度偏移量，以提高目标的定位精度。利用RoIAlign，我们的平衡深度提取(BDE)模块捕获场景级深度关系和特定于对象的深度特征，同时通过包含相机校准参数来考虑几何属性。此外，我们还提出了一种新的深度图丢失方法，该方法将对象层的深度特征正则化，以减少不平衡的损失传播。我们的模型在Kitti 3D目标检测基准上达到了最先进的结果，同时支持实时检测。此外，还进行了过度烧蚀研究，以证明我们建议的模块的有效性。

**[Paper URL](https://proceedings.mlr.press/v229/chen23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23d/chen23d.pdf)** 

# Heteroscedastic Gaussian Processes and Random Features: Scalable Motion Primitives with Guarantees
**题目:** 异方差高斯过程和随机特征：带保证的可扩展运动基元

**作者:** Edoardo Caldarelli, Antoine Chatalic, Adrià Colomé, Lorenzo Rosasco, Carme Torras

**Abstract:** Heteroscedastic Gaussian processes (HGPs) are kernel-based, non-parametric models that can be used to infer nonlinear functions with time-varying noise. In robotics, they can be employed for learning from demonstration as motion primitives, i.e. as a model of the trajectories to be executed by the robot. HGPs provide variance estimates around the reference signal modeling the trajectory, capturing both the predictive uncertainty and the motion variability. However, similarly to standard Gaussian processes they suffer from a cubic complexity in the number of training points, due to the inversion of the kernel matrix. The uncertainty can be leveraged for more complex learning tasks, such as inferring the variable impedance profile required from a robotic manipulator. However, suitable approximations are needed to make HGPs scalable, at the price of potentially worsening the posterior mean and variance profiles. Motivated by these observations, we study the combination of HGPs and random features, which are a popular, data-independent approximation strategy of kernel functions. In a theoretical analysis, we provide novel guarantees on the approximation error of the HGP posterior due to random features. Moreover, we validate this scalable motion primitive on real robot data, related to the problem of variable impedance learning. In this way, we show that random features offer a viable and theoretically sound alternative for speeding up the trajectory processing, without sacrificing accuracy.

**摘要:** 异方差高斯过程(HGP)是一种基于核的非参数模型，可用于推断具有时变噪声的非线性函数。在机器人学中，它们可以作为运动基元从演示中学习，即作为机器人执行的轨迹的模型。HGPS提供关于模拟轨迹的参考信号的方差估计，捕捉到预测不确定性和运动可变性。然而，类似于标准的高斯过程，由于核矩阵的求逆，它们在训练点的数量上受到立方复杂性的影响。这种不确定性可用于更复杂的学习任务，例如推断机器人机械手所需的可变阻抗曲线。然而，需要适当的近似来使HGP具有可伸缩性，代价是潜在地恶化后验均值和方差分布。在这些观察的启发下，我们研究了HGP和随机特征的组合，这是一种流行的、与数据无关的核函数逼近策略。在理论分析中，我们对随机特征引起的HGP后验逼近误差提供了新的保证。此外，我们在与变阻抗学习问题相关的真实机器人数据上对该可伸缩运动原语进行了验证。通过这种方式，我们证明了随机特征为在不牺牲精度的情况下加速轨迹处理提供了一种可行且理论上合理的选择。

**[Paper URL](https://proceedings.mlr.press/v229/caldarelli23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/caldarelli23a/caldarelli23a.pdf)** 

# Human-in-the-Loop Task and Motion Planning for Imitation Learning
**题目:** 模拟学习的人在环任务和运动规划

**作者:** Ajay Mandlekar, Caelan Reed Garrett, Danfei Xu, Dieter Fox

**Abstract:** Imitation learning from human demonstrations can teach robots complex manipulation skills, but is time-consuming and labor intensive. In contrast, Task and Motion Planning (TAMP) systems are automated and excel at solving long-horizon tasks, but they are difficult to apply to contact-rich tasks. In this paper, we present Human-in-the-Loop Task and Motion Planning (HITL-TAMP), a novel system that leverages the benefits of both approaches. The system employs a TAMP-gated control mechanism, which selectively gives and takes control to and from a human teleoperator. This enables the human teleoperator to manage a fleet of robots, maximizing data collection efficiency. The collected human data is then combined with an imitation learning framework to train a TAMP-gated policy, leading to superior performance compared to training on full task demonstrations. We compared HITL-TAMP to a conventional teleoperation system — users gathered more than 3x the number of demos given the same time budget. Furthermore, proficient agents ($75%$+ success) could be trained from just 10 minutes of non-expert teleoperation data. Finally, we collected 2.1K demos with HITL-TAMP across 12 contact-rich, long-horizon tasks and show that the system often produces near-perfect agents. Videos and additional results at https://hitltamp.github.io .

**摘要:** 从人类演示中学习模仿可以教会机器人复杂的操作技能，但耗时耗力。相比之下，任务和行动计划(TAMP)系统是自动化的，擅长解决长期任务，但它们很难应用于联系人丰富的任务。在本文中，我们提出了人在回路任务和运动规划(HITL-TAMP)，一个新的系统，利用这两种方法的优点。该系统采用了一种篡改门控控制机制，它选择性地向人类遥操作人员提供控制，并从人类遥操作人员那里获得控制。这使人类远程操作员能够管理一队机器人，从而最大限度地提高数据收集效率。然后，将收集的人类数据与模仿学习框架相结合，以训练篡改策略，导致与完整任务演示的培训相比，获得更好的性能。我们将HITL-TAMP与传统的遥操作系统进行了比较--在相同的时间预算下，用户收集的演示数量是前者的3倍以上。此外，熟练的代理($75%$+Success)可以从10分钟的非专家遥操作数据中进行培训。最后，我们收集了12个接触丰富、长期任务的2.1K演示，表明该系统经常产生近乎完美的代理。Https://hitltamp.github.io上的视频和其他结果。

**[Paper URL](https://proceedings.mlr.press/v229/mandlekar23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/mandlekar23b/mandlekar23b.pdf)** 

# Gesture-Informed Robot Assistance via Foundation Models
**题目:** 通过基金会模型提供手势智能机器人协助

**作者:** Li-Heng Lin, Yuchen Cui, Yilun Hao, Fei Xia, Dorsa Sadigh

**Abstract:** Gestures serve as a fundamental and significant mode of non-verbal communication among humans. Deictic gestures (such as pointing towards an object), in particular, offer valuable means of efficiently expressing intent in situations where language is inaccessible, restricted, or highly specialized. As a result, it is essential for robots to comprehend gestures in order to infer human intentions and establish more effective coordination with them. Prior work often rely on a rigid hand-coded library of gestures along with their meanings. However, interpretation of gestures is often context-dependent, requiring more flexibility and common-sense reasoning. In this work, we propose a framework, GIRAF, for more flexibly interpreting gesture and language instructions by leveraging the power of large language models. Our framework is able to accurately infer human intent and contextualize the meaning of their gestures for more effective human-robot collaboration. We instantiate the framework for three table-top manipulation tasks and demonstrate that it is both effective and preferred by users. We further demonstrate GIRAF’s ability on reasoning about diverse types of gestures by curating a GestureInstruct dataset consisting of 36 different task scenarios. GIRAF achieved $81%$ success rate on finding the correct plan for tasks in GestureInstruct. Videos and datasets can be found on our project website: https://tinyurl.com/giraf23

**摘要:** 手势是人类之间进行非语言交流的一种基本而重要的方式。尤其是指示性手势(如指向一个物体)，在语言难以理解、受限或高度专门化的情况下，提供了有效表达意图的宝贵手段。因此，为了推断人类的意图并与其建立更有效的协调，机器人理解手势是至关重要的。以前的工作通常依赖于手势及其含义的严格手工编码库。然而，手势的解释通常依赖于上下文，需要更多的灵活性和常识性推理。在这项工作中，我们提出了一个框架Giraf，通过利用大型语言模型的能力来更灵活地解释手势和语言指令。我们的框架能够准确地推断人类的意图并将其手势的含义与上下文关联起来，以便更有效地进行人-机器人协作。我们对三个桌面操作任务的框架进行了实例化，并证明了该框架的有效性和用户的偏好。我们通过管理一个包含36个不同任务场景的GestureInstruct数据集，进一步展示了Giraf对不同类型手势的推理能力。Giraf在GestureInstruct中为任务找到正确计划的成功率为81%$。视频和数据集可以在我们的项目网站上找到：https://tinyurl.com/giraf23

**[Paper URL](https://proceedings.mlr.press/v229/lin23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lin23a/lin23a.pdf)** 

# TactileVAD: Geometric Aliasing-Aware Dynamics for High-Resolution Tactile Control
**题目:** TActileVAR：用于高分辨率触觉控制的几何混叠感知动力学

**作者:** Miquel Oller, Dmitry Berenson, Nima Fazeli

**Abstract:** Touch-based control is a promising approach to dexterous manipulation. However, existing tactile control methods often overlook tactile geometric aliasing which can compromise control performance and reliability. This type of aliasing occurs when different contact locations yield similar tactile signatures. To address this, we propose TactileVAD, a generative decoder-only linear latent dynamics formulation compatible with standard control methods that is capable of resolving geometric aliasing. We evaluate TactileVAD on two mechanically-distinct tactile sensors, SoftBubbles (pointcloud data) and Gelslim 3.0 (RGB data), showcasing its effectiveness in handling different sensing modalities. Additionally, we introduce the tactile cartpole, a novel benchmarking setup to evaluate the ability of a control method to respond to disturbances based on tactile input. Evaluations comparing TactileVAD to baselines suggest that our method is better able to achieve goal tactile configurations and hand poses.

**摘要:** 基于触摸的控制是一种很有前途的灵巧操作方法。然而，现有的触觉控制方法往往忽略了触觉几何混叠，这会影响控制性能和可靠性。当不同的接触位置产生相似的触觉签名时，就会出现这种类型的混叠。为了解决这一问题，我们提出了TactileVAD，这是一种仅针对生成式解码器的线性潜在动力学公式，与能够解决几何混叠的标准控制方法兼容。我们在两个机械上不同的触觉传感器，SoftBubble(点云数据)和GelSlim 3.0(RGB数据)上评估了TactileVAD，展示了它在处理不同传感模式方面的有效性。此外，我们还引入了触觉触觉CartPole，这是一种新的基准测试设置，用于评估基于触觉输入的控制方法对干扰的响应能力。比较TactileVAD和基线的评估表明，我们的方法能够更好地实现目标触觉配置和手部姿势。

**[Paper URL](https://proceedings.mlr.press/v229/oller23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/oller23a/oller23a.pdf)** 

# FastRLAP: A System for Learning High-Speed Driving via Deep RL and Autonomous Practicing
**题目:** FastRSYS：通过深度RL和自主练习学习高速驾驶的系统

**作者:** Kyle Stachowicz, Dhruv Shah, Arjun Bhorkar, Ilya Kostrikov, Sergey Levine

**Abstract:** We present a system that enables an autonomous small-scale RC car to drive aggressively from visual observations using reinforcement learning (RL). Our system, FastRLAP, trains autonomously in the real world, without human interventions, and without requiring any simulation or expert demonstrations. Our system integrates a number of important components to make this possible: we initialize the representations for the RL policy and value function from a large prior dataset of other robots navigating in other environments (at low speed), which provides a navigation-relevant representation. From here, a sample-efficient online RL method uses a single low-speed user-provided demonstration to determine the desired driving course, extracts a set of navigational checkpoints, and autonomously practices driving through these checkpoints, resetting automatically on collision or failure. Perhaps surprisingly, we find that with appropriate initialization and choice of algorithm, our system can learn to drive over a variety of racing courses with less than 20 minutes of online training. The resulting policies exhibit emergent aggressive driving skills, such as timing braking and acceleration around turns and avoiding areas which impede the robot’s motion, approaching the performance of a human driver using a similar first-person interface over the course of training.

**摘要:** 我们提出了一个使用强化学习(RL)的系统，该系统使自主小规模RC汽车能够从视觉观察中积极驾驶。我们的系统FastRLAP在真实世界中自主训练，不需要人工干预，也不需要任何模拟或专家演示。我们的系统集成了许多重要的组件来实现这一点：我们从其他环境中导航(低速)的其他机器人的大型先前数据集初始化RL策略和值函数的表示，这提供了与导航相关的表示。从这里开始，一种样本高效的在线RL方法使用用户提供的单个低速演示来确定所需的驾驶路线，提取一组导航检查点，并自动练习驾驶通过这些检查点，在碰撞或故障时自动重置。或许令人惊讶的是，我们发现，在适当的初始化和算法选择下，我们的系统可以在不到20分钟的在线训练时间内学习驾驶各种赛道。由此产生的政策展示了突如其来的攻击性驾驶技能，例如在转弯时计时刹车和加速，以及避开阻碍机器人运动的区域，在培训过程中使用类似的第一人称界面接近人类司机的表现。

**[Paper URL](https://proceedings.mlr.press/v229/stachowicz23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/stachowicz23a/stachowicz23a.pdf)** 

# Energy-based Potential Games for Joint Motion Forecasting and Control
**题目:** 关节运动预测和控制的基于能量的潜在游戏

**作者:** Christopher Diehl, Tobias Klosek, Martin Krueger, Nils Murzyn, Timo Osterburg, Torsten Bertram

**Abstract:** This work uses game theory as a mathematical framework to address interaction modeling in multi-agent motion forecasting and control. Despite its interpretability, applying game theory to real-world robotics, like automated driving, faces challenges such as unknown game parameters. To tackle these, we establish a connection between differential games, optimal control, and energy-based models, demonstrating how existing approaches can be unified under our proposed Energy-based Potential Game formulation. Building upon this, we introduce a new end-to-end learning application that combines neural networks for game-parameter inference with a differentiable game-theoretic optimization layer, acting as an inductive bias. The analysis provides empirical evidence that the game-theoretic layer adds interpretability and improves the predictive performance of various neural network backbones using two simulations and two real-world driving datasets.

**摘要:** 这项工作使用博弈论作为数学框架来解决多智能体运动预测和控制中的交互建模。尽管具有可解释性，但将博弈论应用于现实世界的机器人（例如自动驾驶），仍面临着未知游戏参数等挑战。为了解决这些问题，我们在差异博弈、最优控制和基于能量的模型之间建立了联系，展示了如何在我们提出的基于能量的潜在博弈公式下统一现有方法。在此基础上，我们引入了一种新的端到端学习应用程序，该应用程序将用于游戏参数推理的神经网络与可微博弈论优化层相结合，充当归纳偏差。该分析提供了经验证据，表明博弈论层使用两个模拟和两个现实世界驾驶数据集增加了可解释性并提高了各种神经网络主干的预测性能。

**[Paper URL](https://proceedings.mlr.press/v229/diehl23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/diehl23a/diehl23a.pdf)** 

# Dexterity from Touch: Self-Supervised Pre-Training of Tactile Representations with Robotic Play
**题目:** 触摸带来的灵活性：通过机器人游戏对触觉表示进行自我监督的预训练

**作者:** Irmak Guzey, Ben Evans, Soumith Chintala, Lerrel Pinto

**Abstract:** Teaching dexterity to multi-fingered robots has been a longstanding challenge in robotics. Most prominent work in this area focuses on learning controllers or policies that either operate on visual observations or state estimates derived from vision. However, such methods perform poorly on fine-grained manipulation tasks that require reasoning about contact forces or about objects occluded by the hand itself. In this work, we present T-Dex, a new approach for tactile-based dexterity, that operates in two phases. In the first phase, we collect 2.5 hours of play data, which is used to train self-supervised tactile encoders. This is necessary to bring high-dimensional tactile readings to a lower-dimensional embedding. In the second phase, given a handful of demonstrations for a dexterous task, we learn non-parametric policies that combine the tactile observations with visual ones. Across five challenging dexterous tasks, we show that our tactile-based dexterity models outperform purely vision and torque-based models by an average of 1.7X. Finally, we provide a detailed analysis on factors critical to T-Dex including the importance of play data, architectures, and representation learning.

**摘要:** 教授多指机器人的灵巧性一直是机器人学中的一个长期挑战。这一领域最突出的工作集中在学习控制器或策略，这些控制器或策略要么基于视觉观察，要么基于视觉得出的状态估计。然而，这种方法在细粒度的操作任务中表现不佳，这些任务需要关于接触力或关于被手本身遮挡的对象的推理。在这项工作中，我们提出了一种新的基于触觉的灵活性方法T-Dex，它分两个阶段进行操作。在第一阶段，我们收集了2.5小时的游戏数据，用于训练自我监督的触觉编码器。这对于将高维触觉读数带入低维嵌入是必要的。在第二阶段，给出几个灵活任务的演示，我们学习将触觉观察和视觉观察相结合的非参数策略。在五项具有挑战性的灵活任务中，我们展示了我们的基于触觉的灵巧模型比基于纯视觉和基于扭矩的模型平均性能高出1.7倍。最后，我们对T-Dex的关键因素进行了详细的分析，包括游戏数据、体系结构和表征学习的重要性。

**[Paper URL](https://proceedings.mlr.press/v229/guzey23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/guzey23a/guzey23a.pdf)** 

# ADU-Depth: Attention-based Distillation with Uncertainty Modeling for Depth Estimation
**题目:** ADU-Depth：基于注意力的蒸馏，具有深度估计的不确定性建模

**作者:** ZiZhang Wu, Zhuozheng Li, Zhi-Gang Fan, Yunzhe Wu, Xiaoquan Wang, Rui Tang, Jian Pu

**Abstract:** Monocular depth estimation is challenging due to its inherent ambiguity and ill-posed nature, yet it is quite important to many applications. While recent works achieve limited accuracy by designing increasingly complicated networks to extract features with limited spatial geometric cues from a single RGB image, we intend to introduce spatial cues by training a teacher network that leverages left-right image pairs as inputs and transferring the learned 3D geometry-aware knowledge to the monocular student network. Specifically, we present a novel knowledge distillation framework, named ADU-Depth, with the goal of leveraging the well-trained teacher network to guide the learning of the student network, thus boosting the precise depth estimation with the help of extra spatial scene information. To enable domain adaptation and ensure effective and smooth knowledge transfer from teacher to student, we apply both attention-adapted feature distillation and focal-depth-adapted response distillation in the training stage. In addition, we explicitly model the uncertainty of depth estimation to guide distillation in both feature space and result space to better produce 3D-aware knowledge from monocular observations and thus enhance the learning for hard-to-predict image regions. Our extensive experiments on the real depth estimation datasets KITTI and DrivingStereo demonstrate the effectiveness of the proposed method, which ranked 1st on the challenging KITTI online benchmark.

**摘要:** 单目深度估计由于其固有的模糊性和病态性质而具有挑战性，但它在许多应用中都是相当重要的。虽然最近的工作通过设计越来越复杂的网络来从单个RGB图像中提取具有有限空间几何线索的特征来实现有限的精度，但我们打算通过训练教师网络来引入空间线索，该网络利用左右图像对作为输入，并将学习的3D几何感知知识传输到单目学生网络。具体地说，我们提出了一种新的知识提取框架ADU-Depth，其目的是利用训练有素的教师网络来指导学生网络的学习，从而借助额外的空间场景信息来提高精确的深度估计。为了实现领域自适应，并确保从教师到学生的有效和顺畅的知识传递，我们在训练阶段采用了注意力适应的特征提取和焦点深度适应的反应提取。此外，我们对深度估计的不确定性进行了显式建模，以指导特征空间和结果空间的蒸馏，从而更好地从单目观测中产生3D感知知识，从而增强对难以预测的图像区域的学习。我们在真实深度估计数据集Kitti和DrivingStereo上的大量实验证明了该方法的有效性，该方法在具有挑战性的Kitti在线基准测试中排名第一。

**[Paper URL](https://proceedings.mlr.press/v229/wu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wu23c/wu23c.pdf)** 

# Structural Concept Learning via Graph Attention for Multi-Level Rearrangement Planning
**题目:** 通过图形注意力进行结构概念学习用于多层重新排列规划

**作者:** Manav Kulshrestha, Ahmed H. Qureshi

**Abstract:** Robotic manipulation tasks, such as object rearrangement, play a crucial role in enabling robots to interact with complex and arbitrary environments. Existing work focuses primarily on single-level rearrangement planning and, even if multiple levels exist, dependency relations among substructures are geometrically simpler, like tower stacking. We propose Structural Concept Learning (SCL), a deep learning approach that leverages graph attention networks to perform multi-level object rearrangement planning for scenes with structural dependency hierarchies. It is trained on a self-generated simulation data set with intuitive structures, works for unseen scenes with an arbitrary number of objects and higher complexity of structures, infers independent substructures to allow for task parallelization over multiple manipulators, and generalizes to the real world. We compare our method with a range of classical and model-based baselines to show that our method leverages its scene understanding to achieve better performance, flexibility, and efficiency. The dataset, demonstration videos, supplementary details, and code implementation are available at: https://manavkulshrestha.github.io/scl

**摘要:** 机器人操作任务，如物体重排，在使机器人能够与复杂和任意的环境交互方面发挥着至关重要的作用。现有的工作主要集中在单层重排规划上，即使存在多层，子结构之间的依赖关系在几何上更简单，就像塔楼堆叠一样。我们提出了结构概念学习(SCL)，这是一种深度学习方法，它利用图注意网络对具有结构依赖层次的场景进行多层次的对象重排规划。它在具有直观结构的自生成仿真数据集上进行训练，适用于具有任意数量的对象和更高结构复杂性的不可见场景，推断独立的子结构以允许在多个机械手上进行任务并行，并推广到真实世界。我们将我们的方法与一系列经典的和基于模型的基线进行了比较，表明我们的方法利用了它的场景理解来获得更好的性能、灵活性和效率。数据集、演示视频、补充细节和代码实现可在以下网站获得：https://manavkulshrestha.github.io/scl

**[Paper URL](https://proceedings.mlr.press/v229/kulshrestha23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/kulshrestha23a/kulshrestha23a.pdf)** 

# Few-Shot In-Context Imitation Learning via Implicit Graph Alignment
**题目:** 通过隐式图对齐的少镜头上下文模仿学习

**作者:** Vitalis Vosylius, Edward Johns

**Abstract:** Consider the following problem: given a few demonstrations of a task across a few different objects, how can a robot learn to perform that same task on new, previously unseen objects? This is challenging because the large variety of objects within a class makes it difficult to infer the task-relevant relationship between the new objects and the objects in the demonstrations. We address this by formulating imitation learning as a conditional alignment problem between graph representations of objects. Consequently, we show that this conditioning allows for in-context learning, where a robot can perform a task on a set of new objects immediately after the demonstrations, without any prior knowledge about the object class or any further training. In our experiments, we explore and validate our design choices, and we show that our method is highly effective for few-shot learning of several real-world, everyday tasks, whilst outperforming baselines. Videos are available on our project webpage at https://www.robot-learning.uk/implicit-graph-alignment.

**摘要:** 考虑以下问题：给出几个跨越几个不同对象的任务演示，机器人如何学习在以前未见过的新对象上执行相同的任务？这是具有挑战性的，因为类中的对象种类繁多，因此很难推断新对象和演示中的对象之间的任务相关关系。我们通过将模仿学习描述为对象的图形表示之间的条件对齐问题来解决这一问题。因此，我们表明，这种条件允许情景学习，其中机器人可以在演示后立即对一组新对象执行任务，而不需要任何关于对象类的先验知识或任何进一步的培训。在我们的实验中，我们探索并验证了我们的设计选择，我们证明了我们的方法对于几个现实世界中的日常任务的极少机会学习是非常有效的，同时超过了基线。视频可以在我们的项目网页上找到，网址是https://www.robot-learning.uk/implicit-graph-alignment.

**[Paper URL](https://proceedings.mlr.press/v229/vosylius23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/vosylius23a/vosylius23a.pdf)** 

# Topology-Matching Normalizing Flows for Out-of-Distribution Detection in Robot Learning
**题目:** 机器人学习中用于分布外检测的布局匹配规范化流程

**作者:** Jianxiang Feng, Jongseok Lee, Simon Geisler, Stephan Günnemann, Rudolph Triebel

**Abstract:** To facilitate reliable deployments of autonomous robots in the real world, Out-of-Distribution (OOD) detection capabilities are often required. A powerful approach for OOD detection is based on density estimation with Normalizing Flows (NFs). However, we find that prior work with NFs attempts to match the complex target distribution topologically with naïve base distributions leading to adverse implications. In this work, we circumvent this topological mismatch using an expressive class-conditional base distribution trained with an information-theoretic objective to match the required topology. The proposed method enjoys the merits of wide compatibility with existing learned models without any performance degradation and minimum computation overhead while enhancing OOD detection capabilities. We demonstrate superior results in density estimation and 2D object detection benchmarks in comparison with extensive baselines. Moreover, we showcase the applicability of the method with a real-robot deployment.

**摘要:** 为了促进自主机器人在现实世界中的可靠部署，通常需要具有分布外(OOD)检测能力。一种有效的OOD检测方法是基于归一化流密度估计的方法。然而，我们发现，以前使用NFS的工作试图在拓扑上将复杂的目标分布与天真的基分布匹配，从而导致不利的影响。在这项工作中，我们使用一个具有表现力的类条件基分布来避免这种拓扑失配，该分布以信息论的目标进行训练，以匹配所需的拓扑。该方法具有与已有学习模型广泛兼容的优点，在不降低性能和最小计算开销的同时，提高了OOD检测能力。与广泛的基线相比，我们在密度估计和2D目标检测基准方面表现出了优越的结果。此外，我们通过一个真实的机器人部署展示了该方法的适用性。

**[Paper URL](https://proceedings.mlr.press/v229/feng23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/feng23b/feng23b.pdf)** 

# Compositional Diffusion-Based Continuous Constraint Solvers
**题目:** 基于成分扩散的连续约束求解器

**作者:** Zhutian Yang, Jiayuan Mao, Yilun Du, Jiajun Wu, Joshua B. Tenenbaum, Tomás Lozano-Pérez, Leslie Pack Kaelbling

**Abstract:** This paper introduces an approach for learning to solve continuous constraint satisfaction problems (CCSP) in robotic reasoning and planning. Previous methods primarily rely on hand-engineering or learning generators for specific constraint types and then rejecting the value assignments when other constraints are violated. By contrast, our model, the compositional diffusion continuous constraint solver (Diffusion-CCSP) derives global solutions to CCSPs by representing them as factor graphs and combining the energies of diffusion models trained to sample for individual constraint types. Diffusion-CCSP exhibits strong generalization to novel combinations of known constraints, and it can be integrated into a task and motion planner to devise long-horizon plans that include actions with both discrete and continuous parameters.

**摘要:** 本文介绍了一种在机器人推理和规划中学习解决连续约束满足问题（CCSP）的方法。以前的方法主要依赖于特定约束类型的手工工程或学习生成器，然后在违反其他约束时拒绝值分配。相比之下，我们的模型，即成分扩散连续约束求解器（扩散-CCSP），通过将它们表示为因子图并结合经过训练以针对单个约束类型进行抽样的扩散模型的能量来推导CCSP的全局解。扩散-CCSP对已知约束的新颖组合具有很强的概括性，并且它可以集成到任务和运动规划器中，以设计包括具有离散和连续参数的动作的长期计划。

**[Paper URL](https://proceedings.mlr.press/v229/yang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23d/yang23d.pdf)** 

# Precise Robotic Needle-Threading with Tactile Perception and Reinforcement Learning
**题目:** 具有触觉感知和强化学习的精确机器人穿针

**作者:** Zhenjun Yu, Wenqiang Xu, Siqiong Yao, Jieji Ren, Tutian Tang, Yutong Li, Guoying Gu, Cewu Lu

**Abstract:** This work presents a novel tactile perception-based method, named T-NT, for performing the needle-threading task, an application of deformable linear object (DLO) manipulation. This task is divided into two main stages: Tail-end Finding and Tail-end Insertion. In the first stage, the agent traces the contour of the thread twice using vision-based tactile sensors mounted on the gripper fingers. The two-run tracing is to locate the tail-end of the thread. In the second stage, it employs a tactile-guided reinforcement learning (RL) model to drive the robot to insert the thread into the target needle eyelet. The RL model is trained in a Unity-based simulated environment. The simulation environment supports tactile rendering which can produce realistic tactile images and thread modeling. During insertion, the position of the poke point and the center of the eyelet are obtained through a pre-trained segmentation model, Grounded-SAM, which predicts the masks for both the needle eye and thread imprints. These positions are then fed into the reinforcement learning model, aiding in a smoother transition to real-world applications. Extensive experiments on real robots are conducted to demonstrate the efficacy of our method. More experiments and videos can be found in the supplementary materials and on the website: https://sites.google.com/view/tac-needlethreading.

**摘要:** 提出了一种新的基于触觉感知的穿针方法T-NT，它是可变形线性物体(DLO)操作的一种应用。这项任务分为两个主要阶段：尾端发现和尾端插入。在第一阶段，代理使用安装在夹持器手指上的基于视觉的触觉传感器跟踪两次线的轮廓。两次跟踪是为了定位线程的尾部。在第二阶段，它采用触觉引导的强化学习(RL)模型来驱动机器人将线插入目标针孔。RL模型在基于Unity的模拟环境中进行训练。该仿真环境支持触觉渲染，可以生成逼真的触觉图像和线程建模。在插入过程中，通过预先训练的分割模型GROUND-SAM获得穿刺点的位置和小孔的中心，该模型预测针眼和线印的掩模。然后，这些位置被输入强化学习模型，帮助更平稳地过渡到真实世界的应用程序。在真实机器人上进行了大量的实验，以验证该方法的有效性。更多实验和视频可以在补充材料中找到，也可以在以下网站上找到：https://sites.google.com/view/tac-needlethreading.

**[Paper URL](https://proceedings.mlr.press/v229/yu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yu23c/yu23c.pdf)** 

# Cold Diffusion on the Replay Buffer: Learning to Plan from Known Good States
**题目:** 重播缓冲区的冷扩散：学习从已知的良好状态进行计划

**作者:** Zidan Wang, Takeru Oba, Takuma Yoneda, Rui Shen, Matthew Walter, Bradly C. Stadie

**Abstract:** Learning from demonstrations (LfD) has successfully trained robots to exhibit remarkable generalization capabilities. However, many powerful imitation techniques do not prioritize the feasibility of the robot behaviors they generate. In this work, we explore the feasibility of plans produced by LfD. As in prior work, we employ a temporal diffusion model with fixed start and goal states to facilitate imitation through in-painting. Unlike previous studies, we apply cold diffusion to ensure the optimization process is directed through the agent’s replay buffer of previously visited states. This routing approach increases the likelihood that the final trajectories will predominantly occupy the feasible region of the robot’s state space. We test this method in simulated robotic environments with obstacles and observe a significant improvement in the agent’s ability to avoid these obstacles during planning.

**摘要:** 从演示中学习（LfD）已成功训练机器人，使其表现出出色的概括能力。然而，许多强大的模仿技术并没有优先考虑其产生的机器人行为的可行性。在这项工作中，我们探索了LfD制定计划的可行性。与之前的工作一样，我们采用具有固定开始和目标状态的时间扩散模型，以促进通过内绘进行模仿。与之前的研究不同，我们应用冷扩散来确保优化过程通过代理之前访问状态的重播缓冲区来指导。这种路由方法增加了最终轨迹主要占据机器人状态空间可行区域的可能性。我们在带有障碍物的模拟机器人环境中测试了这种方法，并观察到代理在规划期间避开这些障碍物的能力有了显着提高。

**[Paper URL](https://proceedings.mlr.press/v229/wang23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23e/wang23e.pdf)** 

# Self-Improving Robots: End-to-End Autonomous Visuomotor Reinforcement Learning
**题目:** 自我改进机器人：端到端自主视觉强化学习

**作者:** Archit Sharma, Ahmed M. Ahmed, Rehaan Ahmad, Chelsea Finn

**Abstract:** In imitation and reinforcement learning (RL), the cost of human supervision limits the amount of data that the robots can be trained on. While RL offers a framework for building self-improving robots that can learn via trial-and-error autonomously, practical realizations end up requiring extensive human supervision for reward function design and repeated resetting of the environment between episodes of interactions. In this work, we propose MEDAL++, a novel design for self-improving robotic systems: given a small set of expert demonstrations at the start, the robot autonomously practices the task by learning to both do and undo the task, simultaneously inferring the reward function from the demonstrations. The policy and reward function are learned end-to-end from high-dimensional visual inputs, bypassing the need for explicit state estimation or task-specific pre-training for visual encoders used in prior work. We first evaluate our proposed system on a simulated non-episodic benchmark EARL, finding that MEDAL++ is both more data efficient and gets up to $30%$ better final performance compared to state-of-the-art vision-based methods. Our real-robot experiments show that MEDAL++ can be applied to manipulation problems in larger environments than those considered in prior work, and autonomous self-improvement can improve the success rate by $30%$ to $70%$ over behavioral cloning on just the expert data.

**摘要:** 在模拟和强化学习(RL)中，人工监督的成本限制了机器人可以训练的数据量。虽然RL提供了一个框架，用于构建能够通过反复试验自主学习的自我改进机器人，但实际实现最终需要对奖励功能设计进行广泛的人类监督，并在交互事件之间反复重置环境。在这项工作中，我们提出了一种用于自我改进的机器人系统的新颖设计Medal++：在开始时给出一小组专家演示，机器人通过学习执行和取消任务来自主练习任务，同时从演示中推断奖励函数。策略和奖励函数是从高维视觉输入端到端学习的，绕过了对先前工作中使用的视觉编码器的显式状态估计或特定于任务的预训练的需要。我们首先在一个模拟的无情节基准Earl上对我们提出的系统进行了评估，发现Medal++与最先进的基于视觉的方法相比，既有更高的数据效率，又获得了高达30%的最终性能。我们的真实机器人实验表明，与以前的工作相比，Medal++可以应用于更大环境中的操作问题，自主自我改进可以比仅基于专家数据的行为克隆的成功率提高30%-70%。

**[Paper URL](https://proceedings.mlr.press/v229/sharma23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/sharma23b/sharma23b.pdf)** 

# Equivariant Reinforcement Learning under Partial Observability
**题目:** 部分可观察性下的等变强化学习

**作者:** Hai Huu Nguyen, Andrea Baisero, David Klee, Dian Wang, Robert Platt, Christopher Amato

**Abstract:** Incorporating inductive biases is a promising approach for tackling challenging robot learning domains with sample-efficient solutions. This paper identifies partially observable domains where symmetries can be a useful inductive bias for efficient learning. Specifically, by encoding the equivariance regarding specific group symmetries into the neural networks, our actor-critic reinforcement learning agents can reuse solutions in the past for related scenarios. Consequently, our equivariant agents outperform non-equivariant approaches significantly in terms of sample efficiency and final performance, demonstrated through experiments on a range of robotic tasks in simulation and real hardware.

**摘要:** 消除归纳偏差是通过样本高效的解决方案解决具有挑战性的机器人学习领域的一种有前途的方法。本文确定了部分可观察的领域，其中对称性可以成为有效学习的有用的归纳偏差。具体来说，通过将特定群对称性的等方差编码到神经网络中，我们的行动者-批评者强化学习代理可以将过去的解决方案重复用于相关场景。因此，我们的等变代理在样本效率和最终性能方面显着优于非等变方法，这通过模拟和真实硬件中的一系列机器人任务的实验得到了证明。

**[Paper URL](https://proceedings.mlr.press/v229/nguyen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/nguyen23a/nguyen23a.pdf)** 

# UniFolding: Towards Sample-efficient, Scalable, and Generalizable Robotic Garment Folding
**题目:** UniFounding：迈向样本高效、可扩展和可推广的机器人服装折叠

**作者:** Han Xue, Yutong Li, Wenqiang Xu, Huanyu Li, Dongzhe Zheng, Cewu Lu

**Abstract:** This paper explores the development of UniFolding, a sample-efficient, scalable, and generalizable robotic system for unfolding and folding various garments. UniFolding employs the proposed UFONet neural network to integrate unfolding and folding decisions into a single policy model that is adaptable to different garment types and states. The design of UniFolding is based on a garment’s partial point cloud, which aids in generalization and reduces sensitivity to variations in texture and shape. The training pipeline prioritizes low-cost, sample-efficient data collection. Training data is collected via a human-centric process with offline and online stages. The offline stage involves human unfolding and folding actions via Virtual Reality, while the online stage utilizes human-in-the-loop learning to fine-tune the model in a real-world setting. The system is tested on two garment types: long-sleeve and short-sleeve shirts. Performance is evaluated on 20 shirts with significant variations in textures, shapes, and materials. More experiments and videos can be found in the supplementary materials and on the website: https://unifolding.robotflow.ai.

**摘要:** 本文探讨了UniFolding的开发，这是一个样本高效、可扩展和可推广的机器人系统，用于展开和折叠各种服装。UniFolding使用提出的UFONet神经网络将展开和折叠决策集成到一个单一的策略模型中，该模型可适应不同的服装类型和状态。UniFolding的设计基于服装的局部点云，这有助于泛化并降低对纹理和形状变化的敏感度。培训渠道优先考虑低成本、样本效率高的数据收集。培训数据通过以人为中心的流程收集，包括离线和在线阶段。离线阶段包括通过虚拟现实的人类展开和折叠动作，而在线阶段利用人在回路中的学习在真实世界环境中对模型进行微调。该系统在两种服装类型上进行了测试：长袖和短袖衬衫。对20件质地、形状和材质差异很大的衬衫进行了性能评估。更多实验和视频可以在补充材料中找到，也可以在以下网站上找到：https://unifolding.robotflow.ai.

**[Paper URL](https://proceedings.mlr.press/v229/xue23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xue23b/xue23b.pdf)** 

# A Universal Semantic-Geometric Representation for Robotic Manipulation
**题目:** 机器人操纵的通用语义-几何表示

**作者:** Tong Zhang, Yingdong Hu, Hanchen Cui, Hang Zhao, Yang Gao

**Abstract:** Robots rely heavily on sensors, especially RGB and depth cameras, to perceive and interact with the world. RGB cameras record 2D images with rich semantic information while missing precise spatial information. On the other side, depth cameras offer critical 3D geometry data but capture limited semantics. Therefore, integrating both modalities is crucial for learning representations for robotic perception and control. However, current research predominantly focuses on only one of these modalities, neglecting the benefits of incorporating both. To this end, we present Semantic-Geometric Representation (SGR), a universal perception module for robotics that leverages the rich semantic information of large-scale pre-trained 2D models and inherits the merits of 3D spatial reasoning. Our experiments demonstrate that SGR empowers the agent to successfully complete a diverse range of simulated and real-world robotic manipulation tasks, outperforming state-of-the-art methods significantly in both single-task and multi-task settings. Furthermore, SGR possesses the capability to generalize to novel semantic attributes, setting it apart from the other methods. Project website: https://semantic-geometric-representation.github.io.

**摘要:** 机器人严重依赖传感器，特别是RGB和深度相机，来感知世界并与之互动。RGB相机记录的2D图像具有丰富的语义信息，但缺少精确的空间信息。另一方面，深度摄像头提供了关键的3D几何数据，但捕捉到的语义有限。因此，整合这两种模式对于学习机器人感知和控制的表示是至关重要的。然而，目前的研究主要集中在这两种模式中的一种，而忽视了将两者结合起来的好处。为此，我们提出了语义几何表示(SGR)，这是一种面向机器人的通用感知模块，它利用了大规模预先训练的2D模型丰富的语义信息，继承了3D空间推理的优点。我们的实验表明，SGR使代理能够成功完成各种模拟和真实世界的机器人操作任务，在单任务和多任务设置下的性能都显著优于最先进的方法。此外，SGR具有对新的语义属性进行泛化的能力，使其有别于其他方法。项目网站：https://semantic-geometric-representation.github.io.

**[Paper URL](https://proceedings.mlr.press/v229/zhang23j.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23j/zhang23j.pdf)** 

# LabelFormer: Object Trajectory Refinement for Offboard Perception from LiDAR Point Clouds
**题目:** LabelFormer：激光雷达点云离机感知的物体轨迹细化

**作者:** Anqi Joyce Yang, Sergio Casas, Nikita Dvornik, Sean Segal, Yuwen Xiong, Jordan Sir Kwang Hu, Carter Fang, Raquel Urtasun

**Abstract:** A major bottleneck to scaling-up training of self-driving perception systems are the human annotations required for supervision. A promising alternative is to leverage “auto-labelling" offboard perception models that are trained to automatically generate annotations from raw LiDAR point clouds at a fraction of the cost. Auto-labels are most commonly generated via a two-stage approach – first objects are detected and tracked over time, and then each object trajectory is passed to a learned refinement model to improve accuracy. Since existing refinement models are overly complex and lack advanced temporal reasoning capabilities, in this work we propose LabelFormer, a simple, efficient, and effective trajectory-level refinement approach. Our approach first encodes each frame’s observations separately, then exploits self-attention to reason about the trajectory with full temporal context, and finally decodes the refined object size and per-frame poses. Evaluation on both urban and highway datasets demonstrates that LabelFormer outperforms existing works by a large margin. Finally, we show that training on a dataset augmented with auto-labels generated by our method leads to improved downstream detection performance compared to existing methods. Please visit the project website for details https://waabi.ai/labelformer/.

**摘要:** 自动驾驶感知系统扩大训练的一个主要瓶颈是监督所需的人类注释。一种很有前途的替代方案是利用“自动标记”场外感知模型，这些模型经过训练，能够以极低的成本从原始的LiDAR点云中自动生成注释。自动标签通常是通过两个阶段的方法生成的-首先检测和跟踪对象，然后将每个对象轨迹传递到学习的细化模型以提高精度。针对现有精化模型过于复杂且缺乏高级时序推理能力的问题，本文提出了一种简单、高效、有效的轨迹级精化方法LabelFormer。我们的方法首先对每一帧的观测数据进行单独编码，然后利用自我注意在完整的时间背景下对轨迹进行推理，最后解码出精细化的对象大小和每帧的姿势。在城市和高速公路数据集上的评估表明，LabelFormer的性能远远超过现有的工作。最后，我们表明，与现有方法相比，在添加了自动标签的数据集上进行训练可以提高下行检测性能。请访问项目网站了解详细信息https://waabi.ai/labelformer/.

**[Paper URL](https://proceedings.mlr.press/v229/yang23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yang23e/yang23e.pdf)** 

# Language-Conditioned Path Planning
**题目:** 受影响的路径规划

**作者:** Amber Xie, Youngwoon Lee, Pieter Abbeel, Stephen James

**Abstract:** Contact is at the core of robotic manipulation. At times, it is desired (e.g. manipulation and grasping), and at times, it is harmful (e.g. when avoiding obstacles). However, traditional path planning algorithms focus solely on collision-free paths, limiting their applicability in contact-rich tasks. To address this limitation, we propose the domain of Language-Conditioned Path Planning, where contact-awareness is incorporated into the path planning problem. As a first step in this domain, we propose Language-Conditioned Collision Functions (LACO), a novel approach that learns a collision function using only a single-view image, language prompt, and robot configuration. LACO predicts collisions between the robot and the environment, enabling flexible, conditional path planning without the need for manual object annotations, point cloud data, or ground-truth object meshes. In both simulation and the real world, we demonstrate that LACO can facilitate complex, nuanced path plans that allow for interaction with objects that are safe to collide, rather than prohibiting any collision.

**摘要:** 接触是机器人操作的核心。有时，它是人们所需要的(例如，操纵和抓取)，有时，它是有害的(例如，在躲避障碍物时)。然而，传统的路径规划算法只关注无冲突路径，限制了它们在接触丰富任务中的适用性。为了解决这一局限性，我们提出了语言条件路径规划领域，其中接触感知被结合到路径规划问题中。作为这一领域的第一步，我们提出了语言条件碰撞函数(LACO)，这是一种仅使用单视图、语言提示和机器人配置来学习碰撞函数的新方法。LACO预测机器人与环境之间的碰撞，从而实现灵活的有条件的路径规划，而不需要手动对象注释、点云数据或地面真实对象网格。在模拟和现实世界中，我们都展示了LACO可以促进复杂的、细微差别的路径规划，允许与安全碰撞的对象进行交互，而不是阻止任何碰撞。

**[Paper URL](https://proceedings.mlr.press/v229/xie23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xie23b/xie23b.pdf)** 

# Open-World Object Manipulation using Pre-Trained Vision-Language Models
**题目:** 使用预训练的视觉语言模型进行开放世界对象操作

**作者:** Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, Chelsea Finn, Karol Hausman

**Abstract:** For robots to follow instructions from people, they must be able to connect the rich semantic information in human vocabulary, e.g. “can you get me the pink stuffed whale?” to their sensory observations and actions. This brings up a notably difficult challenge for robots: while robot learning approaches allow robots to learn many different behaviors from first-hand experience, it is impractical for robots to have first-hand experiences that span all of this semantic information. We would like a robot’s policy to be able to perceive and pick up the pink stuffed whale, even if it has never seen any data interacting with a stuffed whale before. Fortunately, static data on the internet has vast semantic information, and this information is captured in pre-trained vision-language models. In this paper, we study whether we can interface robot policies with these pre-trained models, with the aim of allowing robots to complete instructions involving object categories that the robot has never seen first-hand. We develop a simple approach, which we call Manipulation of Open-World Objects (MOO), which leverages a pre-trained vision-language model to extract object-identifying information from the language command and image, and conditions the robot policy on the current image, the instruction, and the extracted object information. In a variety of experiments on a real mobile manipulator, we find that MOO generalizes zero-shot to a wide range of novel object categories and environments. In addition, we show how MOO generalizes to other, non-language-based input modalities to specify the object of interest such as finger pointing, and how it can be further extended to enable open-world navigation and manipulation. The project’s website and evaluation videos can be found at https://robot-moo.github.io/.

**摘要:** 要让机器人听从人类的指令，它们必须能够连接人类词汇中丰富的语义信息，例如：你能给我拿到粉色的填充鲸鱼吗？他们的感官观察和行动。这给机器人带来了一个非常困难的挑战：虽然机器人学习方法允许机器人从第一手经验中学习许多不同的行为，但对于机器人来说，拥有跨越所有这些语义信息的第一手经验是不切实际的。我们希望机器人的策略能够感知和捕捉粉色填充鲸鱼，即使它以前从未见过任何数据与填充鲸鱼互动。幸运的是，互联网上的静态数据有大量的语义信息，这些信息是在预先训练好的视觉语言模型中捕获的。在本文中，我们研究是否可以将机器人策略与这些预先训练的模型相接口，目的是允许机器人完成涉及机器人从未亲眼目睹的对象类别的指令。我们开发了一种简单的方法，称为开放世界对象操纵(MoO)，它利用预先训练的视觉语言模型从语言命令和图像中提取对象识别信息，并根据当前图像、指令和提取的对象信息来调整机器人的策略。在真实移动机械手上的各种实验中，我们发现MOO将零射击推广到了广泛的新对象类别和环境中。此外，我们还展示了MOO如何推广到其他非基于语言的输入模式，以指定感兴趣的对象，如手指指向，以及如何进一步扩展它以实现开放世界的导航和操作。该项目的网站和评估视频可在https://robot-moo.github.io/.上找到

**[Paper URL](https://proceedings.mlr.press/v229/stone23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/stone23a/stone23a.pdf)** 

# Learning Generalizable Manipulation Policies with Object-Centric 3D Representations
**题目:** 使用以对象为中心的3D表示学习可推广的操纵策略

**作者:** Yifeng Zhu, Zhenyu Jiang, Peter Stone, Yuke Zhu

**Abstract:** We introduce GROOT, an imitation learning method for learning robust policies with object-centric and 3D priors. GROOT builds policies that generalize beyond their initial training conditions for vision-based manipulation. It constructs object-centric 3D representations that are robust toward background changes and camera views and reason over these representations using a transformer-based policy. Furthermore, we introduce a segmentation correspondence model that allows policies to generalize to new objects at test time. Through comprehensive experiments, we validate the robustness of GROOT policies against perceptual variations in simulated and real-world environments. GROOT’s performance excels in generalization over background changes, camera viewpoint shifts, and the presence of new object instances, whereas both state-of-the-art end-to-end learning methods and object proposal-based approaches fall short. We also extensively evaluate GROOT policies on real robots, where we demonstrate the efficacy under very wild changes in setup. More videos and model details can be found in the appendix and the project website https://ut-austin-rpl.github.io/GROOT.

**摘要:** 我们引入了GROOT，这是一种模仿学习方法，用于学习以对象为中心的3D先验的健壮策略。格鲁特建立的策略超越了最初的视觉操纵培训条件。它构建了以对象为中心的3D表示，这些表示对于背景变化和相机视图是健壮的，并使用基于变压器的策略对这些表示进行推理。此外，我们引入了一个分段对应模型，允许在测试时将策略泛化到新对象。通过综合实验，验证了GROOT策略在模拟和真实环境中对感知变化的稳健性。GROOT的性能优于对背景变化、相机视点变化和新对象实例的泛化，而最先进的端到端学习方法和基于对象建议的方法都存在不足。我们还在真实机器人上广泛评估了GROOT策略，在那里我们展示了在设置非常剧烈的变化下的有效性。更多视频和型号详细信息可在附录和项目网站https://ut-austin-rpl.github.io/GROOT.中找到

**[Paper URL](https://proceedings.mlr.press/v229/zhu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhu23b/zhu23b.pdf)** 

# AdaptSim: Task-Driven Simulation Adaptation for Sim-to-Real Transfer
**题目:** AdaptSim：任务驱动的模拟自适应实时传输

**作者:** Allen Z. Ren, Hongkai Dai, Benjamin Burchfiel, Anirudha Majumdar

**Abstract:** Simulation parameter settings such as contact models and object geometry approximations are critical to training robust manipulation policies capable of transferring from simulation to real-world deployment. There is often an irreducible gap between simulation and reality: attempting to match the dynamics between simulation and reality may be infeasible and may not lead to policies that perform well in reality for a specific task. We propose AdaptSim, a new task-driven adaptation framework for sim-to-real transfer that aims to optimize task performance in target (real) environments. First, we meta-learn an adaptation policy in simulation using reinforcement learning for adjusting the simulation parameter distribution based on the current policy’s performance in a target environment. We then perform iterative real-world adaptation by inferring new simulation parameter distributions for policy training. Our extensive simulation and hardware experiments demonstrate AdaptSim achieving 1-3x asymptotic performance and 2x real data efficiency when adapting to different environments, compared to methods based on Sys-ID and directly training the task policy in target environments.

**摘要:** 仿真参数设置，例如接触模型和对象几何近似，对于训练能够从仿真转移到真实世界部署的健壮操作策略至关重要。模拟和现实之间往往存在无法缩小的鸿沟：试图匹配模拟和现实之间的动态可能是不可行的，也可能不会导致针对特定任务在现实中表现良好的策略。我们提出了AdaptSim，这是一个新的任务驱动的自适应框架，旨在优化目标(真实)环境中的任务性能。首先，我们使用强化学习在仿真中元学习自适应策略，以根据当前策略在目标环境中的性能来调整仿真参数分布。然后，我们通过推断用于策略训练的新的模拟参数分布来执行迭代的真实世界适应。大量的模拟和硬件实验表明，与基于Sys-ID和直接训练目标环境中的任务策略的方法相比，AdaptSim在适应不同环境时获得了1-3倍的渐近性能和2倍的真实数据效率。

**[Paper URL](https://proceedings.mlr.press/v229/ren23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ren23b/ren23b.pdf)** 

# Dexterous Functional Grasping
**题目:** 灵巧的功能抓取

**作者:** Ananye Agarwal, Shagun Uppal, Kenneth Shaw, Deepak Pathak

**Abstract:** While there have been significant strides in dexterous manipulation, most of it is limited to benchmark tasks like in-hand reorientation which are of limited utility in the real world. The main benefit of dexterous hands over two-fingered ones is their ability to pickup tools and other objects (including thin ones) and grasp them firmly in order to apply force. However, this task requires both a complex understanding of functional affordances as well as precise low-level control. While prior work obtains affordances from human data this approach doesn’t scale to low-level control. Similarly, simulation training cannot give the robot an understanding of real-world semantics. In this paper, we aim to combine the best of both worlds to accomplish functional grasping for in-the-wild objects. We use a modular approach. First, affordances are obtained by matching corresponding regions of different objects and then a low-level policy trained in sim is run to grasp it. We propose a novel application of eigengrasps to reduce the search space of RL using a small amount of human data and find that it leads to more stable and physically realistic motion. We find that eigengrasp action space beats baselines in simulation and outperforms hardcoded grasping in real and matches or outperforms a trained human teleoperator. Videos at https://dexfunc.github.io/.

**摘要:** 虽然在灵巧操作方面已经有了很大的进步，但大多数都局限于基准任务，比如手部重定向，这些任务在现实世界中的实用价值有限。灵巧的手比两指灵巧的手的主要优点是它们能够拿起工具和其他物体(包括细的)，并牢牢抓住它们以施力。然而，这项任务既需要对功能负担的复杂理解，也需要精确的低水平控制。虽然以前的工作是从人类数据中获得负担能力，但这种方法并不适用于低水平的控制。同样，模拟训练不能让机器人理解真实世界的语义。在本文中，我们的目标是结合两者的优点来实现对野生物体的功能性抓取。我们使用模块化方法。首先，通过匹配不同对象的相应区域来获得支持度，然后运行在SIM中训练的低层策略来把握该支持度。我们提出了一种新的应用，利用少量的人类数据来缩小RL的搜索空间，并发现它导致了更稳定和更真实的运动。我们发现，eigengrasp动作空间在模拟中击败了基线，在真实中超过了硬编码抓取，并与训练有素的人类遥操作机器人相匹配或更好。Https://dexfunc.github.io/.上的视频

**[Paper URL](https://proceedings.mlr.press/v229/agarwal23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/agarwal23a/agarwal23a.pdf)** 

# REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction
**题目:** 反思：总结机器人经验以解释和纠正故障

**作者:** Zeyi Liu, Arpit Bahety, Shuran Song

**Abstract:** The ability to detect and analyze failed executions automatically is crucial for an explainable and robust robotic system. Recently, Large Language Models (LLMs) have demonstrated strong reasoning abilities on textual inputs. To leverage the power of LLMs for robot failure explanation, we introduce REFLECT, a framework which queries LLM for failure reasoning based on a hierarchical summary of robot past experiences generated from multisensory observations. The failure explanation can further guide a language-based planner to correct the failure and complete the task. To systematically evaluate the framework, we create the RoboFail dataset with a variety of tasks and failure scenarios. We demonstrate that the LLM-based framework is able to generate informative failure explanations that assist successful correction planning.

**摘要:** 自动检测和分析失败执行的能力对于可解释且稳健的机器人系统至关重要。最近，大型语言模型（LLM）在文本输入上表现出了强大的推理能力。为了利用LLM的功能来解释机器人故障，我们引入了RECLECT，这是一个框架，该框架基于多感官观察生成的机器人过去经验的分层总结，向LLM查询故障推理。失败解释可以进一步指导基于语言的规划者纠正失败并完成任务。为了系统性地评估该框架，我们创建了包含各种任务和故障场景的RoboFail数据集。我们证明，基于LLM的框架能够生成信息丰富的故障解释，以帮助成功的纠正规划。

**[Paper URL](https://proceedings.mlr.press/v229/liu23g.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23g/liu23g.pdf)** 

# Task Generalization with Stability Guarantees via Elastic Dynamical System Motion Policies
**题目:** 通过弹性动态系统运动策略实现具有稳定性保证的任务概括

**作者:** Tianyu Li, Nadia Figueroa

**Abstract:** Dynamical System (DS) based Learning from Demonstration (LfD) allows learning of reactive motion policies with stability and convergence guarantees from a few trajectories. Yet, current DS learning techniques lack the flexibility to generalize to new task instances as they overlook explicit task parameters that inherently change the underlying demonstrated trajectories. In this work, we propose Elastic-DS, a novel DS learning and generalization approach that embeds task parameters into the Gaussian Mixture Model (GMM) based Linear Parameter Varying (LPV) DS formulation. Central to our approach is the Elastic-GMM, a GMM constrained to SE(3) task-relevant frames. Given a new task instance/context, the Elastic-GMM is transformed with Laplacian Editing and used to re-estimate the LPV-DS policy. Elastic-DS is compositional in nature and can be used to construct flexible multi-step tasks. We showcase its strength on a myriad of simulated and real-robot experiments while preserving desirable control-theoretic guarantees.

**摘要:** 基于动态系统(DS)的从演示中学习(LFD)允许从几个轨迹学习具有稳定性和收敛保证的反应性运动策略。然而，当前的DS学习技术缺乏推广到新任务实例的灵活性，因为它们忽略了内在地改变潜在演示轨迹的显式任务参数。在这项工作中，我们提出了弹性DS，一种新的DS学习和泛化方法，它将任务参数嵌入到基于高斯混合模型(GMM)的线性参数变化(LPV)DS公式中。我们方法的核心是弹性GMM，这是一种受限于SE(3)任务相关框架的GMM。在给定新的任务实例/上下文的情况下，使用拉普拉斯编辑来转换弹性-GMM，并使用它来重新估计LPV-DS策略。弹性-DS本质上是构成性的，可以用来构建灵活的多步骤任务。我们在无数的模拟和真实机器人实验中展示了它的优势，同时保留了理想的控制理论保证。

**[Paper URL](https://proceedings.mlr.press/v229/li23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/li23b/li23b.pdf)** 

# Push Past Green: Learning to Look Behind Plant Foliage by Moving It
**题目:** 超越绿色：学会通过移动植物叶子来寻找植物叶子的背后

**作者:** Xiaoyu Zhang, Saurabh Gupta

**Abstract:** Autonomous agriculture applications (e.g., inspection, phenotyping, plucking fruits) require manipulating the plant foliage to look behind the leaves and the branches. Partial visibility, extreme clutter, thin structures, and unknown geometry and dynamics for plants make such manipulation challenging. We tackle these challenges through data-driven methods. We use self-supervision to train SRPNet, a neural network that predicts what space is revealed on execution of a candidate action on a given plant. We use SRPNet with the cross-entropy method to predict actions that are effective at revealing space beneath plant foliage. Furthermore, as SRPNet does not just predict how much space is revealed but also where it is revealed, we can execute a sequence of actions that incrementally reveal more and more space beneath the plant foliage. We experiment with a synthetic (vines) and a real plant (Dracaena) on a physical test-bed across 5 settings including 2 settings that test generalization to novel plant configurations. Our experiments reveal the effectiveness of our overall method, PPG, over a competitive hand-crafted exploration method, and the effectiveness of SRPNet over a hand-crafted dynamics model and relevant ablations. Project website with execution videos, code, data, and models: https://sites.google.com/view/pushingfoliage/.

**摘要:** 自主农业应用(例如，检查、表型鉴定、采摘水果)需要操纵植物的叶子来观察树叶和树枝的后面。植物的局部可见性、极端杂乱、薄薄的结构以及未知的几何和动力学使这种操作具有挑战性。我们通过数据驱动的方法应对这些挑战。我们使用自我监督来训练SRPNet，这是一个神经网络，它预测在给定工厂上执行候选操作时会显示出什么空间。我们使用SRPNet和交叉熵方法来预测在揭示植物叶片下面的空间方面有效的动作。此外，由于SRPNet不仅预测有多少空间被揭示，而且还预测它被揭示在哪里，我们可以执行一系列行动，逐步揭示植物叶子下越来越多的空间。我们在物理试验台上对合成的(藤本植物)和真实的植物(龙血树)进行了实验，涉及5个设置，其中包括2个测试对新植物配置的泛化的设置。我们的实验表明，我们的整体方法PPG的有效性优于竞争对手手工制作的勘探方法，以及SRPNet在手工制作的动力学模型和相关消融上的有效性。包含执行视频、代码、数据和模型的项目网站：https://sites.google.com/view/pushingfoliage/.

**[Paper URL](https://proceedings.mlr.press/v229/zhang23k.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhang23k/zhang23k.pdf)** 

# XSkill: Cross Embodiment Skill Discovery
**题目:** XSkill：交叉体现技能发现

**作者:** Mengda Xu, Zhenjia Xu, Cheng Chi, Manuela Veloso, Shuran Song

**Abstract:** Human demonstration videos are a widely available data source for robot learning and an intuitive user interface for expressing desired behavior. However, directly extracting reusable robot manipulation skills from unstructured human videos is challenging due to the big embodiment difference and unobserved action parameters. To bridge this embodiment gap, this paper introduces XSkill, an imitation learning framework that 1) discovers a cross-embodiment representation called skill prototypes purely from unlabeled human and robot manipulation videos, 2) transfers the skill representation to robot actions using conditional diffusion policy, and finally, 3) composes the learned skill to accomplish unseen tasks specified by a human prompt video. Our experiments in simulation and real-world environments show that the discovered skill prototypes facilitate both skill transfer and composition for unseen tasks, resulting in a more general and scalable imitation learning framework.

**摘要:** 人类演示视频是机器人学习的广泛可用的数据源，也是表达所需行为的直观用户界面。然而，由于存在很大的实施差异和不可观察的动作参数，从非结构化人类视频中直接提取可重复使用的机器人操作技能具有挑战性。为了弥合这一实施例差距，本文引入了XSkill，这是一个模仿学习框架，它1）纯粹从未标记的人类和机器人操纵视频中发现了一种称为技能原型的跨实施例表示，2）使用条件扩散策略将技能表示转移到机器人动作，最后，3）编写学习的技能来完成人类提示视频指定的未见任务。我们在模拟和现实世界环境中的实验表明，发现的技能原型促进了不可见任务的技能转移和合成，从而形成了更通用和可扩展的模仿学习框架。

**[Paper URL](https://proceedings.mlr.press/v229/xu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/xu23a/xu23a.pdf)** 

# SayTap: Language to Quadrupedal Locomotion
**题目:** SayTap：四足运动语言

**作者:** Yujin Tang, Wenhao Yu, Jie Tan, Heiga Zen, Aleksandra Faust, Tatsuya Harada

**Abstract:** Large language models (LLMs) have demonstrated the potential to perform high-level planning. Yet, it remains a challenge for LLMs to comprehend low-level commands, such as joint angle targets or motor torques. This paper proposes an approach to use foot contact patterns as an interface that bridges human commands in natural language and a locomotion controller that outputs these low-level commands. This results in an interactive system for quadrupedal robots that allows the users to craft diverse locomotion behaviors flexibly. We contribute an LLM prompt design, a reward function, and a method to expose the controller to the feasible distribution of contact patterns. The results are a controller capable of achieving diverse locomotion patterns that can be transferred to real robot hardware. Compared with other design choices, the proposed approach enjoys more than $50%$ success rate in predicting the correct contact patterns and can solve 10 more tasks out of a total of 30 tasks. (https://saytap.github.io)

**摘要:** 大型语言模型(LLM)已经证明了执行高级规划的潜力。然而，对于LLMS来说，理解诸如关节角度目标或马达扭矩等低级命令仍然是一个挑战。本文提出了一种使用脚接触模式作为连接人类自然语言命令和输出这些低级命令的运动控制器的接口的方法。这导致了一个用于四足机器人的交互系统，允许用户灵活地设计不同的运动行为。我们给出了一个LLM提示设计，一个奖励函数，以及一种使控制器接触模式可行分布的方法。其结果是一种能够实现多种运动模式的控制器，这些运动模式可以传输到真实的机器人硬件上。与其他设计方案相比，该方法在预测正确的联系模式方面具有超过50%的成功率，并且可以在总共30个任务中多解决10个任务。(https://saytap.github.io))

**[Paper URL](https://proceedings.mlr.press/v229/tang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/tang23a/tang23a.pdf)** 

# SLAP: Spatial-Language Attention Policies
**题目:** 空间语言关注政策

**作者:** Priyam Parashar, Vidhi Jain, Xiaohan Zhang, Jay Vakil, Sam Powers, Yonatan Bisk, Chris Paxton

**Abstract:** Despite great strides in language-guided manipulation, existing work has been constrained to table-top settings. Table-tops allow for perfect and consistent camera angles, properties are that do not hold in mobile manipulation. Task plans that involve moving around the environment must be robust to egocentric views and changes in the plane and angle of grasp. A further challenge is ensuring this is all true while still being able to learn skills efficiently from limited data. We propose Spatial-Language Attention Policies (SLAP) as a solution. SLAP uses three-dimensional tokens as the input representation to train a single multi-task, language-conditioned action prediction policy. Our method shows an $80%$ success rate in the real world across eight tasks with a single model, and a $47.5%$ success rate when unseen clutter and unseen object configurations are introduced, even with only a handful of examples per task. This represents an improvement of $30%$ over prior work ($20%$ given unseen distractors and configurations). We see a 4x improvement over baseline in mobile manipulation setting. In addition, we show how SLAPs robustness allows us to execute Task Plans from open-vocabulary instructions using a large language model for multi-step mobile manipulation. For videos, see the website: https://robotslap.github.io

**摘要:** 尽管在语言引导操作方面取得了很大进展，但现有的工作一直局限于桌面设置。桌面允许完美和一致的相机角度，这是移动操作中所不具备的特性。涉及在环境中移动的任务计划必须对以自我为中心的观点以及抓取平面和角度的变化保持稳健。另一个挑战是确保所有这些都是真实的，同时仍然能够从有限的数据中高效地学习技能。我们提出了空间语言注意策略(SLAP)作为解决方案。SLAP使用三维标记作为输入表示来训练单个多任务、受语言限制的动作预测策略。我们的方法显示，在现实世界中，使用单一模型的八项任务的成功率为$80%$，当引入看不见的杂乱和看不见的对象配置时，成功率为$47.5%$，即使每个任务只有几个例子。这比之前的工作改进了$30%$(考虑到看不见的干扰和配置，$20%$)。我们看到在移动操作设置方面比基线提高了4倍。此外，我们还展示了SLAPS的健壮性如何允许我们使用用于多步骤移动操作的大型语言模型从开放词汇指令执行任务计划。有关视频，请访问网站：https://robotslap.github.io

**[Paper URL](https://proceedings.mlr.press/v229/parashar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/parashar23a/parashar23a.pdf)** 

# Learning Human Contribution Preferences in Collaborative Human-Robot Tasks
**题目:** 学习人机协作任务中人类贡献偏好

**作者:** Michelle D Zhao, Reid Simmons, Henny Admoni

**Abstract:** In human-robot collaboration, both human and robotic agents must work together to achieve a set of shared objectives. However, each team member may have individual preferences, or constraints, for how they would like to contribute to the task. Effective teams align their actions to optimize task performance while satisfying each team member’s constraints to the greatest extent possible. We propose a framework for representing human and robot contribution constraints in collaborative human-robot tasks. Additionally, we present an approach for learning a human partner’s contribution constraint online during a collaborative interaction. We evaluate our approach using a variety of simulated human partners in a collaborative decluttering task. Our results demonstrate that our method improves team performance over baselines with some, but not all, simulated human partners. Furthermore, we conducted a pilot user study to gather preliminary insights into the effectiveness of our approach on task performance and collaborative fluency. Preliminary results suggest that pilot users performed fluently with our method, motivating further investigation into considering preferences that emerge from collaborative interactions.

**摘要:** 在人-机器人协作中，人类和机器人智能体必须共同努力，以实现一组共同的目标。然而，每个团队成员对于他们希望如何贡献任务可能有各自的偏好或限制。有效的团队协调他们的行动，以优化任务绩效，同时最大限度地满足每个团队成员的限制。提出了一种在人机协同任务中表示人和机器人贡献约束的框架。此外，我们还提出了一种在协作交互过程中在线学习人类伙伴的贡献约束的方法。我们在协作清理任务中使用了各种模拟的人类伙伴来评估我们的方法。我们的结果表明，我们的方法在一些(但不是全部)模拟人类合作伙伴的基线上提高了团队绩效。此外，我们进行了一项试点用户研究，以收集对我们的方法在任务绩效和协作流畅性方面的有效性的初步见解。初步结果表明，试点用户使用我们的方法表现流畅，这激发了进一步的研究，以考虑从协作交互中产生的偏好。

**[Paper URL](https://proceedings.mlr.press/v229/zhao23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhao23b/zhao23b.pdf)** 

# M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place
**题目:** M2 T2：用于以对象为中心的拾取和放置的多任务掩蔽Transformer

**作者:** Wentao Yuan, Adithyavairavan Murali, Arsalan Mousavian, Dieter Fox

**Abstract:** With the advent of large language models and large-scale robotic datasets, there has been tremendous progress in high-level decision-making for object manipulation. These generic models are able to interpret complex tasks using language commands, but they often have difficulties generalizing to out-of-distribution objects due to the inability of low-level action primitives. In contrast, existing task-specific models excel in low-level manipulation of unknown objects, but only work for a single type of action. To bridge this gap, we present M2T2, a single model that supplies different types of low-level actions that work robustly on arbitrary objects in cluttered scenes. M2T2 is a transformer model which reasons about contact points and predicts valid gripper poses for different action modes given a raw point cloud of the scene. Trained on a large-scale synthetic dataset with 128K scenes, M2T2 achieves zero-shot sim2real transfer on the real robot, outperforming the baseline system with state-of-the-art task-specific models by about $19%$ in overall performance and $37.5%$ in challenging scenes were the object needs to be re-oriented for collision-free placement. M2T2 also achieves state-of-the-art results on a subset of language conditioned tasks in RLBench. Videos of robot experiments on unseen objects in both real world and simulation are available at m2-t2.github.io.

**摘要:** 随着大型语言模型和大规模机器人数据集的出现，在对象操作的高层决策方面取得了巨大的进步。这些通用模型能够使用语言命令解释复杂的任务，但由于无法使用低级操作原语，它们通常难以泛化到分布之外的对象。相比之下，现有的特定于任务的模型擅长于对未知对象的低级操作，但仅适用于单一类型的操作。为了弥补这一差距，我们提出了M2T2，这是一个单一的模型，它提供了不同类型的低级操作，可以在混乱场景中的任意对象上健壮地工作。M2T2是一个变形模型，它可以对接触点进行推理，并在给定场景的原始点云的情况下，为不同的动作模式预测有效的抓手姿势。在具有128K场景的大规模合成数据集上进行训练，M2T2在真实机器人上实现了零镜头sim2Real传输，在总体性能上比采用最先进任务特定模型的基准系统性能高出约19%$，在具有挑战性的场景中性能高出约37.5%$，因为对象需要重新定位以实现无碰撞放置。M2T2还在RLBitch语言条件化任务的子集上实现了最先进的结果。机器人在现实世界和模拟世界中对看不见的物体进行实验的视频可以在m2-t2.githeb.io上获得。

**[Paper URL](https://proceedings.mlr.press/v229/yuan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/yuan23a/yuan23a.pdf)** 

# Learning to Drive Anywhere
**题目:** 学习在任何地方开车

**作者:** Ruizhao Zhu, Peng Huang, Eshed Ohn-Bar, Venkatesh Saligrama

**Abstract:** Human drivers can seamlessly adapt their driving decisions across geographical locations with diverse conditions and rules of the road, e.g., left vs. right-hand traffic. In contrast, existing models for autonomous driving have been thus far only deployed within restricted operational domains, i.e., without accounting for varying driving behaviors across locations or model scalability. In this work, we propose GeCo, a single geographically-aware conditional imitation learning (CIL) model that can efficiently learn from heterogeneous and globally distributed data with dynamic environmental, traffic, and social characteristics. Our key insight is to introduce a high-capacity, geo-location-based channel attention mechanism that effectively adapts to local nuances while also flexibly modeling similarities among regions in a data-driven manner. By optimizing a contrastive imitation objective, our proposed approach can efficiently scale across the inherently imbalanced data distributions and location-dependent events. We demonstrate the benefits of our GeCo agent across multiple datasets, cities, and scalable deployment paradigms, i.e., centralized, semi-supervised, and distributed agent training. Specifically, GeCo outperforms CIL baselines by over $14%$ in open-loop evaluation and $30%$ in closed-loop testing on CARLA.

**摘要:** 人类驾驶员可以无缝地在具有不同条件和道路规则的地理位置上调整他们的驾驶决定，例如，左交通和右交通。相比之下，到目前为止，用于自动驾驶的现有模型仅部署在受限的操作域内，即没有考虑跨位置的不同驾驶行为或模型可伸缩性。在这项工作中，我们提出了GECO，一个单一的地理感知条件模仿学习(CIL)模型，可以有效地从具有动态环境、交通和社会特征的异质和全球分布的数据中学习。我们的主要见解是引入一种基于地理位置的大容量通道注意机制，该机制有效地适应局部细微差别，同时以数据驱动的方式灵活地模拟区域之间的相似性。通过优化对比模仿目标，我们提出的方法可以有效地扩展固有的不平衡的数据分布和位置相关的事件。我们展示了我们的GECO代理在多个数据集、城市和可扩展部署范例(即集中式、半监督和分布式代理培训)中的优势。具体地说，GECO在开环评估中的表现比CIL基线高出14%$，在CALA上的闭环测试中超过了30%$。

**[Paper URL](https://proceedings.mlr.press/v229/zhu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/zhu23c/zhu23c.pdf)** 

# MOTO: Offline Pre-training to Online Fine-tuning for Model-based Robot Learning
**题目:** MOTO：基于模型的机器人学习的离线预训练到在线微调

**作者:** Rafael Rafailov, Kyle Beltran Hatch, Victor Kolev, John D. Martin, Mariano Phielipp, Chelsea Finn

**Abstract:** We study the problem of offline pre-training and online fine-tuning for reinforcement learning from high-dimensional observations in the context of realistic robot tasks. Recent offline model-free approaches successfully use online fine-tuning to either improve the performance of the agent over the data collection policy or adapt to novel tasks. At the same time, model-based RL algorithms have achieved significant progress in sample efficiency and the complexity of the tasks they can solve, yet remain under-utilized in the fine-tuning setting. In this work, we argue that existing methods for high-dimensional model-based offline RL are not suitable for offline-to-online fine-tuning due to issues with distribution shifts, off-dynamics data, and non-stationary rewards. We propose an on-policy model-based method that can efficiently reuse prior data through model-based value expansion and policy regularization, while preventing model exploitation by controlling epistemic uncertainty. We find that our approach successfully solves tasks from the MetaWorld benchmark, as well as the Franka Kitchen robot manipulation environment completely from images. To our knowledge, MOTO is the first and only method to solve this environment from pixels.

**摘要:** 本文结合实际机器人任务，研究了基于高维观测的强化学习离线预训练和在线微调问题。最近的离线无模型方法成功地使用在线微调来提高代理在数据收集策略上的性能或适应新的任务。与此同时，基于模型的RL算法在样本效率和所能解决的任务的复杂性方面取得了显著的进步，但在微调环境下仍未得到充分利用。在这项工作中，我们认为现有的基于高维模型的离线RL方法不适合从离线到在线的微调，因为存在分布平移、非动态数据和非平稳回报的问题。我们提出了一种基于策略模型的方法，该方法通过基于模型的值扩展和策略正规化来有效地重用先验数据，同时通过控制认知不确定性来防止模型利用。我们发现，我们的方法成功地解决了MetaWorld基准测试以及Franka Kitchen机器人操作环境中完全来自图像的任务。据我们所知，MOTO是第一个也是唯一一个从像素解决这个环境的方法。

**[Paper URL](https://proceedings.mlr.press/v229/rafailov23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/rafailov23a/rafailov23a.pdf)** 

# Ready, Set, Plan! Planning to Goal Sets Using Generalized Bayesian Inference
**题目:** 准备、设置、计划！使用广义Bayesian推理规划目标集

**作者:** Jana Pavlasek, Stanley Robert Lewis, Balakumar Sundaralingam, Fabio Ramos, Tucker Hermans

**Abstract:** Many robotic tasks can have multiple and diverse solutions and, as such, are naturally expressed as goal sets. Examples include navigating to a room, finding a feasible placement location for an object, or opening a drawer enough to reach inside. Using a goal set as a planning objective requires that a model for the objective be explicitly given by the user. However, some goals are intractable to model, leading to uncertainty over the goal (e.g. stable grasping of an object). In this work, we propose a technique for planning directly to a set of sampled goal configurations. We formulate a planning as inference problem with a novel goal likelihood evaluated against the goal samples. To handle the intractable goal likelihood, we employ Generalized Bayesian Inference to approximate the trajectory distribution. The result is a fully differentiable cost which generalizes across a diverse range of goal set objectives for which samples can be obtained. We show that by considering all goal samples throughout the planning process, our method reliably finds plans on manipulation and navigation problems where heuristic approaches fail.

**摘要:** 许多机器人任务可以有多个不同的解决方案，因此，自然地表示为目标集。例如，导航到一个房间，为一个物体找到一个可行的放置位置，或者打开一个足够够到里面的抽屉。使用目标集作为规划目标需要用户明确给出目标的模型。然而，有些目标很难建模，导致对目标的不确定性(例如，对对象的稳定把握)。在这项工作中，我们提出了一种直接规划一组采样目标配置的技术。我们将规划问题描述为推理问题，并根据目标样本评估一个新的目标似然。为了处理难以处理的目标似然性，我们使用广义贝叶斯推理来近似轨迹分布。其结果是一种完全可区分的成本，它概括了可以获得样本的各种目标设定目标。我们表明，通过在整个规划过程中考虑所有目标样本，我们的方法能够可靠地找到启发式方法失败的操纵和导航问题的计划。

**[Paper URL](https://proceedings.mlr.press/v229/pavlasek23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/pavlasek23a/pavlasek23a.pdf)** 

# Online Model Adaptation with Feedforward Compensation
**题目:** 具有前向补偿的在线模型适应

**作者:** Abulikemu Abuduweili, Changliu Liu

**Abstract:** To cope with distribution shifts or non-stationarity in system dynamics, online adaptation algorithms have been introduced to update offline-learned prediction models in real-time. Existing online adaptation methods focus on optimizing the prediction model by utilizing feedback from the latest prediction error. Unfortunately, this feedback-based approach is susceptible to forgetting past information. This work proposes an online adaptation method with feedforward compensation, which uses critical data samples from a memory buffer, instead of the latest samples, to optimize the prediction model. We prove that the proposed approach achieves a smaller error bound compared to previously utilized methods in slow time-varying systems. We conducted experiments on several prediction tasks, which clearly illustrate the superiority of the proposed feedforward adaptation method. Furthermore, our feedforward adaptation technique is capable of estimating an uncertainty bound for predictions.

**摘要:** 为了应对系统动态中的分布变化或非平稳性，引入了在线自适应算法来实时更新离线学习的预测模型。现有的在线适应方法专注于通过利用最新预测误差的反馈来优化预测模型。不幸的是，这种基于反馈的方法很容易忘记过去的信息。这项工作提出了一种具有前向补偿的在线自适应方法，该方法使用来自存储缓冲区的关键数据样本而不是最新样本来优化预测模型。我们证明，与之前在慢时变系统中使用的方法相比，所提出的方法实现了更小的误差界限。我们对几项预测任务进行了实验，这清楚地说明了所提出的前向自适应方法的优越性。此外，我们的反馈自适应技术能够估计预测的不确定性界限。

**[Paper URL](https://proceedings.mlr.press/v229/abuduweili23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/abuduweili23a/abuduweili23a.pdf)** 

# Generating Transferable Adversarial Simulation Scenarios for Self-Driving via Neural Rendering
**题目:** 通过神经渲染生成可转移的自动驾驶对抗模拟场景

**作者:** Yasasa Abeysirigoonawardena, Kevin Xie, Chuhan Chen, Salar Hosseini Khorasgani, Ruiting Chen, Ruiqi Wang, Florian Shkurti

**Abstract:** Self-driving software pipelines include components that are learned from a significant number of training examples, yet it remains challenging to evaluate the overall system’s safety and generalization performance. Together with scaling up the real-world deployment of autonomous vehicles, it is of critical importance to automatically find simulation scenarios where the driving policies will fail. We propose a method that efficiently generates adversarial simulation scenarios for autonomous driving by solving an optimal control problem that aims to maximally perturb the policy from its nominal trajectory. Given an image-based driving policy, we show that we can inject new objects in a neural rendering representation of the deployment scene, and optimize their texture in order to generate adversarial sensor inputs to the policy. We demonstrate that adversarial scenarios discovered purely in the neural renderer (surrogate scene) can often be successfully transferred to the deployment scene, without further optimization. We demonstrate this transfer occurs both in simulated and real environments, provided the learned surrogate scene is sufficiently close to the deployment scene.

**摘要:** 自动驾驶软件管道包括从大量训练实例中学习的组件，但评估整个系统的安全性和泛化性能仍然具有挑战性。随着自动驾驶汽车在现实世界中的大规模部署，自动找到驾驶策略将失败的模拟场景至关重要。我们提出了一种方法，通过求解最优控制问题来有效地生成自动驾驶的对抗性模拟场景，该最优控制问题旨在最大限度地扰动策略的名义轨迹。在给定基于图像的驾驶策略的情况下，我们证明了我们可以在部署场景的神经渲染表示中注入新对象，并优化它们的纹理，以便为策略生成对抗性传感器输入。我们证明了纯粹在神经呈现器(代理场景)中发现的对抗性场景通常可以成功地转移到部署场景，而不需要进一步的优化。我们演示了这种传输在模拟和真实环境中都发生，前提是学习的代理场景足够接近部署场景。

**[Paper URL](https://proceedings.mlr.press/v229/abeysirigoonawardena23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/abeysirigoonawardena23a/abeysirigoonawardena23a.pdf)** 

# STOW: Discrete-Frame Segmentation and Tracking of Unseen Objects for Warehouse Picking Robots
**题目:** STOW：仓库拣货机器人的不可见物体的离散帧分割和跟踪

**作者:** Yi Li, Muru Zhang, Markus Grotz, Kaichun Mo, Dieter Fox

**Abstract:** Segmentation and tracking of unseen object instances in discrete frames pose a significant challenge in dynamic industrial robotic contexts, such as distribution warehouses. Here, robots must handle object rearrangements, including shifting, removal, and partial occlusion by new items, and track these items after substantial temporal gaps. The task is further complicated when robots encounter objects beyond their training sets, thereby requiring the ability to segment and track previously unseen items. Considering that continuous observation is often inaccessible in such settings, our task involves working with a discrete set of frames separated by indefinite periods, during which substantial changes to the scene may occur. This task also translates to domestic robotic applications, such as table rearrangement. To address these demanding challenges, we introduce new synthetic and real-world datasets that replicate these industrial and household scenarios. Furthermore, we propose a novel paradigm for joint segmentation and tracking in discrete frames, alongside a transformer module that facilitates efficient inter-frame communication. Our approach significantly outperforms recent methods in our experiments. For additional results and videos, please visit https://sites.google.com/view/stow-corl23. Code and dataset will be released.

**摘要:** 在动态工业机器人环境中，如配送仓库，分割和跟踪离散框架中的不可见对象实例是一个巨大的挑战。在这里，机器人必须处理对象的重新排列，包括移动、移除和新项目的部分遮挡，并在显著的时间间隔后跟踪这些项目。当机器人遇到它们训练集之外的物体时，任务就变得更加复杂，从而需要分割和跟踪以前未见过的物体。考虑到在这样的设置中通常无法进行连续观察，我们的任务涉及处理由不确定时间段分隔的一组离散帧，在此期间可能会发生场景的重大变化。这项任务也适用于国内的机器人应用，例如桌子的重新排列。为了应对这些苛刻的挑战，我们引入了复制这些工业和家庭场景的新的合成和真实世界数据集。此外，我们还提出了一种新的用于离散帧中的联合分割和跟踪的范例，以及一个促进帧间高效通信的转换模块。在我们的实验中，我们的方法远远超过了最近的方法。欲了解更多结果和视频，请访问https://sites.google.com/view/stow-corl23.。代码和数据集将被发布。

**[Paper URL](https://proceedings.mlr.press/v229/li23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/li23c/li23c.pdf)** 

# DORT: Modeling Dynamic Objects in Recurrent for Multi-Camera 3D Object Detection and Tracking
**题目:** DORT：循环建模动态对象，用于多摄像机3D对象检测和跟踪

**作者:** Qing LIAN, Tai Wang, Dahua Lin, Jiangmiao Pang

**Abstract:** Recent multi-camera 3D object detectors usually leverage temporal information to construct multi-view stereo that alleviates the ill-posed depth estimation. However, they typically assume all the objects are static and directly aggregate features across frames. This work begins with a theoretical and empirical analysis to reveal that ignoring the motion of moving objects can result in serious localization bias. Therefore, we propose to model Dynamic Objects in RecurrenT (DORT) to tackle this problem. In contrast to previous global BirdEye-View (BEV) methods, DORT extracts object-wise local volumes for motion estimation that also alleviates the heavy computational burden. By iteratively refining the estimated object motion and location, the preceding features can be precisely aggregated to the current frame to mitigate the aforementioned adverse effects. The simple framework has two significant appealing properties. It is flexible and practical that can be plugged into most camera-based 3D object detectors. As there are predictions of object motion in the loop, it can easily track objects across frames according to their nearest center distances. Without bells and whistles, DORT outperforms all the previous methods on the nuScenes detection and tracking benchmarks with $62.8%$ NDS and $57.6%$ AMOTA, respectively. The source code will be available at https://github.com/OpenRobotLab/DORT.

**摘要:** 目前的多摄像机3D目标检测器通常利用时间信息来构建多视点立体，从而缓解了不适定的深度估计。但是，它们通常假设所有对象都是静态的，并直接跨框架聚合要素。这项工作从理论和实证分析开始，揭示了忽视运动对象的运动会导致严重的定位偏差。因此，我们提出用递归模型(DORT)对动态对象进行建模来解决这个问题。与以往的全局BirdEye-View(BEV)方法不同，DORT方法提取对象的局部体积用于运动估计，也减轻了沉重的计算负担。通过迭代地改进估计的对象运动和位置，可以将先前的特征精确地聚集到当前帧，以减轻上述不利影响。这个简单的框架有两个重要的吸引人的属性。它灵活实用，可以插入大多数基于相机的3D对象探测器中。由于循环中有对象运动的预测，因此它可以根据对象最近的中心距离轻松地跨帧跟踪对象。在没有花哨的情况下，dot在nuScenes检测和跟踪基准测试上的性能分别为$62.8%$NDS和$57.6%$AMOTA。源代码将在https://github.com/OpenRobotLab/DORT.上提供

**[Paper URL](https://proceedings.mlr.press/v229/lian23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/lian23a/lian23a.pdf)** 

# Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition
**题目:** 放大和提炼：数字引导机器人技能获取

**作者:** Huy Ha, Pete Florence, Shuran Song

**Abstract:** We present a framework for robot skill acquisition, which 1) efficiently scale up data generation of language-labelled robot data and 2) effectively distills this data down into a robust multi-task language-conditioned visuo-motor policy. For (1), we use a large language model (LLM) to guide high-level planning, and sampling-based robot planners (e.g. motion or grasp samplers) for generating diverse and rich manipulation trajectories. To robustify this data-collection process, the LLM also infers a code-snippet for the success condition of each task, simultaneously enabling the data-collection process to detect failure and retry as well as the automatic labeling of trajectories with success/failure. For (2), we extend the diffusion policy single-task behavior-cloning approach to multi-task settings with language conditioning. Finally, we propose a new multi-task benchmark with 18 tasks across five domains to test long-horizon behavior, common-sense reasoning, tool-use, and intuitive physics. We find that our distilled policy successfully learned the robust retrying behavior in its data collection procedure, while improving absolute success rates by $33.2%$ on average across five domains. Code, data, and additional qualitative results are available on https://www.cs.columbia.edu/ huy/scalingup/.

**摘要:** 我们提出了一个机器人技能获取的框架，它1)有效地放大语言标记的机器人数据的数据生成，2)有效地将这些数据提取到一个健壮的多任务语言条件视觉-运动策略中。对于(1)，我们使用大型语言模型(LLM)来指导高层规划，并使用基于采样的机器人规划器(例如运动采样器或抓取采样器)来生成多样化和丰富的操作轨迹。为了使这一数据收集过程更具实用性，LLM还为每项任务的成功条件推断代码片段，同时使数据收集过程能够检测失败和重试，并自动标记成功/失败的轨迹。对于(2)，我们将扩散策略单任务行为克隆方法扩展到具有语言条件作用的多任务设置。最后，我们提出了一个新的多任务基准，包括五个领域的18个任务，以测试长期行为、常识性推理、工具使用和直观物理。我们发现，我们的提炼策略在其数据收集过程中成功地学习了健壮的重试行为，同时在五个域中平均提高了33.2%$的绝对成功率。有关代码、数据和其他定性结果，请访问https://www.cs.columbia.edu/：huy/scalingup/。

**[Paper URL](https://proceedings.mlr.press/v229/ha23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ha23a/ha23a.pdf)** 

# Marginalized Importance Sampling for Off-Environment Policy Evaluation
**题目:** 非环境政策评估的边缘重要性抽样

**作者:** Pulkit Katdare, Nan Jiang, Katherine Rose Driggs-Campbell

**Abstract:** Reinforcement Learning (RL) methods are typically sample-inefficient, making it challenging to train and deploy RL-policies in real world robots. Even a robust policy trained in simulation requires a real-world deployment to assess their performance. This paper proposes a new approach to evaluate the real-world performance of agent policies prior to deploying them in the real world. Our approach incorporates a simulator along with real-world offline data to evaluate the performance of any policy using the framework of Marginalized Importance Sampling (MIS). Existing MIS methods face two challenges: (1) large density ratios that deviate from a reasonable range and (2) indirect supervision, where the ratio needs to be inferred indirectly, thus exacerbating estimation error. Our approach addresses these challenges by introducing the target policy’s occupancy in the simulator as an intermediate variable and learning the density ratio as the product of two terms that can be learned separately. The first term is learned with direct supervision and the second term has a small magnitude, thus making it computationally efficient. We analyze the sample complexity as well as error propagation of our two step-procedure. Furthermore, we empirically evaluate our approach on Sim2Sim environments such as Cartpole, Reacher, and Half-Cheetah. Our results show that our method generalizes well across a variety of Sim2Sim gap, target policies and offline data collection policies. We also demonstrate the performance of our algorithm on a Sim2Real task of validating the performance of a 7 DoF robotic arm using offline data along with the Gazebo simulator.

**摘要:** 强化学习(RL)方法通常是样本效率低的，这使得在现实世界的机器人中训练和部署RL策略具有挑战性。即使是经过模拟训练的强大政策也需要进行真实世界的部署来评估其性能。本文提出了一种新的方法，在将代理策略部署到现实世界之前，评估它们在现实世界的性能。我们的方法结合了模拟器和真实世界的离线数据来评估任何政策的性能，使用边缘化重要性抽样(MIS)的框架。现有的管理信息系统方法面临两个挑战：(1)较大的密度比偏离合理范围；(2)间接监督，需要间接推断密度比，从而加剧估计误差。我们的方法通过引入目标策略在模拟器中的占用率作为中间变量，并将密度比学习为可以分别学习的两个项的乘积，从而解决了这些挑战。第一项是在直接监督下学习的，第二项的量值很小，因此计算效率很高。我们分析了这两个步骤的样本复杂度和误差传播。此外，我们在卡特波尔、里奇和半猎豹等Sim2Sim环境上对我们的方法进行了经验评估。我们的结果表明，我们的方法能够很好地适用于各种Sim2Sim缺口、目标策略和离线数据收集策略。我们还展示了我们的算法在Sim2Real任务中的性能，该任务使用离线数据和Gazebo模拟器来验证7自由度机械臂的性能。

**[Paper URL](https://proceedings.mlr.press/v229/katdare23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/katdare23a/katdare23a.pdf)** 

# Policy Stitching: Learning Transferable Robot Policies
**题目:** 政策缝合：学习可转移机器人政策

**作者:** Pingcheng Jian, Easop Lee, Zachary Bell, Michael M. Zavlanos, Boyuan Chen

**Abstract:** Training robots with reinforcement learning (RL) typically involves heavy interactions with the environment, and the acquired skills are often sensitive to changes in task environments and robot kinematics. Transfer RL aims to leverage previous knowledge to accelerate learning of new tasks or new body configurations. However, existing methods struggle to generalize to novel robot-task combinations and scale to realistic tasks due to complex architecture design or strong regularization that limits the capacity of the learned policy. We propose Policy Stitching, a novel framework that facilitates robot transfer learning for novel combinations of robots and tasks. Our key idea is to apply modular policy design and align the latent representations between the modular interfaces. Our method allows direct stitching of the robot and task modules trained separately to form a new policy for fast adaptation. Our simulated and real-world experiments on various 3D manipulation tasks demonstrate the superior zero-shot and few-shot transfer learning performances of our method.

**摘要:** 用强化学习(RL)训练机器人通常涉及到与环境的大量交互，并且所获得的技能往往对任务环境和机器人运动学的变化敏感。Transfer RL旨在利用以前的知识来加快学习新任务或新身体配置的速度。然而，由于复杂的体系结构设计或强烈的正则化限制了学习策略的能力，现有的方法难以推广到新的机器人-任务组合并扩展到现实任务。我们提出了策略缝合，这是一个新的框架，促进了机器人迁移学习的新组合的机器人和任务。我们的关键思想是应用模块化策略设计，并对齐模块接口之间的潜在表示。我们的方法允许机器人和单独训练的任务模块直接拼接，形成一种新的快速适应策略。我们在各种3D操作任务上的模拟和真实世界实验表明，该方法具有优越的零镜头和少镜头迁移学习性能。

**[Paper URL](https://proceedings.mlr.press/v229/jian23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/jian23a/jian23a.pdf)** 

# Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation
**题目:** 顺序灵活性：为长期操纵制定灵活性政策

**作者:** Yuanpei Chen, Chen Wang, Li Fei-Fei, Karen Liu

**Abstract:** Many real-world manipulation tasks consist of a series of subtasks that are significantly different from one another. Such long-horizon, complex tasks highlight the potential of dexterous hands, which possess adaptability and versatility, capable of seamlessly transitioning between different modes of functionality without the need for re-grasping or external tools. However, the challenges arise due to the high-dimensional action space of dexterous hand and complex compositional dynamics of the long-horizon tasks. We present Sequential Dexterity, a general system based on reinforcement learning (RL) that chains multiple dexterous policies for achieving long-horizon task goals. The core of the system is a transition feasibility function that progressively finetunes the sub-policies for enhancing chaining success rate, while also enables autonomous policy-switching for recovery from failures and bypassing redundant stages. Despite being trained only in simulation with a few task objects, our system demonstrates generalization capability to novel object shapes and is able to zero-shot transfer to a real-world robot equipped with a dexterous hand. Code and videos are available at https://sequential-dexterity.github.io.

**摘要:** 许多现实世界中的操作任务由一系列彼此显著不同的子任务组成。这种长期、复杂的任务突出了灵巧手的潜力，它具有适应性和多功能性，能够在不同的功能模式之间无缝转换，而不需要重新抓取或外部工具。然而，由于灵巧手的高维动作空间和长视距任务的复杂组成动力学，这些挑战出现了。我们提出了一种基于强化学习(RL)的通用系统Sequential Dexterity，它链接了多个灵活的策略以实现长期任务目标。该系统的核心是过渡可行性功能，该功能逐步微调用于提高链接成功率的子策略，同时还支持用于从故障恢复和绕过冗余阶段的自主策略切换。尽管我们只对少数几个任务对象进行了模拟训练，但我们的系统展示了对新对象形状的泛化能力，并能够零射击传输到配备灵巧手的真实机器人上。代码和视频可在https://sequential-dexterity.github.io.上查看

**[Paper URL](https://proceedings.mlr.press/v229/chen23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chen23e/chen23e.pdf)** 

# Deception Game: Closing the Safety-Learning Loop in Interactive Robot Autonomy
**题目:** 欺骗游戏：关闭交互式机器人自治中的安全学习循环

**作者:** Haimin Hu, Zixu Zhang, Kensuke Nakamura, Andrea Bajcsy, Jaime Fernández Fisac

**Abstract:** An outstanding challenge for the widespread deployment of robotic systems like autonomous vehicles is ensuring safe interaction with humans without sacrificing performance. Existing safety methods often neglect the robot’s ability to learn and adapt at runtime, leading to overly conservative behavior. This paper proposes a new closed-loop paradigm for synthesizing safe control policies that explicitly account for the robot’s evolving uncertainty and its ability to quickly respond to future scenarios as they arise, by jointly considering the physical dynamics and the robot’s learning algorithm. We leverage adversarial reinforcement learning for tractable safety analysis under high-dimensional learning dynamics and demonstrate our framework’s ability to work with both Bayesian belief propagation and implicit learning through large pre-trained neural trajectory predictors.

**摘要:** 自动驾驶汽车等机器人系统的广泛部署面临的一个突出挑战是确保与人类的安全互动而不牺牲性能。现有的安全方法常常忽视机器人在运行时学习和适应的能力，导致行为过于保守。本文提出了一种新的闭环范式，用于合成安全控制政策，通过联合考虑物理动力学和机器人的学习算法，明确考虑机器人不断变化的不确定性及其对未来场景出现时快速响应的能力。我们利用对抗性强化学习在多维学习动态下进行易于管理的安全分析，并展示了我们的框架通过大型预训练神经轨迹预测器处理Bayesian信念传播和隐式学习的能力。

**[Paper URL](https://proceedings.mlr.press/v229/hu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/hu23b/hu23b.pdf)** 

# Improving Behavioural Cloning with Positive Unlabeled Learning
**题目:** 通过积极的无标签学习改善行为克隆

**作者:** Qiang Wang, Robert McCarthy, David Cordova Bulens, Kevin McGuinness, Noel E. O’Connor, Francisco Roldan Sanchez, Nico Gürtler, Felix Widmaier, Stephen J. Redmond

**Abstract:** Learning control policies offline from pre-recorded datasets is a promising avenue for solving challenging real-world problems. However, available datasets are typically of mixed quality, with a limited number of the trajectories that we would consider as positive examples; i.e., high-quality demonstrations. Therefore, we propose a novel iterative learning algorithm for identifying expert trajectories in unlabeled mixed-quality robotics datasets given a minimal set of positive examples, surpassing existing algorithms in terms of accuracy. We show that applying behavioral cloning to the resulting filtered dataset outperforms several competitive offline reinforcement learning and imitation learning baselines. We perform experiments on a range of simulated locomotion tasks and on two challenging manipulation tasks on a real robotic system; in these experiments, our method showcases state-of-the-art performance. Our website: https://sites.google.com/view/offline-policy-learning-pubc.

**摘要:** 从预先记录的数据集离线学习控制策略是解决具有挑战性的现实世界问题的一个有希望的途径。然而，可用的数据集通常质量不一，我们认为是积极例子的轨迹数量有限;即，高质量的演示。因此，我们提出了一种新颖的迭代学习算法，用于在未标记的混合质量机器人数据集中识别专家轨迹，给出最少的正例集，在准确性方面超过了现有算法。我们表明，将行为克隆应用于最终的过滤数据集优于几种竞争对手的离线强化学习和模仿学习基线。我们在真实机器人系统上对一系列模拟运动任务和两项具有挑战性的操纵任务进行了实验;在这些实验中，我们的方法展示了最先进的性能。我们的网站：https://sites.google.com/view/offline-policy-learning-pubc。

**[Paper URL](https://proceedings.mlr.press/v229/wang23f.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/wang23f/wang23f.pdf)** 

# $α$-MDF: An Attention-based Multimodal Differentiable Filter for Robot State Estimation
**题目:** $a $-RST：一种用于机器人状态估计的基于注意力的多峰可微过滤器

**作者:** Xiao Liu, Yifan Zhou, Shuhei Ikemoto, Heni Ben Amor

**Abstract:** Differentiable Filters are recursive Bayesian estimators that derive the state transition and measurement models from data alone. Their data-driven nature eschews the need for explicit analytical models, while remaining algorithmic components of the filtering process intact. As a result, the gain mechanism – a critical component of the filtering process – remains non-differentiable and cannot be adjusted to the specific nature of the task or context. In this paper, we propose an attention-based Multimodal Differentiable Filter ($\alpha$-MDF) which utilizes modern attention mechanisms to learn multimodal latent representations. Unlike previous differentiable filter frameworks, $\alpha$-MDF substitutes the traditional gain, e.g., the Kalman gain, with a neural attention mechanism. The approach generates specialized, context-dependent gains that can effectively combine multiple input modalities and observed variables. We validate $\alpha$-MDF on a diverse set of robot state estimation tasks in real world and simulation. Our results show $\alpha$-MDF achieves significant reductions in state estimation errors, demonstrating nearly 4-fold improvements compared to state-of-the-art sensor fusion strategies for rigid body robots. Additionally, the $\alpha$-MDF consistently outperforms differentiable filter baselines by up to $45%$ in soft robotics tasks. The project is available at alpha-mdf.github.io and the codebase is at github.com/ir-lab/alpha-MDF

**摘要:** 可微过滤器是递归贝叶斯估计器，它仅从数据中得出状态转换和测量模型。它们的数据驱动性质避免了对显式分析模型的需要，同时保持了过滤过程的算法组件不变。因此，收益机制--过滤过程的一个关键组成部分--仍然是不可区分的，不能根据任务或背景的具体性质进行调整。本文提出了一种基于注意力的多通道可区分过滤器($-α$-MDF)，它利用现代注意力机制来学习多通道潜在表征。与以往的可微滤波框架不同，$-α$-MDF用神经注意机制取代了传统的增益，例如卡尔曼增益。该方法产生专门的、上下文相关的增益，可以有效地将多个输入模式和观察到的变量结合在一起。我们在现实世界和仿真中的一组不同的机器人状态估计任务上验证了$\α$-MDF。我们的结果表明，$\α$-MDF显著减少了状态估计误差，与用于刚体机器人的最先进的传感器融合策略相比，改进了近4倍。此外，在软机器人任务中，$\Alpha$-MDF始终比可区分的过滤器基线高出$45%$。该项目的网址为：pha-mdf.gihub.io，代码库网址为：githorb.com/ir-lab/pha-mdf

**[Paper URL](https://proceedings.mlr.press/v229/liu23h.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/liu23h/liu23h.pdf)** 

# Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control
**题目:** 指令遵循的目标表示：半监督语言控制界面

**作者:** Vivek Myers, Andre Wang He, Kuan Fang, Homer Rich Walke, Philippe Hansen-Estruch, Ching-An Cheng, Mihai Jalobeanu, Andrey Kolobov, Anca Dragan, Sergey Levine

**Abstract:** Our goal is for robots to follow natural language instructions like “put the towel next to the microwave.” But getting large amounts of labeled data, i.e. data that contains demonstrations of tasks labeled with the language instruction, is prohibitive. In contrast, obtaining policies that respond to image goals is much easier, because any autonomous trial or demonstration can be labeled in hindsight with its final state as the goal. In this work, we contribute a method that taps into joint image- and goal- conditioned policies with language using only a small amount of language data. Prior work has made progress on this using vision-language models or by jointly training language-goal-conditioned policies, but so far neither method has scaled effectively to real-world robot tasks without significant human annotation. Our method achieves robust performance in the real world by learning an embedding from the labeled data that aligns language not to the goal image, but rather to the desired change between the start and goal images that the instruction corresponds to. We then train a policy on this embedding: the policy benefits from all the unlabeled data, but the aligned embedding provides an *interface* for language to steer the policy. We show instruction following across a variety of manipulation tasks in different scenes, with generalization to language instructions outside of the labeled data.

**摘要:** 我们的目标是让机器人遵循自然语言的指示，比如“把毛巾放在微波炉旁边。”但是，获取大量的标记数据，即包含用语言指令标记的任务的演示的数据，是令人望而却步的。相比之下，获得响应图像目标的策略要容易得多，因为任何自主试验或演示都可以事后贴上标签，以其最终状态为目标。在这项工作中，我们贡献了一种方法，该方法利用仅使用少量语言数据的语言来利用联合图像和目标条件策略。以前的工作已经使用视觉语言模型或通过联合训练语言目标制约的策略在这方面取得了进展，但到目前为止，这两种方法都没有在没有重要的人类注释的情况下有效地扩展到现实世界的机器人任务。我们的方法通过从标记数据学习嵌入，从而在真实世界中实现稳健的性能，该嵌入不将语言与目标图像对齐，而是将语言与指令对应的开始图像和目标图像之间的期望变化对齐。然后我们训练一个关于这种嵌入的策略：该策略受益于所有未标记的数据，但对齐的嵌入为语言提供了一个*接口*来指导策略。我们展示了不同场景中各种操作任务的指令跟随，并将其概括为标签数据之外的语言指令。

**[Paper URL](https://proceedings.mlr.press/v229/myers23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/myers23a/myers23a.pdf)** 

# Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions
**题目:** Q-Transformer：通过自回归Q-函数的可扩展离线强化学习

**作者:** Yevgen Chebotar, Quan Vuong, Karol Hausman, Fei Xia, Yao Lu, Alex Irpan, Aviral Kumar, Tianhe Yu, Alexander Herzog, Karl Pertsch, Keerthana Gopalakrishnan, Julian Ibarz, Ofir Nachum, Sumedh Anand Sontakke, Grecia Salazar, Huong T. Tran, Jodilyn Peralta, Clayton Tan, Deeksha Manjunath, Jaspiar Singh, Brianna Zitkovich, Tomas Jackson, Kanishka Rao, Chelsea Finn, Sergey Levine

**Abstract:** In this work, we present a scalable reinforcement learning method for training multi-task policies from large offline datasets that can leverage both human demonstrations and autonomously collected data. Our method uses a Transformer to provide a scalable representation for Q-functions trained via offline temporal difference backups. We therefore refer to the method as Q-Transformer. By discretizing each action dimension and representing the Q-value of each action dimension as separate tokens, we can apply effective high-capacity sequence modeling techniques for Q-learning. We present several design decisions that enable good performance with offline RL training, and show that Q-Transformer outperforms prior offline RL algorithms and imitation learning techniques on a large diverse real-world robotic manipulation task suite.

**摘要:** 在这项工作中，我们提出了一种可扩展的强化学习方法，用于从大型离线数据集训练多任务策略，该方法可以利用人类演示和自主收集的数据。我们的方法使用Transformer为通过离线时间差异备份训练的Q函数提供可扩展的表示。因此，我们将该方法称为Q-Transformer。通过离散化每个动作维度并将每个动作维度的Q值表示为单独的令牌，我们可以将有效的高容量序列建模技术应用于Q学习。我们提出了几项设计决策，通过离线RL训练实现良好的性能，并表明Q-Transformer在大型多样化的现实世界机器人操纵任务套件上优于之前的离线RL算法和模仿学习技术。

**[Paper URL](https://proceedings.mlr.press/v229/chebotar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/chebotar23a/chebotar23a.pdf)** 

# Preference learning for guiding the tree search in continuous POMDPs
**题目:** 用于指导连续POMDPs中树搜索的偏好学习

**作者:** Jiyong Ahn, Sanghyeon Son, Dongryung Lee, Jisu Han, Dongwon Son, Beomjoon Kim

**Abstract:** A robot operating in a partially observable environment must perform sensing actions to achieve a goal, such as clearing the objects in front of a shelf to better localize a target object at the back, and estimate its shape for grasping. A POMDP is a principled framework for enabling robots to perform such information-gathering actions. Unfortunately, while robot manipulation domains involve high-dimensional and continuous observation and action spaces, most POMDP solvers are limited to discrete spaces. Recently, POMCPOW has been proposed for continuous POMDPs, which handles continuity using sampling and progressive widening. However, for robot manipulation problems involving camera observations and multiple objects, POMCPOW is too slow to be practical. We take inspiration from the recent work in learning to guide task and motion planning to propose a framework that learns to guide POMCPOW from past planning experience. Our method uses preference learning that utilizes both success and failure trajectories, where the preference label is given by the results of the tree search. We demonstrate the efficacy of our framework in several continuous partially observable robotics domains, including real-world manipulation, where our framework explicitly reasons about the uncertainty in off-the-shelf segmentation and pose estimation algorithms.

**摘要:** 在部分可观察的环境中操作的机器人必须执行传感动作来实现目标，例如清除货架前面的对象以更好地定位后面的目标对象，并估计其形状以供抓取。POMDP是一个原则框架，使机器人能够执行此类信息收集行动。不幸的是，虽然机器人操作领域涉及高维和连续的观察和动作空间，但大多数POMDP解算器仅限于离散空间。最近，POMCPOW被提出用于连续的POMDP，它通过采样和渐进加宽来处理连续性。然而，对于涉及摄像机观察和多目标的机器人操作问题，POMCPOW速度太慢而不实用。我们从最近在学习指导任务和运动规划方面的工作中得到启发，提出了一个从过去的规划经验中学习指导POMCPOW的框架。我们的方法使用偏好学习，利用成功和失败轨迹，其中偏好标签由树搜索的结果给出。我们证明了我们的框架在几个连续的部分可观测的机器人领域的有效性，包括真实世界的操作，我们的框架明确地解释了现成的分割和姿势估计算法中的不确定性。

**[Paper URL](https://proceedings.mlr.press/v229/ahn23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/ahn23a/ahn23a.pdf)** 

# Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation
**题目:** Act 3D：用于多任务机器人操纵的3D特征场变形机

**作者:** Theophile Gervet, Zhou Xian, Nikolaos Gkanatsios, Katerina Fragkiadaki

**Abstract:** 3D perceptual representations are well suited for robot manipulation as they easily encode occlusions and simplify spatial reasoning. Many manipulation tasks require high spatial precision in end-effector pose prediction, which typically demands high-resolution 3D feature grids that are computationally expensive to process. As a result, most manipulation policies operate directly in 2D, foregoing 3D inductive biases. In this paper, we introduce Act3D, a manipulation policy transformer that represents the robot’s workspace using a 3D feature field with adaptive resolutions dependent on the task at hand. The model lifts 2D pre-trained features to 3D using sensed depth, and attends to them to compute features for sampled 3D points. It samples 3D point grids in a coarse to fine manner, featurizes them using relative-position attention, and selects where to focus the next round of point sampling. In this way, it efficiently computes 3D action maps of high spatial resolution. Act3D sets a new state-of-the-art in RLBench, an established manipulation benchmark, where it achieves $10%$ absolute improvement over the previous SOTA 2D multi-view policy on 74 RLBench tasks and $22%$ absolute improvement with 3x less compute over the previous SOTA 3D policy. We quantify the importance of relative spatial attention, large-scale vision-language pre-trained 2D backbones, and weight tying across coarse-to-fine attentions in ablative experiments.

**摘要:** 3D感知表示非常适合机器人操作，因为它们很容易对遮挡进行编码，并简化空间推理。许多操纵任务对末端执行器姿态预测的空间精度要求很高，这通常需要高分辨率的3D特征网格，而这些网格的计算代价很高。因此，大多数操作策略直接在2D中操作，而不是3D诱导偏差。在本文中，我们介绍了Act3D，一个操作策略转换器，它使用一个3D特征域来表示机器人的工作空间，该特征域具有根据手头任务的自适应分辨率。该模型利用感知深度将预先训练好的2D特征提升到3D，并关注它们来计算采样的3D点的特征。它以从粗到精的方式对三维点网格进行采样，使用相对位置注意力对其进行特征化，并选择下一轮点采样的焦点位置。通过这种方式，它可以高效地计算出高空间分辨率的3D动作图。Act3D在RLBuch中设定了新的最先进水平，这是一项既定的操作基准，在74个RBch任务上，它实现了$10%$与先前的SOTA 2D多视图策略相比的绝对改进，以及$22%$的绝对改进，计算量比先前的Sota 3D策略减少了3倍。在消融实验中，我们量化了相对空间注意力、大规模视觉语言预先训练的2D主干和从粗到精的注意的权重挂钩的重要性。

**[Paper URL](https://proceedings.mlr.press/v229/gervet23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/gervet23a/gervet23a.pdf)** 

# Simultaneous Learning of Contact and Continuous Dynamics
**题目:** 同时学习接触和持续动态

**作者:** Bibit Bianchini, Mathew Halm, Michael Posa

**Abstract:** Robotic manipulation can greatly benefit from the data efficiency, robustness, and predictability of model-based methods if robots can quickly generate models of novel objects they encounter. This is especially difficult when effects like complex joint friction lack clear first-principles models and are usually ignored by physics simulators. Further, numerically-stiff contact dynamics can make common model-building approaches struggle. We propose a method to simultaneously learn contact and continuous dynamics of a novel, possibly multi-link object by observing its motion through contact-rich trajectories. We formulate a system identification process with a loss that infers unmeasured contact forces, penalizing their violation of physical constraints and laws of motion given current model parameters. Our loss is unlike prediction-based losses used in differentiable simulation. Using a new dataset of real articulated object trajectories and an existing cube toss dataset, our method outperforms differentiable simulation and end-to-end alternatives with more data efficiency. See our project page for code, datasets, and media: https://sites.google.com/view/continuous-contact-nets/home

**摘要:** 如果机器人能够快速生成它们遇到的新对象的模型，那么机器人操作可以从基于模型的方法的数据效率、健壮性和可预测性中受益匪浅。当复杂的关节摩擦等效应缺乏明确的第一原理模型，并且通常被物理模拟器忽略时，这一点尤其困难。此外，数值僵硬的接触动力学可能会使常见的建模方法变得困难。我们提出了一种方法，通过观察接触丰富的轨迹来观察一个新的、可能是多个链接的对象的运动，从而同时学习接触和连续动力学。我们制定了一个系统辨识过程，用损失来推断不可测量的接触力，惩罚他们在给定当前模型参数的情况下违反物理约束和运动规律。我们的损失不同于可微模拟中使用的基于预测的损失。使用一个新的真实关节物体轨迹数据集和一个现有的立方体Toss数据集，我们的方法比可区分模拟和端到端方案具有更高的数据效率。有关代码、数据集和媒体，请参阅我们的项目页面：https://sites.google.com/view/continuous-contact-nets/home

**[Paper URL](https://proceedings.mlr.press/v229/bianchini23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v229/bianchini23a/bianchini23a.pdf)** 

