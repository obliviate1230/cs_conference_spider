# RoboTube: Learning Household Manipulation from Human Videos with Simulated Twin Environments
**题目:** RoboTube：通过模拟双胞胎环境从人类视频中学习家庭操纵

**作者:** Haoyu Xiong, Haoyuan Fu, Jieyi Zhang, Chen Bao, Qiang Zhang, Yongxi Huang, Wenqiang Xu, Animesh Garg, Cewu Lu

**Abstract:** We aim to build a useful, reproducible, democratized benchmark for learning household robotic manipulation from human videos. To realize this goal, a diverse, high-quality human video dataset curated specifically for robots is desired. To evaluate the learning progress, a simulated twin environment that resembles the appearance and the dynamics of the physical world would help roboticists and AI researchers validate their algorithms convincingly and efficiently before testing on a real robot. Hence, we present RoboTube, a human video dataset, and its digital twins for learning various robotic manipulation tasks. RoboTube video dataset contains 5,000 video demonstrations recorded with multi-view RGB-D cameras of human-performing everyday household tasks including manipulation of rigid objects, articulated objects, deformable objects, and bimanual manipulation. RT-sim, as the simulated twin environments, consists of 3D scanned, photo-realistic objects, minimizing the visual domain gap between the physical world and the simulated environment. After extensively benchmarking existing methods in the field of robot learning from videos, the empirical results suggest that knowledge and models learned from the RoboTube video dataset can be deployed, benchmarked, and reproduced in RT-sim and be transferred to a real robot. We hope RoboTube can lower the barrier to robotics research for beginners while facilitating reproducible research in the community. More experiments and videos can be found in the supplementary materials and on the website: https://sites.google.com/view/robotube

**摘要:** 我们的目标是建立一个有用的、可重复的、民主化的基准，用于从人类视频中学习家用机器人操作。为了实现这一目标，需要一个专门为机器人策划的多样化、高质量的人类视频数据集。为了评估学习进展，一个与物理世界的外观和动态相似的模拟孪生环境将帮助机器人学家和人工智能研究人员在真实机器人上测试之前，令人信服地、高效地验证他们的算法。因此，我们介绍了RoboTube，一个人类视频数据集，以及它的数字双胞胎，用于学习各种机器人操作任务。RoboTube视频数据集包含用多视角RGB-D摄像机记录的5,000个视频演示-人类执行日常家居任务，包括操纵刚性物体、铰接物、变形物和双手操纵。RT-SIM作为模拟的孪生环境，由3D扫描的照片级真实感物体组成，最小化了物理世界和模拟环境之间的视域差距。在对机器人从视频中学习的现有方法进行了广泛的基准测试后，实验结果表明，从RoboTube视频数据集中学习的知识和模型可以在RT-SIM中部署、基准测试和复制，并可以传输到真实的机器人上。我们希望RoboTube能够降低初学者进行机器人研究的门槛，同时促进社区中的可重复研究。更多实验和视频可以在补充材料中找到，也可以在以下网站上找到：https://sites.google.com/view/robotube

**[Paper URL](https://proceedings.mlr.press/v205/xiong23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/xiong23a/xiong23a.pdf)** 

# Training Robots to Evaluate Robots: Example-Based Interactive Reward Functions for Policy Learning
**题目:** 训练机器人以评估机器人：基于示例的交互式奖励功能用于政策学习

**作者:** Kun Huang, Edward S. Hu, Dinesh Jayaraman

**Abstract:** Physical interactions can often help reveal information that is not readily apparent. For example, we may tug at a table leg to evaluate whether it is built well, or turn a water bottle upside down to check that it is watertight. We propose to train robots to acquire such interactive behaviors automatically, for the purpose of evaluating the result of an attempted robotic skill execution. These evaluations in turn serve as "interactive reward functions" (IRFs) for training reinforcement learning policies to perform the target skill, such as screwing the table leg tightly. In addition, even after task policies are fully trained, IRFs can serve as verification mechanisms that improve online task execution. For any given task, our IRFs can be conveniently trained using only examples of successful outcomes, and no further specification is needed to train the task policy thereafter. In our evaluations on door locking and weighted block stacking in simulation, and screw tightening on a real robot, IRFs enable large performance improvements, even outperforming baselines with access to demonstrations or carefully engineered rewards.

**摘要:** 物理相互作用通常有助于揭示不太明显的信息。例如，我们可以拉一条桌腿来评估它是否做得很好，或者把一个水瓶倒过来检查它是不是防水的。我们建议训练机器人自动获取这种交互行为，目的是评估尝试的机器人技能执行的结果。这些评估反过来充当“互动奖励功能”(IRF)，用于训练强化学习策略以执行目标技能，例如紧紧拧紧桌腿。此外，即使在任务策略得到充分培训之后，IRF也可以作为改进在线任务执行的验证机制。对于任何给定的任务，只需使用成功结果的示例就可以方便地训练我们的IRF，此后不需要进一步指定来训练任务策略。在我们对模拟中的门锁和加权积木堆叠以及在真实机器人上拧紧螺丝进行的评估中，红外线能够大幅提高性能，甚至通过访问演示或精心设计的奖励来超越基线。

**[Paper URL](https://proceedings.mlr.press/v205/huang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/huang23a/huang23a.pdf)** 

# Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior
**题目:** 采用这些方式：调整机器人控制以实现行为多样性的概括

**作者:** Gabriel B. Margolis, Pulkit Agrawal

**Abstract:** Learned locomotion policies can rapidly adapt to diverse environments similar to those experienced during training but lack a mechanism for fast tuning when they fail in an out-of-distribution test environment. This necessitates a slow and iterative cycle of reward and environment redesign to achieve good performance on a new task. As an alternative, we propose learning a single policy that encodes a structured family of locomotion strategies that solve training tasks in different ways, resulting in Multiplicity of Behavior (MoB). Different strategies generalize differently and can be chosen in real-time for new tasks or environments, bypassing the need for time-consuming retraining. We release a fast, robust open-source MoB locomotion controller, Walk These Ways, that can execute diverse gaits with variable footswing, posture, and speed, unlocking diverse downstream tasks: crouching, hopping, high-speed running, stair traversal, bracing against shoves, rhythmic dance, and more. Video and code release: https://gmargo11.github.io/walk-these-ways

**摘要:** 学习的移动策略可以快速适应不同的环境，类似于在培训期间经历的环境，但在分布外的测试环境中失败时，缺乏快速调整的机制。这就需要一个缓慢而迭代的奖励和环境重新设计循环，以在新任务中取得良好的表现。作为替代方案，我们建议学习一种单一的策略，该策略编码一系列结构化的运动策略，这些策略以不同的方式解决训练任务，从而导致行为的多样性(MOB)。不同的策略具有不同的一般性，可以针对新的任务或环境实时选择，从而绕过了耗时的再培训需要。我们发布了一个快速，强大的开源暴徒运动控制器，Walk Three Ways，它可以通过可变的脚步摆动、姿势和速度执行不同的步态，释放不同的下游任务：蹲伏、跳跃、高速跑步、楼梯穿越、靠推支撑、有节奏的舞蹈等。视频和代码发布：https://gmargo11.github.io/walk-these-ways

**[Paper URL](https://proceedings.mlr.press/v205/margolis23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf)** 

# Watch and Match: Supercharging Imitation with Regularized Optimal Transport
**题目:** 观看并匹配：通过常规化的最佳运输来增强模仿

**作者:** Siddhant Haldar, Vaibhav Mathur, Denis Yarats, Lerrel Pinto

**Abstract:** Imitation learning holds tremendous promise in learning policies efficiently for complex decision making problems. Current state-of-the-art algorithms often use inverse reinforcement learning (IRL), where given a set of expert demonstrations, an agent alternatively infers a reward function and the associated optimal policy. However, such IRL approaches often require substantial online interactions for complex control problems. In this work, we present Regularized Optimal Transport (ROT), a new imitation learning algorithm that builds on recent advances in optimal transport based trajectory-matching. Our key technical insight is that adaptively combining trajectory-matching rewards with behavior cloning can significantly accelerate imitation even with only a few demonstrations. Our experiments on 20 visual control tasks across the DeepMind Control Suite, the OpenAI Robotics Suite, and the Meta-World Benchmark demonstrate an average of 7.8x faster imitation to reach 90% of expert performance compared to prior state-of-the-art methods. On real-world robotic manipulation, with just one demonstration and an hour of online training, ROT achieves an average success rate of 90.1% across 14 tasks.

**摘要:** 模仿学习在有效地学习复杂决策问题的策略方面有着巨大的前景。目前最先进的算法通常使用逆强化学习(IRL)，在给定一组专家演示的情况下，代理交替推断奖励函数和相关的最优策略。然而，对于复杂的控制问题，这种IRL方法通常需要大量的在线交互。在这项工作中，我们提出了正则化最优传输(ROT)，这是一种新的模拟学习算法，它建立在基于最优传输的轨迹匹配的最新进展的基础上。我们的关键技术见解是，将轨迹匹配奖励与行为克隆自适应地结合起来，即使只有几个演示，也可以显着加快模仿速度。我们在DeepMind Control Suite、OpenAI Robotics Suite和Meta-World基准上对20个视觉控制任务进行的实验表明，与之前最先进的方法相比，模拟速度平均快7.8倍，达到90%的专家性能。在现实世界的机器人操作上，只需一次演示和一小时的在线培训，ROT在14项任务中的平均成功率为90.1%。

**[Paper URL](https://proceedings.mlr.press/v205/haldar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/haldar23a/haldar23a.pdf)** 

# Offline Reinforcement Learning for Visual Navigation
**题目:** 视觉导航的离线强化学习

**作者:** Dhruv Shah, Arjun Bhorkar, Hrishit Leen, Ilya Kostrikov, Nicholas Rhinehart, Sergey Levine

**Abstract:** Reinforcement learning can enable robots to navigate to distant goals while optimizing user-specified reward functions, including preferences for following lanes, staying on paved paths, or avoiding freshly mowed grass. However, online learning from trial-and-error for real-world robots is logistically challenging, and methods that instead can utilize existing datasets of robotic navigation data could be significantly more scalable and enable broader generalization. In this paper, we present ReViND, the first offline RL system for robotic navigation that can leverage previously collected data to optimize user-specified reward functions in the real-world. We evaluate our system for off-road navigation without any additional data collection or fine-tuning, and show that it can navigate to distant goals using only offline training from this dataset, and exhibit behaviors that qualitatively differ based on the user-specified reward function.

**摘要:** 强化学习可以使机器人能够导航到遥远的目标，同时优化用户指定的奖励功能，包括沿着车道行驶、停留在铺砌的道路上或避开新修剪的草的偏好。然而，现实世界的机器人通过试错进行在线学习在后勤上具有挑战性，而可以利用机器人导航数据的现有数据集的方法可能会更具可扩展性，并能够实现更广泛的概括。在本文中，我们介绍了ReViND，这是第一个用于机器人导航的离线RL系统，可以利用之前收集的数据来优化现实世界中用户指定的奖励功能。我们在没有任何额外数据收集或微调的情况下评估了我们的越野导航系统，并表明它可以仅使用来自该数据集的离线训练导航到遥远的目标，并表现出基于用户指定的奖励函数的定性不同的行为。

**[Paper URL](https://proceedings.mlr.press/v205/shah23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/shah23a/shah23a.pdf)** 

# Graph Inverse Reinforcement Learning from Diverse Videos
**题目:** 来自不同视频的图逆强化学习

**作者:** Sateesh Kumar, Jonathan Zamora, Nicklas Hansen, Rishabh Jangir, Xiaolong Wang

**Abstract:** Research on Inverse Reinforcement Learning (IRL) from third-person videos has shown encouraging results on removing the need for manual reward design for robotic tasks. However, most prior works are still limited by training from a relatively restricted domain of videos. In this paper, we argue that the true potential of third-person IRL lies in increasing the diversity of videos for better scaling. To learn a reward function from diverse videos, we propose to perform graph abstraction on the videos followed by temporal matching in the graph space to measure the task progress. Our insight is that a task can be described by entity interactions that form a graph, and this graph abstraction can help remove irrelevant information such as textures, resulting in more robust reward functions. We evaluate our approach, GraphIRL, on cross-embodiment learning in X-MAGICAL and learning from human demonstrations for real-robot manipulation. We show significant improvements in robustness to diverse video demonstrations over previous approaches, and even achieve better results than manual reward design on a real robot pushing task. Videos are available at https://sateeshkumar21.github.io/GraphIRL/.

**摘要:** 对第三人称视频的反向强化学习(IRL)的研究表明，在消除机器人任务的人工奖励设计方面取得了令人鼓舞的结果。然而，大多数先前的工作仍然受到来自相对受限的视频领域的训练的限制。在本文中，我们认为第三人称IRL的真正潜力在于增加视频的多样性，以实现更好的伸缩性。为了从不同的视频中学习奖励函数，我们提出了对视频进行图抽象，然后在图空间中进行时间匹配来衡量任务进度。我们的观点是，任务可以用形成图的实体交互来描述，这种图抽象可以帮助删除不相关的信息，如纹理，从而产生更健壮的奖励函数。我们评估了我们的方法，GraphIRL，在X魔术中的跨具身学习和从人类演示中学习真实机器人操作。与以前的方法相比，我们在对不同视频演示的稳健性方面有了显著的提高，甚至在真实的机器人推送任务上取得了比人工奖励设计更好的结果。有关视频，请访问https://sateeshkumar21.github.io/GraphIRL/.

**[Paper URL](https://proceedings.mlr.press/v205/kumar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/kumar23a/kumar23a.pdf)** 

# Inferring Smooth Control: Monte Carlo Posterior Policy Iteration with Gaussian Processes
**题目:** 推断平滑控制：采用高斯过程的蒙特卡洛后验策略迭代

**作者:** Joe Watson, Jan Peters

**Abstract:** Monte Carlo methods have become increasingly relevant for control of non-differentiable systems, approximate dynamics models, and learning from data.These methods scale to high-dimensional spaces and are effective at the non-convex optimization often seen in robot learning. We look at sample-based methods from the perspective of inference-based control, specifically posterior policy iteration. From this perspective, we highlight how Gaussian noise priors produce rough control actions that are unsuitable for physical robot deployment. Considering smoother Gaussian process priors, as used in episodic reinforcement learning and motion planning, we demonstrate how smoother model predictive control can be achieved using online sequential inference. This inference is realized through an efficient factorization of the action distribution, and novel means of optimizing the likelihood temperature for to improve importance sampling accuracy. We evaluate this approach on several high-dimensional robot control tasks, matching the sample efficiency of prior heuristic methods while also ensuring smoothness. Simulation results can be seen at monte-carlo-ppi.github.io.

**摘要:** 蒙特卡罗方法在控制不可微系统、近似动力学模型和从数据中学习方面变得越来越重要，这些方法可以扩展到高维空间，并且对于机器人学习中常见的非凸优化是有效的。我们从基于推理的控制的角度来看待基于样本的方法，特别是后验策略迭代。从这个角度，我们强调了高斯噪声先验是如何产生不适合物理机器人部署的粗略控制动作的。考虑到更平滑的高斯过程先验，如用于情节强化学习和运动规划，我们演示了如何使用在线序贯推理来实现更平滑的模型预测控制。这一推断是通过对动作分布进行有效的因式分解，以及优化似然温度以提高重要采样精度的新方法来实现的。我们在几个高维机器人控制任务上对该方法进行了评估，在保证光滑性的同时与先前启发式方法的样本效率相匹配。模拟结果可以在monte-carlo-ppi.githeb.io上看到。

**[Paper URL](https://proceedings.mlr.press/v205/watson23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/watson23a/watson23a.pdf)** 

# BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation
**题目:** 行为-1 K：具有1，000项日常活动和现实模拟的人工智能基准

**作者:** Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, Karen Liu, Jiajun Wu, Li Fei-Fei

**Abstract:** We present BEHAVIOR-1K, a comprehensive simulation benchmark for human-centered robotics. BEHAVIOR-1K includes two components, guided and motivated by the results of an extensive survey on "what do you want robots to do for you?". The first is the definition of 1,000 everyday activities, grounded in 50 scenes (houses, gardens, restaurants, offices, etc.) with more than 5,000 objects annotated with rich physical and semantic properties. The second is OmniGibson, a novel simulation environment that supports these activities via realistic physics simulation and rendering of rigid bodies, deformable bodies, and liquids. Our experiments indicate that the activities in BEHAVIOR-1K are long-horizon and dependent on complex manipulation skills, both of which remain a challenge for even state-of-the-art robot learning solutions. To calibrate the simulation-to-reality gap of BEHAVIOR-1K, we provide an initial study on transferring solutions learned with a mobile manipulator in a simulated apartment to its real-world counterpart. We hope that BEHAVIOR-1K’s human-grounded nature, diversity, and realism make it valuable for embodied AI and robot learning research. Project website: https://behavior.stanford.edu.

**摘要:** 我们提出了Behavior-1K，这是一个以人为中心的机器人综合模拟基准。Behavior-1K包括两个组成部分，这两个组成部分是由一项关于“你希望机器人为你做什么？”的广泛调查的结果指导和激励的。第一个是对1000项日常活动的定义，基于50个场景(房屋、花园、餐馆、办公室等)有5,000多个对象，用丰富的物理和语义属性进行注释。第二个是OmniGibson，这是一个新的模拟环境，通过对刚体、变形体和液体进行逼真的物理模拟和渲染来支持这些活动。我们的实验表明，Behavior-1K中的活动是长期的，依赖于复杂的操作技能，这两个方面即使是最先进的机器人学习解决方案也是一个挑战。为了校准行为1K的模拟与现实的差距，我们提供了将模拟公寓中的移动机械手学习的解决方案转移到现实世界中的解决方案的初步研究。我们希望Behavior-1K以人为本的本性、多样性和现实主义使其对嵌入式人工智能和机器人学习研究具有价值。项目网站：https://behavior.stanford.edu.

**[Paper URL](https://proceedings.mlr.press/v205/li23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/li23a/li23a.pdf)** 

# Temporal Logic Imitation: Learning Plan-Satisficing Motion Policies from Demonstrations
**题目:** 时间逻辑模仿：学习计划--从示威中满足运动政策

**作者:** Yanwei Wang, Nadia Figueroa, Shen Li, Ankit Shah, Julie Shah

**Abstract:** Learning from demonstration (LfD) has successfully solved tasks featuring a long time horizon. However, when the problem complexity also includes human-in-the-loop perturbations, state-of-the-art approaches do not guarantee the successful reproduction of a task. In this work, we identify the roots of this challenge as the failure of a learned continuous policy to satisfy the discrete plan implicit in the demonstration. By utilizing modes (rather than subgoals) as the discrete abstraction and motion policies with both mode invariance and goal reachability properties, we prove our learned continuous policy can simulate any discrete plan specified by a linear temporal logic (LTL) formula. Consequently, an imitator is robust to both task- and motion-level perturbations and guaranteed to achieve task success.

**摘要:** 从演示中学习（LfD）已成功解决了具有较长时间范围的任务。然而，当问题的复杂性还包括人在回路中的扰动时，最先进的方法并不能保证任务的成功复制。在这项工作中，我们将这一挑战的根源确定为习得的连续政策未能满足演示中隐含的离散计划。通过利用模式（而不是子目标）作为具有模式不变性和目标可达性的离散抽象和运动策略，我们证明我们学习的连续策略可以模拟线性时态逻辑（LTL）公式指定的任何离散计划。因此，模仿者对任务和运动级扰动都具有鲁棒性，并保证实现任务成功。

**[Paper URL](https://proceedings.mlr.press/v205/wang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wang23a/wang23a.pdf)** 

# Generalization with Lossy Affordances: Leveraging Broad Offline Data for Learning Visuomotor Tasks
**题目:** 利用有损的功能进行概括：利用广泛的离线数据来学习可视化任务

**作者:** Kuan Fang, Patrick Yin, Ashvin Nair, Homer Rich Walke, Gengchen Yan, Sergey Levine

**Abstract:** The use of broad datasets has proven to be crucial for generalization for a wide range of fields. However, how to effectively make use of diverse multi-task data for novel downstream tasks still remains a grand challenge in reinforcement learning and robotics. To tackle this challenge, we introduce a framework that acquires goal-conditioned policies for unseen temporally extended tasks via offline reinforcement learning on broad data, in combination with online fine-tuning guided by subgoals in a learned lossy representation space. When faced with a novel task goal, our framework uses an affordance model to plan a sequence of lossy representations as subgoals that decomposes the original task into easier problems. Learned from the broad prior data, the lossy representation emphasizes task-relevant information about states and goals while abstracting away redundant contexts that hinder generalization. It thus enables subgoal planning for unseen tasks, provides a compact input to the policy, and facilitates reward shaping during fine-tuning. We show that our framework can be pre-trained on large-scale datasets of robot experience from prior work and efficiently fine-tuned for novel tasks, entirely from visual inputs without any manual reward engineering.

**摘要:** 广泛数据集的使用已被证明对广泛领域的推广至关重要。然而，如何有效地利用不同的多任务数据来执行新的下游任务，仍然是强化学习和机器人学中的一个巨大挑战。为了应对这一挑战，我们引入了一个框架，通过对广泛数据的离线强化学习，结合学习有损表示空间中的子目标引导的在线微调，为看不见的时间扩展任务获取目标条件策略。当面对一个新的任务目标时，我们的框架使用一个负担模型来计划一系列有损表示作为子目标，将原始任务分解成更容易的问题。从广泛的先前数据中学习，有损表示强调有关状态和目标的任务相关信息，同时抽象出阻碍泛化的冗余上下文。因此，它可以为看不见的任务制定子目标计划，为政策提供紧凑的输入，并在微调期间促进奖励形成。我们表明，我们的框架可以在来自先前工作的大规模机器人经验数据集上进行预训练，并针对新任务进行有效的微调，完全来自视觉输入，而不需要任何人工奖励工程。

**[Paper URL](https://proceedings.mlr.press/v205/fang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/fang23a/fang23a.pdf)** 

# Real-time Mapping of Physical Scene Properties with an Autonomous Robot Experimenter
**题目:** 使用自主机器人实验员实时绘制物理场景属性

**作者:** Iain Haughton, Edgar Sucar, Andre Mouton, Edward Johns, Andrew Davison

**Abstract:** Neural fields can be trained from scratch to represent the shape and appearance of 3D scenes efficiently. It has also been shown that they can densely map correlated properties such as semantics, via sparse interactions from a human labeller. In this work, we show that a robot can densely annotate a scene with arbitrary discrete or continuous physical properties via its own fully-autonomous experimental interactions, as it simultaneously scans and maps it with an RGB-D camera. A variety of scene interactions are possible, including poking with force sensing to determine rigidity, measuring local material type with single-pixel spectroscopy or predicting force distributions by pushing. Sparse experimental interactions are guided by entropy to enable high efficiency, with tabletop scene properties densely mapped from scratch in a few minutes from a few tens of interactions.

**摘要:** 可以从头开始训练神经场，以有效地表示3D场景的形状和外观。研究还表明，它们可以通过人类标签器的稀疏交互来密集地映射相关属性，例如语义。在这项工作中，我们表明，机器人可以通过其自己的完全自主的实验交互来密集地注释具有任意离散或连续物理属性的场景，因为它同时用RGB-D相机扫描和绘制该场景。多种场景交互都是可能的，包括用力传感戳以确定硬度、用单像素光谱测量局部材料类型或通过推动预测力分布。稀疏的实验交互由信息量指导，以实现高效率，桌面场景属性在几分钟内从零开始密集映射，只需几十个交互。

**[Paper URL](https://proceedings.mlr.press/v205/haughton23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/haughton23a/haughton23a.pdf)** 

# Robust Trajectory Prediction against Adversarial Attacks
**题目:** 针对对抗攻击的稳健轨迹预测

**作者:** Yulong Cao, Danfei Xu, Xinshuo Weng, Zhuoqing Mao, Anima Anandkumar, Chaowei Xiao, Marco Pavone

**Abstract:** Trajectory prediction using deep neural networks (DNNs) is an essential component of autonomous driving (AD) systems.  However, these methods are vulnerable to adversarial attacks, leading to serious consequences such as collisions. In this work, we identify two key ingredients to defend trajectory prediction models against adversarial attacks including (1) designing effective adversarial training methods and (2) adding domain-specific data augmentation to mitigate the performance degradation on clean data. We demonstrate that our method is able to improve the performance by 46% on adversarial data and at the cost of only 3% performance degradation on clean data, compared to the model trained with clean data. Additionally, compared to existing robust methods, our method can improve performance by 21% on adversarial examples and 9% on clean data. Our robust model is evaluated with a planner to study its downstream impacts. We demonstrate that our model can significantly reduce the severe accident rates (e.g., collisions and off-road driving).

**摘要:** 基于深度神经网络的轨迹预测是自动驾驶系统的重要组成部分。然而，这些方法容易受到对抗性攻击，导致碰撞等严重后果。在这项工作中，我们确定了两个关键因素来防御弹道预测模型的对抗性攻击，包括(1)设计有效的对抗性训练方法和(2)添加特定领域的数据增强来缓解在干净数据上的性能下降。我们证明，与使用干净数据训练的模型相比，我们的方法能够在对抗性数据上提高46%的性能，而在干净数据上的代价只有3%的性能下降。此外，与现有的稳健方法相比，我们的方法在对抗性实例上的性能提高了21%，在干净数据上的性能提高了9%。我们的稳健模型由规划者评估，以研究其下游影响。我们证明，我们的模型可以显著降低严重事故率(例如，碰撞和越野驾驶)。

**[Paper URL](https://proceedings.mlr.press/v205/cao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/cao23a/cao23a.pdf)** 

# Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion
**题目:** 深度全身控制：学习统一的操纵和运动策略

**作者:** Zipeng Fu, Xuxin Cheng, Deepak Pathak

**Abstract:** An attached arm can significantly increase the applicability of legged robots to several mobile manipulation tasks that are not possible for the wheeled or tracked counterparts. The standard modular control pipeline for such legged manipulators is to decouple the controller into that of manipulation and locomotion. However, this is ineffective. It requires immense engineering to support coordination between the arm and legs, and error can propagate across modules causing non-smooth unnatural motions. It is also biological implausible given evidence for strong motor synergies across limbs. In this work, we propose to learn a unified policy for whole-body control of a legged manipulator using reinforcement learning. We propose Regularized Online Adaptation to bridge the Sim2Real gap for high-DoF control, and Advantage Mixing exploiting the causal dependency in the action space to overcome local minima during training the whole-body system. We also present a simple design for a low-cost legged manipulator, and find that our unified policy can demonstrate dynamic and agile behaviors across several task setups. Videos are at https://maniploco.github.io

**摘要:** 一个连接的手臂可以显著提高腿式机器人对几个轮式或履带式机器人无法完成的移动操作任务的适用性。这种腿式机械手的标准模块化控制流水线是将控制器解耦为操纵和运动控制器。然而，这是无效的。它需要巨大的工程来支持手臂和腿之间的协调，并且误差可能会在模块之间传播，导致不平稳的非自然运动。这在生物学上也是不可信的，因为有证据表明，四肢之间存在强大的运动协同效应。在这项工作中，我们建议使用强化学习来学习腿部机械手的全身控制的统一策略。我们提出了规则化的在线自适应来弥补Sim2Real在高DOF控制方面的差距，并提出了优势混合利用动作空间中的因果依赖来克服训练全身系统时的局部极小值。我们还提出了一个低成本的腿式机械手的简单设计，并发现我们的统一策略可以在几个任务设置中演示动态和敏捷的行为。视频请访问https://maniploco.github.io

**[Paper URL](https://proceedings.mlr.press/v205/fu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/fu23a/fu23a.pdf)** 

# Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity
**题目:** 学会用紧急的外在灵活性抓住难以理解的东西

**作者:** Wenxuan Zhou, David Held

**Abstract:** A simple gripper can solve more complex manipulation tasks if it can utilize the external environment such as pushing the object against the table or a vertical wall, known as "Extrinsic Dexterity." Previous work in extrinsic dexterity usually has careful assumptions about contacts which impose restrictions on robot design, robot motions, and the variations of the physical parameters. In this work, we develop a system based on reinforcement learning (RL) to address these limitations. We study the task of "Occluded Grasping" which aims to grasp the object in configurations that are initially occluded; the robot needs to move the object into a configuration from which these grasps can be achieved. We present a system with model-free RL that successfully achieves this task using a simple gripper with extrinsic dexterity. The policy learns emergent behaviors of pushing the object against the wall to rotate and then grasp it without additional reward terms on extrinsic dexterity. We discuss important components of the system including the design of the RL problem, multi-grasp training and selection, and policy generalization with automatic curriculum. Most importantly, the policy trained in simulation is zero-shot transferred to a physical robot. It demonstrates dynamic and contact-rich motions with a simple gripper that generalizes across objects with various size, density, surface friction, and shape with a 78% success rate.

**摘要:** 如果一个简单的抓手能够利用外部环境，比如将物体推到桌子上或垂直的墙上，那么它就可以解决更复杂的操作任务，这就是我们所说的“外部灵巧性”。以往关于外部灵巧性的工作通常对接触进行了仔细的假设，这些接触对机器人设计、机器人运动以及物理参数的变化施加了限制。在这项工作中，我们开发了一个基于强化学习(RL)的系统来解决这些限制。我们研究了“遮挡抓取”的任务，它的目标是在最初被遮挡的构形中抓取对象；机器人需要将对象移动到可以实现这些抓取的构形中。我们提出了一个具有无模型RL的系统，它使用一个具有外部灵活性的简单夹爪成功地完成了这一任务。该政策学习了将物体靠墙推以旋转，然后抓住它的紧急行为，而不需要额外的外部灵巧性奖励条款。我们讨论了系统的重要组成部分，包括RL问题的设计，多抓取训练和选择，以及带有自动课程的策略泛化。最重要的是，在模拟中训练的策略是零命中转移到物理机器人上。它用一个简单的抓手演示动态和丰富的接触运动，可以概括不同大小、密度、表面摩擦和形状的对象，成功率为78%。

**[Paper URL](https://proceedings.mlr.press/v205/zhou23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zhou23a/zhou23a.pdf)** 

# PRISM: Probabilistic Real-Time Inference in Spatial World Models
**题目:** 棱镜：空间世界模型中的概率实时推理

**作者:** Atanas Mirchev, Baris Kayalibay, Ahmed Agha, Patrick van der Smagt, Daniel Cremers, Justin Bayer

**Abstract:** We introduce PRISM, a method for real-time filtering in a probabilistic generative model of agent motion and visual perception. Previous approaches either lack uncertainty estimates for the map and agent state, do not run in real-time, do not have a dense scene representation or do not model agent dynamics. Our solution reconciles all of these aspects. We start from a predefined state-space model which combines differentiable rendering and 6-DoF dynamics. Probabilistic inference in this model amounts to simultaneous localisation and mapping (SLAM) and is intractable. We use a series of approximations to Bayesian inference to arrive at probabilistic map and state estimates. We take advantage of well-established methods and closed-form updates, preserving accuracy and enabling real-time capability. The proposed solution runs at 10Hz real-time and is similarly accurate to state-of-the-art SLAM in small to medium-sized indoor environments, with high-speed UAV and handheld camera agents (Blackbird, EuRoC and TUM-RGBD).

**摘要:** 我们介绍了PRISM，这是一种在主体运动和视觉感知的概率生成模型中进行实时过滤的方法。以前的方法要么缺乏对地图和代理状态的不确定性估计，要么不实时运行，要么没有密集的场景表示，要么没有对代理动态进行建模。我们的解决方案协调了所有这些方面。我们从一个预定义的状态空间模型开始，该模型结合了可微渲染和6-DOF动力学。该模型中的概率推理相当于同时定位和映射(SLAM)，并且是难以处理的。我们使用贝叶斯推理的一系列近似来获得概率图和状态估计。我们利用成熟的方法和封闭形式的更新，保持准确性并实现实时能力。建议的解决方案以10赫兹的实时频率运行，与具有高速无人机和手持摄像代理(Blackbird、EURC和TUM-RGBD)的中小型室内环境中的最先进的SLAM类似。

**[Paper URL](https://proceedings.mlr.press/v205/mirchev23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/mirchev23a/mirchev23a.pdf)** 

# Instruction-driven history-aware policies for robotic manipulations
**题目:** 机器人操纵的指令驱动的历史感知策略

**作者:** Pierre-Louis Guhur, Shizhe Chen, Ricardo Garcia Pinel, Makarand Tapaswi, Ivan Laptev, Cordelia Schmid

**Abstract:** In human environments, robots are expected to accomplish a variety of manipulation tasks given simple natural language instructions. Yet, robotic manipulation is extremely challenging as it requires fine-grained motor control, long-term memory as well as generalization to previously unseen tasks and environments. To address these challenges, we propose a unified transformer-based approach that takes into account multiple inputs. In particular, our transformer architecture integrates (i) natural language instructions and (ii) multi-view scene observations while (iii) keeping track of the full history of observations and actions. Such an approach enables learning dependencies between history and instructions and improves manipulation precision using multiple views. We evaluate our method on the challenging RLBench benchmark and on a real-world robot. Notably, our approach scales to 74 diverse RLBench tasks and outperforms the state of the art. We also address instruction-conditioned tasks and demonstrate excellent generalization to previously unseen variations.

**摘要:** 在人类环境中，机器人被期望在简单的自然语言指令下完成各种操作任务。然而，机器人操作是极具挑战性的，因为它需要细粒度的运动控制、长期记忆以及对以前未见过的任务和环境的泛化。为了应对这些挑战，我们提出了一种统一的基于变压器的方法，该方法考虑了多个输入。特别是，我们的变压器架构集成了(I)自然语言指令和(Ii)多视角场景观察，同时(Iii)跟踪观察和行动的完整历史。这种方法能够学习历史和指令之间的依赖关系，并使用多个视图提高操作精度。我们在具有挑战性的RLBENCH基准测试和真实世界的机器人上对我们的方法进行了评估。值得注意的是，我们的方法可扩展到74个不同的RLBch任务，并超过了最先进的水平。我们还解决了指令条件任务，并展示了对以前未见过的变化的出色概括。

**[Paper URL](https://proceedings.mlr.press/v205/guhur23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/guhur23a/guhur23a.pdf)** 

# Embedding Synthetic Off-Policy Experience for Autonomous Driving via Zero-Shot Curricula
**题目:** 通过零镜头课程嵌入自动驾驶的综合非政策体验

**作者:** Eli Bronstein, Sirish Srinivasan, Supratik Paul, Aman Sinha, Matthew O’Kelly, Payam Nikdel, Shimon Whiteson

**Abstract:** ML-based motion planning is a promising approach to produce agents that exhibit complex behaviors, and automatically adapt to novel environments. In the context of autonomous driving, it is common to treat all available training data equally. However, this approach produces agents that do not perform robustly in safety-critical settings, an issue that cannot be addressed by simply adding more data to the training set – we show that an agent trained using only a 10% subset of the data performs just as well as an agent trained on the entire dataset. We present a method to predict the inherent difficulty of a driving situation given data collected from a fleet of autonomous vehicles deployed on public roads. We then demonstrate that this difficulty score can be used in a zero-shot transfer to generate curricula for an imitation-learning based planning agent. Compared to training on the entire unbiased training dataset, we show that prioritizing difficult driving scenarios both reduces collisions by 15% and increases route adherence by 14% in closed-loop evaluation, all while using only 10% of the training data.

**摘要:** 基于ML的运动规划是一种很有前途的方法，可以产生表现出复杂行为的代理，并自动适应新的环境。在自动驾驶的背景下，平等对待所有可用的训练数据是常见的。然而，这种方法产生的代理在安全关键设置下不能很好地执行，这个问题不能通过简单地向训练集添加更多数据来解决-我们证明了仅使用10%的数据子集训练的代理的表现与在整个数据集上训练的代理的表现一样好。我们提出了一种方法，根据从部署在公共道路上的自动驾驶汽车车队收集的数据，预测驾驶情况的固有难度。然后，我们证明了这个难度分数可以用于零命中转移，以生成基于模仿学习的规划代理的课程。与在整个无偏训练数据集上的训练相比，我们表明，在闭环系统评估中，对困难驾驶场景进行优先排序既可以减少15%的碰撞，又可以提高14%的路线忠诚度，而所有这些都只使用了10%的训练数据。

**[Paper URL](https://proceedings.mlr.press/v205/bronstein23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/bronstein23a/bronstein23a.pdf)** 

# LEADER: Learning Attention over Driving Behaviors for Planning under Uncertainty
**题目:** 领导者：学习注意力而不是驾驶行为，以便在不确定性下进行规划

**作者:** Mohamad Hosein Danesh, Panpan Cai, David Hsu

**Abstract:** Uncertainty in human behaviors poses a significant challenge to autonomous driving in crowded urban environments. The partially observable Markov decision process (POMDP) offers a principled general framework for decision making under uncertainty and achieves real-time performance for complex tasks by leveraging Monte Carlo sampling. However, sampling may miss rare, but critical events, leading to potential safety concerns. To tackle this challenge, we propose a new algorithm, LEarning Attention over Driving bEhavioRs (LEADER), which learns to attend to critical human behaviors during planning. LEADER learns a neural network generator to provide attention over human behaviors; it integrates the attention into a belief-space planner through importance sampling, which biases planning towards critical events. To train the attention generator, we form a minimax game between the generator and the planner. By solving this minimax game, LEADER learns to perform risk-aware planning without explicit human effort on data labeling.

**摘要:** 人类行为的不确定性对拥挤的城市环境中的自动驾驶构成了巨大的挑战。部分可观测马尔可夫决策过程(POMDP)为不确定情况下的决策提供了一个有原则的通用框架，并利用蒙特卡罗抽样实现了复杂任务的实时性能。然而，抽样可能会遗漏罕见但关键的事件，导致潜在的安全问题。为了应对这一挑战，我们提出了一种新的算法，学习注意力优先于驾驶行为(Leader)，该算法在规划过程中学习关注关键的人类行为。Leader学习神经网络生成器，以提供对人类行为的关注；它通过重要性抽样将注意力整合到信念空间规划器中，从而使计划偏向于关键事件。为了训练注意力生成器，我们在生成器和规划者之间形成了一个极小极大博弈。通过求解这个极小极大博弈，Leader学会了执行具有风险意识的计划，而无需显式地在数据标签上花费人力。

**[Paper URL](https://proceedings.mlr.press/v205/danesh23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/danesh23a/danesh23a.pdf)** 

# i-Sim2Real: Reinforcement Learning of Robotic Policies in Tight Human-Robot Interaction Loops
**题目:** i-Sim 2 Real：紧密人机交互循环中机器人策略的强化学习

**作者:** Saminda Wishwajith Abeyruwan, Laura Graesser, David B D’Ambrosio, Avi Singh, Anish Shankar, Alex Bewley, Deepali Jain, Krzysztof Marcin Choromanski, Pannag R Sanketi

**Abstract:** Sim-to-real transfer is a powerful paradigm for robotic reinforcement learning. The ability to train policies in simulation enables safe exploration and large-scale data collection quickly at low cost. However, prior works in sim-to-real transfer of robotic policies typically do not involve any human-robot interaction because accurately simulating human behavior is an open problem. In this work, our goal is to leverage the power of simulation to train robotic policies that are proficient at interacting with humans upon deployment. But there is a chicken and egg problem — how to gather examples of a human interacting with a physical robot so as to model human behavior in simulation without already having a robot that is able to interact with a human? Our proposed method, Iterative-Sim-to-Real (i-S2R), attempts to address this. i-S2R bootstraps from a simple model of human behavior and alternates between training in simulation and deploying in the real world. In each iteration, both the human behavior model and the policy are refined. For all training we apply a new evolutionary search algorithm called Blackbox Gradient Sensing (BGS). We evaluate our method on a real world robotic table tennis setting, where the objective for the robot is to play cooperatively with a human player for as long as possible. Table tennis is a high-speed, dynamic task that requires the two players to react quickly to each other’s moves, making for a challenging test bed for research on human-robot interaction. We present results on an industrial robotic arm that is able to cooperatively play table tennis with human players, achieving rallies of 22 successive hits on average and 150 at best. Further, for 80% of players, rally lengths are 70% to 175% longer compared to the sim-to-real plus fine-tuning (S2R+FT) baseline. For videos of our system in action please see https://sites.google.com/view/is2r.

**摘要:** SIM-to-Real迁移是机器人强化学习的有力范例。在模拟中训练策略的能力能够以低成本快速安全地进行探索和大规模数据收集。然而，在机器人策略的模拟到真实传递方面的现有工作通常不涉及任何人-机器人交互，因为准确地模拟人类行为是一个开放的问题。在这项工作中，我们的目标是利用模拟的力量来训练机器人策略，这些策略擅长在部署时与人类交互。但有一个鸡和蛋的问题-如何收集人类与物理机器人互动的例子，以便在没有能够与人类互动的机器人的情况下，在模拟中对人类行为进行建模？我们提出的迭代模拟实数(I-S2R)方法试图解决这个问题。I-S2R从一个简单的人类行为模型开始，在模拟训练和在现实世界中部署之间交替进行。在每一次迭代中，人类行为模型和策略都得到了改进。对于所有的训练，我们应用了一种新的进化搜索算法，称为黑盒梯度感知(BGS)。我们在一个真实的机器人乒乓球环境中对我们的方法进行了评估，其中机器人的目标是尽可能长时间地与人类球员合作。乒乓球是一项高速、动态的任务，需要两名球员对对方的动作做出快速反应，这为研究人与机器人的互动提供了一个具有挑战性的试验台。我们介绍了一种能够与人类运动员合作打乒乓球的工业机械臂的结果，平均达到22连击，最多150次。此外，对于80%的玩家来说，与模拟真实加微调(S2R+FT)基线相比，反弹长度要长70%到175%。有关我们的系统运行的视频，请参阅https://sites.google.com/view/is2r.

**[Paper URL](https://proceedings.mlr.press/v205/abeyruwan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/abeyruwan23a/abeyruwan23a.pdf)** 

# Learning Temporally Extended Skills in Continuous Domains as Symbolic Actions for Planning
**题目:** 在连续领域中学习时间扩展技能作为规划的象征性动作

**作者:** Jan Achterhold, Markus Krimmel, Joerg Stueckler

**Abstract:** Problems which require both long-horizon planning and continuous control capabilities pose significant challenges to existing reinforcement learning agents. In this paper we introduce a novel hierarchical reinforcement learning agent which links temporally extended skills for continuous control with a forward model in a symbolic discrete abstraction of the environment’s state for planning. We term our agent SEADS for Symbolic Effect-Aware Diverse Skills. We formulate an objective and corresponding algorithm which leads to unsupervised learning of a diverse set of skills through intrinsic motivation given a known state abstraction. The skills are jointly learned with the symbolic forward model which captures the effect of skill execution in the state abstraction. After training, we can leverage the skills as symbolic actions using the forward model for long-horizon planning and subsequently execute the plan using the learned continuous-action control skills. The proposed algorithm learns skills and forward models that can be used to solve complex tasks which require both continuous control and long-horizon planning capabilities with high success rate. It compares favorably with other flat and hierarchical reinforcement learning baseline agents and is successfully demonstrated with a real robot.

**摘要:** 需要长期规划和持续控制能力的问题对现有的强化学习代理提出了巨大的挑战。在这篇文章中，我们介绍了一种新的分层强化学习代理，它将时间扩展的连续控制技能与用于规划的环境状态的符号离散抽象的前向模型联系起来。我们称我们的代理SEADS为具有象征性效果感知的各种技能。我们制定了一个客观和相应的算法，通过给定已知的状态抽象，通过内在动机导致对不同技能集的无监督学习。这些技能是通过符号向前模型联合学习的，该模型捕捉了技能执行在状态抽象中的效果。培训后，我们可以利用这些技能作为象征性的行动，使用远期计划的远期模型，并随后使用学习的持续行动控制技能执行计划。该算法学习了可用于解决需要持续控制和长期规划能力的复杂任务的技能和正向模型，成功率很高。与其他平面和层次式强化学习基线代理相比，该算法具有更好的性能，并在真实机器人上得到了成功的验证。

**[Paper URL](https://proceedings.mlr.press/v205/achterhold23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/achterhold23a/achterhold23a.pdf)** 

# Meta-Learning Priors for Safe Bayesian Optimization
**题目:** 安全Bayesian优化的元学习先验

**作者:** Jonas Rothfuss, Christopher Koenig, Alisa Rupenyan, Andreas Krause

**Abstract:** In robotics, optimizing controller parameters under safety constraints is an important challenge. Safe Bayesian optimization (BO) quantifies uncertainty in the objective and constraints to safely guide exploration in such settings. Hand-designing a suitable probabilistic model can be challenging however. In the presence of unknown safety constraints, it is crucial to choose reliable model hyper-parameters to avoid safety violations. Here, we propose a data-driven approach to this problem by em meta-learning priors for safe BO from offline data. We build on a meta-learning algorithm, F-PACOH, capable of providing reliable uncertainty quantification in settings of data scarcity. As core contribution, we develop a novel framework for choosing safety-compliant priors in a data-riven manner via empirical uncertainty metrics and a frontier search algorithm. On benchmark functions and a high-precision motion system, we demonstrate that our meta-learnt priors accelerate convergence of safe BO approaches while maintaining safety.

**摘要:** 在机器人中，在安全约束下优化控制器参数是一个重要的挑战。安全贝叶斯优化(BO)将目标和约束中的不确定性量化，以安全地指导在这种环境下的勘探。然而，手工设计一个合适的概率模型可能是具有挑战性的。在存在未知安全约束的情况下，选择可靠的模型超参数以避免安全违规是至关重要的。在这里，我们提出了一种数据驱动的方法，通过从离线数据中学习安全BO的先验知识来解决这个问题。我们构建了一个元学习算法F-PACOH，它能够在数据稀缺的情况下提供可靠的不确定性量化。作为核心贡献，我们开发了一个新的框架，通过经验不确定性度量和前沿搜索算法，以数据撕裂的方式选择符合安全要求的先验。在基准函数和高精度运动系统上，我们证明了我们的元学习先验加速了安全BO方法的收敛，同时保持了安全。

**[Paper URL](https://proceedings.mlr.press/v205/rothfuss23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/rothfuss23a/rothfuss23a.pdf)** 

# Planning Paths through Occlusions in Urban Environments
**题目:** 规划城市环境中的遮挡路径

**作者:** Yutao Han, Youya Xia, Guo-Jun Qi, Mark Campbell

**Abstract:** This paper presents a novel framework for planning in unknown and occluded urban spaces. We specifically focus on turns and intersections where occlusions significantly impact navigability. Our approach uses an inpainting model to fill in a sparse, occluded, semantic lidar point cloud and plans dynamically feasible paths for a vehicle to traverse through the open and inpainted spaces. We demonstrate our approach using a car’s lidar data with real-time occlusions, and show that by inpainting occluded areas, we can plan longer paths, with more turn options compared to without inpainting; in addition, our approach more closely follows paths derived from a planner with no occlusions (called the ground truth) compared to other state of the art approaches.

**摘要:** 本文提出了一种新的未知和封闭的城市空间规划框架。我们特别关注遮挡会显着影响导航性的转弯和十字路口。我们的方法使用修补模型来填充稀疏、遮挡、语义激光雷达点云，并为车辆穿过开放和修补空间规划动态可行的路径。我们使用具有实时遮挡的汽车激光雷达数据演示了我们的方法，并表明通过修复遮挡区域，我们可以规划更长的路径，与不修复的情况相比，具有更多的转弯选项;此外，与其他最新技术水平的方法相比，我们的方法更紧密地遵循从没有遮挡的规划器中获得的路径（称为地面真相）。

**[Paper URL](https://proceedings.mlr.press/v205/han23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/han23a/han23a.pdf)** 

# Rethinking Optimization with Differentiable Simulation from a Global Perspective
**题目:** 从全球视角重新思考差异化模拟优化

**作者:** Rika Antonova, Jingyun Yang, Krishna Murthy Jatavallabhula, Jeannette Bohg

**Abstract:** Differentiable simulation is a promising toolkit for fast gradient-based policy optimization and system identification. However, existing approaches to differentiable simulation have largely tackled scenarios where obtaining smooth gradients has been relatively easy, such as systems with mostly smooth dynamics. In this work, we study the challenges that differentiable simulation presents when it is not feasible to expect that a single descent reaches a global optimum, which is often a problem in contact-rich scenarios. We analyze the optimization landscapes of diverse scenarios that contain both rigid bodies and deformable objects. In dynamic environments with highly deformable objects and fluids, differentiable simulators produce rugged landscapes with nonetheless useful gradients in some parts of the space. We propose a method that combines Bayesian optimization with semi-local ’leaps’ to obtain a global search method that can use gradients effectively, while also maintaining robust performance in regions with noisy gradients. We show that our approach outperforms several gradient-based and gradient-free baselines on an extensive set of experiments in simulation, and also validate the method using experiments with a real robot and deformables.

**摘要:** 可微仿真是基于梯度的快速政策优化和系统辨识的一个很有前途的工具包。然而，现有的可微模拟方法在很大程度上解决了获得平滑梯度相对容易的场景，例如具有大部分平滑动态的系统。在这项工作中，我们研究了当期望单个下降达到全局最优是不可行的时，可微模拟带来的挑战，这在联系人丰富的场景中通常是一个问题。我们分析了同时包含刚体和变形体的各种场景下的优化景观。在具有高度可变形物体和流体的动态环境中，可区分模拟器在空间的某些部分生成具有仍然有用的渐变的崎岖景观。我们提出了一种结合贝叶斯优化和半局部LEAPS的方法，以获得一种全局搜索方法，该方法可以有效地利用梯度，同时在含有噪声的梯度区域保持稳健的性能。在大量的仿真实验中，我们证明了我们的方法优于几种基于梯度和无梯度的基线，并通过真实机器人和变形器的实验验证了方法的有效性。

**[Paper URL](https://proceedings.mlr.press/v205/antonova23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/antonova23a/antonova23a.pdf)** 

# Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
**题目:** 做我能做的事，而不是按我说的做：机器人功能的基础语言

**作者:** brian ichter, Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, Dmitry Kalashnikov, Sergey Levine, Yao Lu, Carolina Parada, Kanishka Rao, Pierre Sermanet, Alexander T Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Mengyuan Yan, Noah Brown, Michael Ahn, Omar Cortes, Nicolas Sievers, Clayton Tan, Sichun Xu, Diego Reyes, Jarek Rettinghouse, Jornell Quiambao, Peter Pastor, Linda Luu, Kuang-Huei Lee, Yuheng Kuang, Sally Jesmonth, Nikhil J. Joshi, Kyle Jeffrey, Rosario Jauregui Ruano, Jasmine Hsu, Keerthana Gopalakrishnan, Byron David, Andy Zeng, Chuyuan Kelly Fu

**Abstract:** Large language models can encode a wealth of semantic knowledge about the world. Such knowledge could be extremely useful to robots aiming to act upon high-level, temporally extended instructions expressed in natural language. However, a significant weakness of language models is that they lack real-world experience, which makes it difficult to leverage them for decision making within a given embodiment. For example, asking a language model to describe how to clean a spill might result in a reasonable narrative, but it may not be applicable to a particular agent, such as a robot, that needs to perform this task in a particular environment. We propose to provide real-world grounding by means of pretrained skills, which are used to constrain the model to propose natural language actions that are both feasible and contextually appropriate. The robot can act as the language model’s “hands and eyes,” while the language model supplies high-level semantic knowledge about the task. We show how low-level skills can be combined with large language models so  that  the  language model  provides  high-level  knowledge about the procedures for performing complex and temporally extended instructions,  while  value  functions  associated  with  these  skills  provide  the  grounding necessary to connect this knowledge to a particular physical environment. We evaluate our method on a number of real-world robotic tasks, where we show the need for real-world grounding and that this approach is capable of completing long-horizon, abstract, natural language instructions on a mobile manipulator. The project’s website, video, and open source can be found at say-can.github.io.

**摘要:** 大型语言模型可以编码关于世界的丰富语义知识。这样的知识对机器人来说可能非常有用，这些机器人旨在执行以自然语言表达的高级、临时扩展的指令。然而，语言模型的一个重要弱点是它们缺乏实际经验，这使得在给定的实施例中利用它们进行决策是困难的。例如，要求语言模型描述如何清理泄漏可能会导致合理的叙述，但它可能不适用于需要在特定环境中执行此任务的特定代理，如机器人。我们建议通过预先训练的技能来提供现实世界的基础，这些技能被用来约束模型提出既可行又上下文合适的自然语言动作。机器人可以充当语言模型的“手和眼”，而语言模型则提供有关任务的高级语义知识。我们展示了如何将低级别技能与大型语言模型相结合，以便语言模型提供有关执行复杂和临时扩展指令的过程的高级知识，而与这些技能相关联的值函数提供将这些知识连接到特定物理环境所需的基础。我们在一些真实的机器人任务上评估了我们的方法，其中我们展示了真实世界接地的必要性，并且这种方法能够在移动机械手上完成长期的、抽象的、自然语言的指令。该项目的网站、视频和开放源码可以在Say-can.githorb.io上找到。

**[Paper URL](https://proceedings.mlr.press/v205/ichter23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ichter23a/ichter23a.pdf)** 

# MidasTouch: Monte-Carlo inference over distributions across sliding touch
**题目:** MidasTouch：对滑动触摸分布的蒙特卡罗推断

**作者:** Sudharshan Suresh, Zilin Si, Stuart Anderson, Michael Kaess, Mustafa Mukadam

**Abstract:** We present MidasTouch, a tactile perception system for online global localization of a vision-based touch sensor sliding on an object surface. This framework takes in posed tactile images over time, and outputs an evolving distribution of sensor pose on the object’s surface, without the need for visual priors. Our key insight is to estimate local surface geometry with tactile sensing, learn a compact representation for it, and disambiguate these signals over a long time horizon. The backbone of MidasTouch is a Monte-Carlo particle filter, with a measurement model based on a tactile code network learned from tactile simulation. This network, inspired by LIDAR place recognition, compactly summarizes local surface geometries. These generated codes are efficiently compared against a precomputed tactile codebook per-object, to update the pose distribution. We further release the YCB-Slide dataset of real-world and simulated forceful sliding interactions between a vision-based tactile sensor and standard YCB objects. While single-touch localization can be inherently ambiguous, we can quickly localize our sensor by traversing salient surface geometries. Project page: https://suddhu.github.io/midastouch-tactile/

**摘要:** 我们提出了MidasTouch，这是一个触觉感知系统，用于在线定位对象表面上滑动的基于视觉的触摸传感器。该框架接收姿势触觉图像，并输出对象表面上传感器姿势的演变分布，而不需要视觉先验。我们的关键洞察力是用触觉感知来估计局部表面几何形状，学习它的紧凑表示，并在长时间范围内消除这些信号的歧义。MidasTouch的核心是蒙特卡洛粒子过滤器，其测量模型基于从触觉模拟中学习的触觉代码网络。该网络受LIDAR位置识别的启发，简洁地总结了局部曲面几何形状。这些生成的代码被有效地与每个对象的预计算触觉码本进行比较，以更新姿势分布。我们进一步发布了YCB-Slide数据集，其中包括基于视觉的触觉传感器和标准YCB对象之间的真实和模拟的强力滑动交互。虽然单点触摸定位可能本质上是模棱两可的，但我们可以通过遍历突出的表面几何图形来快速定位传感器。项目页面：https://suddhu.github.io/midastouch-tactile/

**[Paper URL](https://proceedings.mlr.press/v205/suresh23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/suresh23a/suresh23a.pdf)** 

# Learning Visuo-Haptic Skewering Strategies for Robot-Assisted Feeding
**题目:** 学习机器人辅助喂食的视觉触觉倾斜策略

**作者:** Priya Sundaresan, Suneel Belkhale, Dorsa Sadigh

**Abstract:** Acquiring food items with a fork poses an immense challenge to a robot-assisted feeding system, due to the wide range of material properties and visual appearances present across food groups. Deformable foods necessitate different skewering strategies than firm ones, but inferring such characteristics for several previously unseen items on a plate remains nontrivial. Our key insight is to leverage visual and haptic observations during interaction with an item to rapidly and reactively plan skewering motions. We learn a generalizable, multimodal representation for a food item from raw sensory inputs which informs the optimal skewering strategy. Given this representation, we propose a zero-shot framework to sense visuo-haptic properties of a previously unseen item and reactively skewer it, all within a single interaction. Real-robot experiments with foods of varying levels of visual and textural diversity demonstrate that our multimodal policy outperforms baselines which do not exploit both visual and haptic cues or do not reactively plan. Across 6 plates of different food items, our proposed framework achieves 71% success over 69 skewering attempts total. Supplementary material, code, and videos can be found on our website: https://sites.google.com/view/hapticvisualnet-corl22/home.

**摘要:** 用叉子获取食物对机器人辅助喂养系统构成了巨大的挑战，因为食物组之间存在着广泛的材料特性和视觉外观。可变形的食物需要不同的串策略，而不是坚固的策略，但为一个盘子里以前看不到的几种食物推断这样的特征仍然不是一件微不足道的事情。我们的关键洞察力是在与物品互动的过程中利用视觉和触觉观察来快速和反应性地计划倾斜运动。我们从原始的感官输入中学习一种可概括的、多模式的食品表示，它通知最优的串策略。给出了这种表示，我们提出了一个零镜头框架来感知以前未见过的物品的视觉触觉属性，并对其进行反应性倾斜，所有这些都在单个交互中完成。对视觉和纹理多样性程度不同的食物进行的真实机器人实验表明，我们的多模式政策优于没有同时利用视觉和触觉线索或没有反应性规划的基线。在6个不同食物的盘子上，我们提出的框架在总共69次烤串尝试中取得了71%的成功率。补充材料、代码和视频可在我们的网站上找到：https://sites.google.com/view/hapticvisualnet-corl22/home.

**[Paper URL](https://proceedings.mlr.press/v205/sundaresan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/sundaresan23a/sundaresan23a.pdf)** 

# Learning Agile Skills via Adversarial Imitation of Rough Partial Demonstrations
**题目:** 通过对粗糙部分演示的对抗模仿学习敏捷技能

**作者:** Chenhao Li, Marin Vlastelica, Sebastian Blaes, Jonas Frey, Felix Grimminger, Georg Martius

**Abstract:** Learning agile skills is one of the main challenges in robotics. To this end, reinforcement learning approaches have achieved impressive results. These methods require explicit task information in terms of a reward function or an expert that can be queried in simulation to provide a target control output, which limits their applicability. In this work, we propose a generative adversarial method for inferring reward functions from partial and potentially physically incompatible demonstrations for successful skill acquirement where reference or expert demonstrations are not easily accessible. Moreover, we show that by using a Wasserstein GAN formulation and transitions from demonstrations with rough and partial information as input, we are able to extract policies that are robust and capable of imitating demonstrated behaviors. Finally, the obtained skills such as a backflip are tested on an agile quadruped robot called Solo 8 and present faithful replication of hand-held human demonstrations.

**摘要:** 学习敏捷技能是机器人学的主要挑战之一。为此，强化学习方法取得了令人印象深刻的成果。这些方法需要以奖励函数或专家的形式明确的任务信息，可以在仿真中查询以提供目标控制输出，这限制了它们的适用性。在这项工作中，我们提出了一种生成性对抗性方法，用于从部分和潜在的物理不相容的演示中推断奖励函数，以便在参考或专家演示不易获得的情况下成功获得技能。此外，我们还证明了，通过使用Wasserstein GAN公式和带有粗略和部分信息的演示的转换作为输入，我们能够提取健壮的、能够模拟演示行为的策略。最后，所获得的技能，如后空翻，在一个名为Solo 8的敏捷四足机器人上进行了测试，并忠实地复制了手持式人类演示。

**[Paper URL](https://proceedings.mlr.press/v205/li23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/li23b/li23b.pdf)** 

# Evo-NeRF: Evolving NeRF for Sequential Robot Grasping of Transparent Objects
**题目:** Evo-NeRF：用于机器人顺序抓取透明物体的演变NeRF

**作者:** Justin Kerr, Letian Fu, Huang Huang, Yahav Avigal, Matthew Tancik, Jeffrey Ichnowski, Angjoo Kanazawa, Ken Goldberg

**Abstract:** Sequential robot grasping of transparent objects, where a robot removes objects one by one from a workspace, is important in many industrial and household scenarios. We propose Evolving NeRF (Evo-NeRF), leveraging recent speedups in NeRF training and further extending it to rapidly train the NeRF representation concurrently to image capturing. Evo-NeRF terminates training early when a sufficient task confidence is achieved, evolves the NeRF weights from grasp to grasp to rapidly adapt to object removal, and applies additional geometry regularizations to make the reconstruction smoother and faster. General purpose grasp planners such as Dex-Net may underperform with NeRF outputs because there can be unreliable geometry from rapidly trained NeRFs. To mitigate this distribution shift, we propose a Radiance-Adjusted Grasp Network (RAG-Net), a grasping network adapted to NeRF’s characteristics through training on depth rendered from NeRFs of synthetic scenes. In experiments, a physical YuMi robot using Evo-NeRF and RAG-Net achieves an 89% grasp success rate over 27 trials on single objects, with early capture termination providing a 41% speed improvement with no loss in reliability. In a sequential grasping task on 6 scenes, Evo-NeRF reusing network weights clears 72% of the objects, retaining similar performance as reconstructing the NeRF from scratch (76%) but using 61% less sensing time. See https://sites.google.com/view/evo-nerf for more materials.

**摘要:** 机器人顺序抓取透明物体，即机器人将物体逐个从工作空间中移除，这在许多工业和家庭场景中非常重要。我们提出了进化神经网络(EVO-NERF)，利用最近在神经网络训练中的加速，并进一步扩展它以快速训练并行于图像捕获的神经网络表示。当获得足够的任务置信度时，Evo-Nerf提前终止训练，使神经网络的权值从抓取进化到抓取以快速适应目标去除，并应用额外的几何正则化使重建更平滑和更快。像Dex-Net这样的通用GRAP规划器在使用NERF输出时可能表现不佳，因为快速训练的NERF可能会产生不可靠的几何图形。为了缓解这种分布变化，我们提出了一种辐射度调整的抓取网络(RAG-Net)，这是一种通过训练合成场景的神经网络绘制的深度来适应NERF特征的抓取网络。在实验中，使用Evo-Nerf和RAG-Net的物理YuMi机器人在对单个对象的27次试验中获得了89%的抓取成功率，早期捕获终止在不损失可靠性的情况下使速度提高了41%。在6个场景的连续抓取任务中，重复使用网络权重的Evo-Nerf清除了72%的对象，保持了与从头开始重建NERF相似的性能(76%)，但使用的感知时间减少了61%。有关更多材质，请参见https://sites.google.com/view/evo-nerf。

**[Paper URL](https://proceedings.mlr.press/v205/kerr23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/kerr23a/kerr23a.pdf)** 

# Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision
**题目:** Fleet-Dagger：具有可扩展人类监督的交互式机器人舰队学习

**作者:** Ryan Hoque, Lawrence Yunliang Chen, Satvik Sharma, Karthik Dharmarajan, Brijen Thananjeyan, Pieter Abbeel, Ken Goldberg

**Abstract:** Commercial and industrial deployments of robot fleets at Amazon, Nimble, Plus One, Waymo, and Zoox query remote human teleoperators when robots are at risk or unable to make task progress. With continual learning, interventions from the remote pool of humans can also be used to improve the robot fleet control policy over time. A central question is how to effectively allocate limited human attention. Prior work addresses this in the single-robot, single-human setting; we formalize the Interactive Fleet Learning (IFL) setting, in which multiple robots interactively query and learn from multiple human supervisors. We propose Return on Human Effort (ROHE) as a new metric and Fleet-DAgger, a family of IFL algorithms. We present an open-source IFL benchmark suite of GPU-accelerated Isaac Gym environments for standardized evaluation and development of IFL algorithms. We compare a novel Fleet-DAgger algorithm to 4 baselines with 100 robots in simulation. We also perform a physical block-pushing experiment with 4 ABB YuMi robot arms and 2 remote humans. Experiments suggest that the allocation of humans to robots significantly affects the performance of the fleet, and that the novel Fleet-DAgger algorithm can achieve up to 8.8x higher ROHE than baselines. See https://tinyurl.com/fleet-dagger for supplemental material.

**摘要:** 当机器人面临风险或无法完成任务时，亚马逊、Nimble、Plus One、Waymo和Zoox的机器人舰队的商业和工业部署可以查询远程人类遥控操作员。在不断学习的情况下，来自远程人类池的干预也可以用于随着时间的推移改进机器人舰队控制政策。一个核心问题是如何有效地分配有限的人类注意力。以前的工作在单机器人、单人设置中解决了这一问题；我们将交互舰队学习(IFL)设置正式化，在该设置中，多个机器人交互地向多个人类主管查询和学习。我们提出了人类努力回报(ROHE)作为一种新的度量标准，以及IFL算法家族Fleet-Dagger。我们提出了一个开源的基于GPU加速的Isaac Gym环境的IFL基准测试套件，用于对IFL算法进行标准化评估和开发。我们将一种新的Fleet-Dagger算法与100个机器人的4条基线进行了仿真比较。我们还用4个ABB YuMi机器人手臂和2个远程人类进行了物理块推实验。实验表明，人类到机器人的分配显著影响了舰队的性能，并且新的Fleet-Dagger算法可以获得比基线高8.8倍的RoHE。有关补充材料，请参阅https://tinyurl.com/fleet-dagger。

**[Paper URL](https://proceedings.mlr.press/v205/hoque23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/hoque23a/hoque23a.pdf)** 

# RAP: Risk-Aware Prediction for Robust Planning
**题目:** RAP：稳健规划的风险意识预测

**作者:** Haruki Nishimura, Jean Mercat, Blake Wulfe, Rowan Thomas McAllister, Adrien Gaidon

**Abstract:** Robust planning in interactive scenarios requires predicting the uncertain future to make risk-aware decisions. Unfortunately, due to long-tail safety-critical events, the risk is often under-estimated by finite-sampling approximations of probabilistic motion forecasts. This can lead to overconfident and unsafe robot behavior, even with robust planners. Instead of assuming full prediction coverage that robust planners require, we propose to make prediction itself risk-aware. We introduce a new prediction objective to learn a risk-biased distribution over trajectories, so that risk evaluation simplifies to an expected cost estimation under this biased distribution. This reduces sample complexity of the risk estimation during online planning, which is needed for safe real-time performance. Evaluation results in a didactic simulation environment and on a real-world dataset demonstrate the effectiveness of our approach. The code and a demo are available.

**摘要:** 交互式场景中的稳健规划需要预测不确定的未来，以做出风险意识的决策。不幸的是，由于长尾安全关键事件，概率运动预测的有限抽样逼近常常低估了风险。即使有强大的规划者，这也可能导致机器人过度自信和不安全的行为。我们建议使预测本身具有风险意识，而不是假设稳健的规划者所需的全面预测覆盖范围。我们引入了一个新的预测目标来学习轨迹上的风险偏向分布，以便风险评估简化为这种偏向分布下的预期成本估计。这降低了在线规划期间风险估计的样本复杂性，而这是安全实时性能所需的。教学模拟环境和现实世界数据集中的评估结果证明了我们方法的有效性。代码和演示已提供。

**[Paper URL](https://proceedings.mlr.press/v205/nishimura23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/nishimura23a/nishimura23a.pdf)** 

# Topological Semantic Graph Memory for Image-Goal Navigation
**题目:** 图像-目标导航的Topic Semantic Map记忆

**作者:** Nuri Kim, Obin Kwon, Hwiyeon Yoo, Yunho Choi, Jeongho Park, Songhwai Oh

**Abstract:** A novel framework is proposed to incrementally collect landmark-based graph memory and use the collected memory for image goal navigation. Given a target image to search, an embodied robot utilizes semantic memory to find the target in an unknown environment. In this paper, we present a topological semantic graph memory (TSGM), which consists of (1) a graph builder that takes the observed RGB-D image to construct a topological semantic graph, (2) a cross graph mixer module that takes the collected nodes to get contextual information, and (3) a memory decoder that takes the contextual memory as an input to find an action to the target. On the task of an image goal navigation, TSGM significantly outperforms competitive baselines by +5.0-9.0% on the success rate and +7.0-23.5% on SPL, which means that the TSGM finds efficient paths. Additionally, we demonstrate our method on a mobile robot in real-world image goal scenarios.

**摘要:** 提出了一种新颖的框架来增量地收集基于地标的图内存，并将收集的内存用于图像目标导航。给定要搜索的目标图像，嵌入式机器人利用语义记忆在未知环境中找到目标。在本文中，我们提出了一种topical semantic graph内存（TSGM），它由（1）一个图构建器，用于获取观察到的RGB-D图像来构建一个topical semantic graph，（2）一个交叉图混合器模块，用于获取收集的节点来获取上下文信息，以及（3）一个内存解码器，用于将上下文记忆作为输入来查找目标的动作。在图像目标导航任务中，TSGM的成功率显着优于竞争基准，增幅为+5.0-9.0%，SPL为+7.0-23.5%，这意味着TSGM找到了有效的路径。此外，我们还在现实世界图像目标场景中的移动机器人上展示了我们的方法。

**[Paper URL](https://proceedings.mlr.press/v205/kim23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/kim23a/kim23a.pdf)** 

# Legged Locomotion in Challenging Terrains using Egocentric Vision
**题目:** 使用自我中心愿景在挑战地形中进行腿部运动

**作者:** Ananye Agarwal, Ashish Kumar, Jitendra Malik, Deepak Pathak

**Abstract:** Animals are capable of precise and agile locomotion using vision. Replicating this ability has been a long-standing goal in robotics. The traditional approach has been to decompose this problem into elevation mapping and foothold planning phases. The elevation mapping, however, is susceptible to failure and large noise artifacts, requires specialized hardware, and is biologically implausible. In this paper, we present the first end-to-end locomotion system capable of traversing stairs, curbs, stepping stones, and gaps. We show this result on a medium-sized quadruped robot using a single front-facing depth camera. The small size of the robot necessitates discovering specialized gait patterns not seen elsewhere. The egocentric camera requires the policy to remember past information to estimate the terrain under its hind feet. We train our policy in simulation. Training has two phases - first, we train a policy using reinforcement learning with a cheap-to-compute variant of depth image and then in phase 2 distill it into the final policy that uses depth using supervised learning. The resulting policy transfers to the real world and is able to run in real-time on the limited compute of the robot. It can traverse a large variety of terrain while being robust to perturbations like pushes, slippery surfaces, and rocky terrain. Videos are at https://vision-locomotion.github.io

**摘要:** 动物能够使用视觉进行精确和灵活的运动。复制这种能力一直是机器人学的一个长期目标。传统的方法是将这个问题分解为高程测绘和立足点规划阶段。然而，高程映射容易出现故障和大的噪声伪影，需要专门的硬件，而且在生物学上是不可信的。在本文中，我们提出了第一个端到端的移动系统，能够穿越楼梯、路缘、踏脚石和缝隙。我们在一个中等大小的四足机器人上展示了这一结果，该机器人使用了一个前置深度相机。机器人的体积很小，需要发现在其他地方看不到的特殊步态模式。以自我为中心的相机要求政策记住过去的信息，以估计其后脚下的地形。我们在模拟中训练我们的政策。训练分为两个阶段-首先，我们使用强化学习和计算成本较低的深度图像变体来训练策略，然后在第二阶段使用监督学习将其提取为使用深度的最终策略。由此产生的策略转移到现实世界，并能够在机器人有限的计算上实时运行。它可以穿越各种地形，同时对推力、光滑表面和岩石地形等扰动具有健壮性。视频请访问https://vision-locomotion.github.io

**[Paper URL](https://proceedings.mlr.press/v205/agarwal23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/agarwal23a/agarwal23a.pdf)** 

# Real-World Robot Learning with Masked Visual Pre-training
**题目:** 通过掩蔽视觉预训练进行现实世界机器人学习

**作者:** Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, Trevor Darrell

**Abstract:** In this work, we explore self-supervised visual pre-training on images from diverse, in-the-wild videos for real-world robotic tasks. Like prior work, our visual representations are pre-trained via a masked autoencoder (MAE), frozen, and then passed into a learnable control module. Unlike prior work, we show that the pre-trained representations are effective across a range of real-world robotic tasks and embodiments. We find that our encoder consistently outperforms CLIP (up to 75%), supervised ImageNet pre-training (up to 81%), and training from scratch (up to 81%). Finally, we train a 307M parameter vision transformer on a massive collection of 4.5M images from the Internet and egocentric videos, and demonstrate clearly the benefits of scaling visual pre-training for robot learning.

**摘要:** 在这项工作中，我们探索了对来自不同野外视频的图像进行自我监督视觉预训练，以执行现实世界的机器人任务。与之前的工作一样，我们的视觉表示通过掩蔽自动编码器（MAE）进行预训练、冻结，然后传递到可学习的控制模块中。与之前的工作不同，我们表明预训练的表示在一系列现实世界的机器人任务和实施中都有效。我们发现我们的编码器始终优于CLIP（高达75%）、监督ImageNet预训练（高达81%）和从头开始训练（高达81%）。最后，我们在来自互联网的450万张图像和以自我为中心的视频的大量集合上训练307 M参数视觉Transformer，并清楚地展示了扩展视觉预训练用于机器人学习的好处。

**[Paper URL](https://proceedings.mlr.press/v205/radosavovic23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/radosavovic23a/radosavovic23a.pdf)** 

# SE(2)-Equivariant Pushing Dynamics Models for Tabletop Object Manipulations
**题目:** SE（2）-桌面对象操纵的等变推动动力学模型

**作者:** Seungyeon Kim, Byeongdo Lim, Yonghyeon Lee, Frank C. Park

**Abstract:** For tabletop object manipulation tasks, learning an accurate pushing dynamics model, which predicts the objects’ motions when a robot pushes an object, is very important. In this work, we claim that an ideal pushing dynamics model should have the SE(2)-equivariance property, i.e., if tabletop objects’ poses and pushing action are transformed by some same planar rigid-body transformation, then the resulting motion should also be the result of the same transformation. Existing state-of-the-art data-driven approaches do not have this equivariance property, resulting in less-than-desirable learning performances. In this paper, we propose a new neural network architecture that by construction has the above equivariance property. Through extensive empirical validations, we show that the proposed model shows significantly improved learning performances over the existing methods. Also, we verify that our pushing dynamics model can be used for various downstream pushing manipulation tasks such as the object moving, singulation, and grasping in both simulation and real robot experiments. Code is available at https://github.com/seungyeon-k/SQPDNet-public.

**摘要:** 对于桌面对象操作任务，学习准确的推动动力学模型是非常重要的，该模型预测了机器人推动对象时对象的运动。在这项工作中，我们认为一个理想的推动动力学模型应该具有SE(2)-等变性质，即如果桌面物体的姿势和推动动作是通过相同的平面刚体变换来变换的，那么所产生的运动也应该是相同变换的结果。现有的最先进的数据驱动方法不具有这种等方差特性，导致学习性能不佳。在本文中，我们提出了一种新的神经网络结构，它通过构造具有上述等方差性质。通过大量的实验验证，我们发现该模型的学习性能比现有的方法有了显著的提高。同时，在仿真实验和真实机器人实验中，我们验证了我们的推力动力学模型可以用于各种下游推力操作任务，如物体的移动、分离和抓取。代码可在https://github.com/seungyeon-k/SQPDNet-public.上找到

**[Paper URL](https://proceedings.mlr.press/v205/kim23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/kim23b/kim23b.pdf)** 

# Vision-based Uneven BEV Representation Learning with Polar Rasterization and Surface Estimation
**题目:** 基于视觉的不均匀BEV表示学习，具有极网格化和表面估计

**作者:** Zhi Liu, Shaoyu Chen, Xiaojie Guo, Xinggang Wang, Tianheng Cheng, Hongmei Zhu, Qian Zhang, Wenyu Liu, Yi Zhang

**Abstract:** In this work, we propose PolarBEV  for vision-based uneven BEV representation learning. To adapt to the foreshortening effect of camera imaging, we rasterize the BEV space both angularly and radially, and introduce polar embedding decomposition to model the associations among polar grids.  Polar grids are rearranged to an array-like regular representation for efficient processing. Besides, to determine the 2D-to-3D correspondence, we iteratively update the BEV surface based on a hypothetical plane, and adopt height-based  feature transformation. PolarBEV keeps real-time inference speed on a single 2080Ti GPU, and outperforms other methods for both BEV semantic segmentation and BEV instance segmentation. Thorough ablations  are presented to validate the design. The code will be released for facilitating further research.

**摘要:** 在这项工作中，我们提出了PolarBEV用于基于视觉的不均匀BEV表示学习。为了适应相机成像的透视效果，我们对BEV空间进行角度和放射状网格化，并引入极嵌入分解来建模极网格之间的关联。  极网格被重新排列为类似阵列的规则表示，以实现高效处理。此外，为了确定2D到3D的对应关系，我们基于假设平面迭代更新BEV表面，并采用基于高度的特征转换。PolarBEV在单个2080 Ti图形处理器上保持实时推理速度，并且在BEV语义分割和BEV实例分割方面优于其他方法。进行彻底消融以验证设计。该代码将发布以促进进一步研究。

**[Paper URL](https://proceedings.mlr.press/v205/liu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/liu23a/liu23a.pdf)** 

# HERD: Continuous Human-to-Robot Evolution for Learning from Human Demonstration
**题目:** HERD：从人类演示中学习的持续人类到机器人进化

**作者:** Xingyu Liu, Deepak Pathak, Kris M. Kitani

**Abstract:** The ability to learn from human demonstration endows robots with the ability to automate various tasks. However, directly learning from human demonstration is challenging since the structure of the human hand can be very different from the desired robot gripper. In this work, we show that manipulation skills can be transferred from a human to a robot through the use of micro-evolutionary reinforcement learning, where a five-finger human dexterous hand robot gradually evolves into a commercial robot, while repeated interacting in a physics simulator to continuously update the policy that is first learned from human demonstration. To deal with the high dimensions of robot parameters, we propose an algorithm for multi-dimensional evolution path searching that allows joint optimization of both the robot evolution path and the policy. Through experiments on human object manipulation datasets, we show that our framework can efficiently transfer the expert human agent policy trained from human demonstrations in diverse modalities to target commercial robots.

**摘要:** 从人类演示中学习的能力赋予了机器人自动化各种任务的能力。然而，直接从人类演示中学习是具有挑战性的，因为人类手的结构可能与所需的机器人抓手非常不同。在这项工作中，我们证明了通过使用微进化强化学习，操作技能可以从人类转移到机器人身上，其中五指人类灵巧手机器人逐渐进化成商业机器人，同时在物理模拟器中反复交互，不断更新最初从人类演示中学习的策略。针对机器人参数的高维性，提出了一种多维进化路径搜索算法，该算法允许机器人进化路径和策略的联合优化。通过在人类对象操作数据集上的实验，我们的框架可以有效地将从不同形式的人类演示训练的专家人类代理策略转移到目标商业机器人上。

**[Paper URL](https://proceedings.mlr.press/v205/liu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/liu23b/liu23b.pdf)** 

# PlanT: Explainable Planning Transformers via Object-Level Representations
**题目:** PlanT：通过对象级表示的可解释规划变形金刚

**作者:** Katrin Renz, Kashyap Chitta, Otniel-Bogdan Mercea, A. Sophia Koepke, Zeynep Akata, Andreas Geiger

**Abstract:** Planning an optimal route in a complex environment requires efficient reasoning about the surrounding scene. While human drivers prioritize important objects and ignore details not relevant to the decision, learning-based planners typically extract features from dense, high-dimensional grid representations containing all vehicle and road context information. In this paper, we propose PlanT, a novel approach for planning in the context of self-driving that uses a standard transformer architecture. PlanT is based on imitation learning with a compact object-level input representation. On the Longest6 benchmark for CARLA, PlanT outperforms all prior methods (matching the driving score of the expert) while being 5.3× faster than equivalent pixel-based planning baselines during inference. Combining PlanT with an off-the-shelf perception module provides a sensor-based driving system that is more than 10 points better in terms of driving score than the existing state of the art. Furthermore, we propose an evaluation protocol to quantify the ability of planners to identify relevant objects, providing insights regarding their decision-making. Our results indicate that PlanT can focus on the most relevant object in the scene, even when this object is geometrically distant.

**摘要:** 在复杂环境中规划最优路径需要对周围场景进行有效的推理。虽然人类司机优先考虑重要的对象，忽略与决策无关的细节，但基于学习的规划者通常从包含所有车辆和道路环境信息的密集高维网格表示中提取特征。在本文中，我们提出了一种新的方法，在自动驾驶的背景下，使用标准的变压器架构来进行规划。植物基于模仿学习，具有紧凑的对象级输入表示。在CALA的Longest6基准上，PLANT的性能优于所有先前的方法(匹配专家的驾驶分数)，同时在推理过程中比同等的基于像素的规划基线快5.3倍。将工厂与现成的感知模块相结合，提供了一种基于传感器的驾驶系统，在驾驶分数方面比现有的最先进水平高出10分以上。此外，我们提出了一个评估方案来量化规划者识别相关对象的能力，为他们的决策提供洞察力。我们的结果表明，植物可以聚焦在场景中最相关的对象上，即使该对象在几何距离上也是如此。

**[Paper URL](https://proceedings.mlr.press/v205/renz23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/renz23a/renz23a.pdf)** 

# Where To Start? Transferring Simple Skills to Complex Environments
**题目:** 从哪里开始？将简单技能转移到复杂环境

**作者:** Vitalis Vosylius, Edward Johns

**Abstract:** Robot learning provides a number of ways to teach robots simple skills, such as grasping. However, these skills are usually trained in open, clutter-free environments, and therefore would likely cause undesirable collisions in more complex, cluttered environments. In this work, we introduce an affordance model based on a graph representation of an environment, which is optimised during deployment to find suitable robot configurations to start a skill from, such that the skill can be executed without any collisions. We demonstrate that our method can generalise a priori acquired skills to previously unseen cluttered and constrained environments, in simulation and in the real world, for both a grasping and a placing task.

**摘要:** 机器人学习提供了多种方法来教机器人简单的技能，例如抓取。然而，这些技能通常是在开放、无混乱的环境中训练的，因此可能会在更复杂、混乱的环境中造成不良碰撞。在这项工作中，我们引入了一种基于环境图形表示的启示模型，该模型在部署期间进行优化，以找到合适的机器人配置来启动技能，以便可以在没有任何冲突的情况下执行技能。我们证明，我们的方法可以将先验获得的技能推广到模拟和现实世界中以前看不见的混乱和约束环境中，用于抓取和放置任务。

**[Paper URL](https://proceedings.mlr.press/v205/vosylius23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/vosylius23a/vosylius23a.pdf)** 

# Efficient and Stable Off-policy Training via Behavior-aware Evolutionary Learning
**题目:** 通过行为感知进化学习进行高效稳定的非政策培训

**作者:** Maiyue Chen, Guangyi He

**Abstract:** Applying reinforcement learning (RL) algorithms to real-world continuos control problems faces many challenges in terms of sample efficiency, stability and exploration. Off-policy RL algorithms show great sample efficiency but can be unstable to train and require effective exploration techniques for sparse reward environments. A simple yet effective approach to address these challenges is to train a population of policies and ensemble them in certain ways. In this work, a novel population based evolutionary training framework inspired by evolution strategies (ES) called Behavior-aware Evolutionary Learning (BEL) is proposed. The main idea is to train a population of behaviorally diverse policies in parallel and conduct selection with simple linear recombination. BEL consists of two mechanisms called behavior-regularized perturbation (BRP) and behavior-targeted training (BTT) to accomplish stable and fine control of the population behavior divergence. Experimental studies showed that BEL not only has superior sample efficiency and stability compared to existing methods, but can also produce diverse agents in sparse reward environments. Due to the parallel implementation, BEL also exhibits relatively good computation efficiency, making it a practical and competitive method to train policies for real-world robots.

**摘要:** 将强化学习(RL)算法应用于实际连续系统控制问题，在样本效率、稳定性和探索性方面面临着许多挑战。非策略RL算法表现出很高的样本效率，但训练不稳定，需要针对稀疏奖励环境的有效探索技术。应对这些挑战的一种简单而有效的方法是培训一批政策，并以某种方式将它们整合在一起。本文提出了一种受进化策略启发的基于群体的进化训练框架，称为行为感知进化学习(BEL)。其主要思想是同时训练一批行为不同的政策，并通过简单的线性重组进行选择。BELL由行为规则化扰动(BRP)和行为定向训练(BTT)两种机制组成，以实现对群体行为分歧的稳定和精细控制。实验研究表明，与现有方法相比，BEL不仅具有更好的样本效率和稳定性，而且可以在稀疏奖励环境中产生多种代理。由于并行实现，BEL也表现出相对较好的计算效率，使其成为一种实用且具有竞争力的方法来训练现实世界中的机器人的策略。

**[Paper URL](https://proceedings.mlr.press/v205/chen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/chen23a/chen23a.pdf)** 

# LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action
**题目:** LM-Nav：具有大型预训练语言、视觉和动作模型的机器人导航

**作者:** Dhruv Shah, Błażej Osiński, brian ichter, Sergey Levine

**Abstract:** Goal-conditioned policies for robotic navigation can be trained on large, unannotated datasets, providing for good generalization to real-world settings. However, particularly in vision-based settings where specifying goals requires an image, this makes for an unnatural interface. Language provides a more convenient modality for communication with robots, but contemporary methods typically require expensive supervision, in the form of trajectories annotated with language descriptions. We present a system, LM-Nav, for robotic navigation that enjoys the benefits of training on unannotated large datasets of trajectories, while still providing a high-level interface to the user. Instead of utilizing a labeled instruction following dataset, we show that such a system can be constructed entirely out of pre-trained models for navigation (ViNG), image-language association (CLIP), and language modeling (GPT-3), without requiring any fine-tuning or language-annotated robot data. LM-Nav extracts landmarks names from an instruction, grounds them in the world via the image-language model, and then reaches them via the (vision-only) navigation model. We instantiate LM-Nav on a real-world  mobile robot and demonstrate long-horizon navigation through complex, outdoor environments from natural language instructions.

**摘要:** 机器人导航的目标约束策略可以在大型、未注释的数据集上进行训练，从而为现实世界的设置提供良好的泛化。然而，特别是在基于视觉的环境中，指定目标需要图像，这就造成了不自然的界面。语言为与机器人的交流提供了一种更方便的方式，但现代方法通常需要昂贵的监督，其形式是带有语言描述的轨迹。我们提出了一个用于机器人导航的系统，LM-Nav，它享受在未注释的大型轨迹数据集上进行培训的好处，同时仍为用户提供高级别的界面。我们没有利用标记的指令跟随数据集，而是表明这样的系统可以完全由预先训练的导航模型(Ving)、图像语言关联(CLIP)和语言建模(GPT-3)构建，而不需要任何微调或语言注释的机器人数据。LM-Nav从指令中提取地标名称，通过图像语言模型将它们放置在世界上，然后通过(仅限视觉的)导航模型到达它们。我们在真实的移动机器人上实例化了LM-Nav，并根据自然语言指令演示了在复杂的室外环境中的长期导航。

**[Paper URL](https://proceedings.mlr.press/v205/shah23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/shah23b/shah23b.pdf)** 

# BusyBot: Learning to Interact, Reason, and Plan in a BusyBoard Environment
**题目:** BusyBot：学会在BusyBoard环境中互动、推理和规划

**作者:** Zeyi Liu, Zhenjia Xu, Shuran Song

**Abstract:** We introduce BusyBoard, a toy-inspired robot learning environment that leverages a diverse set of articulated objects and inter-object functional relations to provide rich visual feedback for robot interactions. Based on this environment, we introduce a learning framework, BusyBot, which allows an agent to jointly acquire three fundamental capabilities (interaction, reasoning, and planning) in an integrated and self-supervised manner. With the rich sensory feedback provided by BusyBoard, BusyBot first learns a policy to efficiently interact with the environment; then with data collected using the policy, BusyBot reasons the inter-object functional relations through a causal discovery network; and finally by combining the learned interaction policy and relation reasoning skill, the agent is able to perform goal-conditioned manipulation tasks. We evaluate BusyBot in both simulated and real-world environments, and validate its generalizability to unseen objects and relations.

**摘要:** 我们引入BusyBoard，这是一个受玩具启发的机器人学习环境，它利用一组不同的关节对象和对象间功能关系，为机器人交互提供丰富的视觉反馈。基于此环境，我们引入了一个学习框架BusyBot，它允许代理以集成和自我监督的方式联合获得三种基本能力（交互、推理和规划）。借助BusyBoard提供的丰富感官反馈，BusyBot首先学习与环境有效交互的策略;然后利用使用该策略收集的数据，BusyBot通过因果发现网络推理对象间功能关系;最后通过结合学习的交互策略和关系推理技能，代理能够执行目标条件操纵任务。我们在模拟和现实世界环境中评估BusyBot，并验证其对不可见对象和关系的概括性。

**[Paper URL](https://proceedings.mlr.press/v205/liu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/liu23c/liu23c.pdf)** 

# NeuralGrasps: Learning Implicit Representations for Grasps of Multiple Robotic Hands
**题目:** NeuralGrasps：学习多个机器人手抓取的隐式表示

**作者:** Ninad Khargonkar, Neil Song, Zesheng Xu, B Prabhakaran, Yu Xiang

**Abstract:** We introduce a neural implicit representation for grasps of objects from multiple robotic hands. Different grasps across multiple robotic hands are encoded into a shared latent space. Each latent vector is learned to decode to the 3D shape of an object and the 3D shape of a robotic hand in a grasping pose in terms of the signed distance functions of the two 3D shapes. In addition, the distance metric in the latent space is learned to preserve the similarity between grasps across different robotic hands, where the similarity of grasps is defined according to contact regions of the robotic hands. This property enables our method to transfer grasps between different grippers including a human hand, and grasp transfer has the potential to share grasping skills between robots and enable robots to learn grasping skills from humans. Furthermore, the encoded signed distance functions of objects and grasps in our implicit representation can be used for 6D object pose estimation with grasping contact optimization from partial point clouds, which enables robotic grasping in the real world.

**摘要:** 我们介绍了一种神经隐式表示法，用于从多个机械手抓取物体。多个机械手之间的不同抓取被编码到共享的潜在空间中。根据两个3D形状的符号距离函数，学习每个潜在向量以解码为物体的3D形状和抓取姿势下的机械手的3D形状。此外，学习潜在空间中的距离度量以保持不同机械手之间的抓取之间的相似性，其中抓取的相似性是根据机械手的接触区域来定义的。这一特性使得我们的方法能够在包括人类手在内的不同抓取器之间转移抓取，并且抓取转移具有在机器人之间共享抓取技能的潜力，并使机器人能够从人类那里学习抓取技能。此外，在我们的隐式表示中，物体和抓取的编码符号距离函数可以用于从部分点云中进行抓取接触优化的6D物体姿态估计，从而使机器人能够在真实世界中抓取。

**[Paper URL](https://proceedings.mlr.press/v205/khargonkar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/khargonkar23a/khargonkar23a.pdf)** 

# Frame Mining: a Free Lunch for Learning Robotic Manipulation from 3D Point Clouds
**题目:** 框架挖掘：从3D点云学习机器人操纵的免费午餐

**作者:** Minghua Liu, Xuanlin Li, Zhan Ling, Yangyan Li, Hao Su

**Abstract:** We study how choices of input point cloud coordinate frames impact learning of manipulation skills from 3D point clouds. There exist a variety of coordinate frame choices to normalize captured robot-object-interaction point clouds. We find that different frames have a profound effect on agent learning performance, and the trend is similar across 3D backbone networks. In particular, the end-effector frame and the target-part frame achieve higher training efficiency than the commonly used world frame and robot-base frame in many tasks, intuitively because they provide helpful alignments among point clouds across time steps and thus can simplify visual module learning. Moreover, the well-performing frames vary across tasks, and some tasks may benefit from multiple frame candidates. We thus propose FrameMiners to adaptively select candidate frames and fuse their merits in a task-agnostic manner. Experimentally, FrameMiners achieves on-par or significantly higher performance than the best single-frame version on five fully physical manipulation tasks adapted from ManiSkill and OCRTOC. Without changing existing camera placements or adding extra cameras, point cloud frame mining can serve as a free lunch to improve 3D manipulation learning.

**摘要:** 我们研究了输入点云坐标系的选择对从三维点云中学习操作技能的影响。存在多种坐标框架选择来归一化捕获的机器人-对象-交互点云。我们发现，不同的帧对代理的学习性能有深刻的影响，并且在3D骨干网络中的趋势是相似的。特别是，在许多任务中，末端效应器帧和目标部分帧比通常使用的世界帧和机器人基帧获得了更高的训练效率，因为它们提供了跨时间步长的点云之间的有用对齐，从而可以简化视觉模块的学习。此外，执行良好的帧因任务而异，并且某些任务可能受益于多个候选帧。因此，我们建议FrameMiners自适应地选择候选帧，并以与任务无关的方式融合它们的优点。在实验中，FrameMiners在从ManiSkill和OCRTOC改编的五个完全物理操作任务中实现了与最佳单帧版本相当或显著更高的性能。无需更改现有的摄像头位置或添加额外的摄像头，点云帧挖掘可以充当免费午餐，以改进3D操作学习。

**[Paper URL](https://proceedings.mlr.press/v205/liu23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/liu23d/liu23d.pdf)** 

# SurroundDepth: Entangling Surrounding Views for Self-Supervised Multi-Camera Depth Estimation
**题目:** SurroundDepth：用于自我监督多摄像机深度估计的纠缠周围视图

**作者:** Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Yongming Rao, Guan Huang, Jiwen Lu, Jie Zhou

**Abstract:** Depth estimation from images serves as the fundamental step of 3D perception for autonomous driving and is an economical alternative to expensive depth sensors like LiDAR. The temporal photometric consistency enables self-supervised depth estimation without labels, further facilitating its application. However, most existing methods predict the depth solely based on each monocular image and ignore the correlations among multiple surrounding cameras, which are typically available for modern self-driving vehicles. In this paper, we propose a SurroundDepth method to incorporate the information from multiple surrounding views to predict depth maps across cameras. Specifically, we employ a joint network to process all the surrounding views and propose a cross-view transformer to effectively fuse the information from multiple views. We apply cross-view self-attention to efficiently enable the global interactions between multi-camera feature maps. Different from self-supervised monocular depth estimation, we are able to predict real-world scales given multi-camera extrinsic matrices. To achieve this goal, we adopt two-frame structure-from-motion to extract scale-aware pseudo depths to pretrain the models. Further, instead of predicting the ego-motion of each individual camera, we estimate a universal ego-motion of the vehicle and transfer it to each view to achieve multi-view consistency. In experiments, our method achieves the state-of-the-art performance on the challenging multi-camera depth estimation datasets DDAD and nuScenes.

**摘要:** 从图像中估计深度是自动驾驶3D感知的基本步骤，是LiDAR等昂贵深度传感器的经济替代方案。时间光度一致性使无标签的自监督深度估计成为可能，进一步方便了其应用。然而，现有的大多数方法仅仅基于每一张单目图像来预测深度，而忽略了多个周围摄像头之间的相关性，这通常适用于现代自动驾驶车辆。在本文中，我们提出了一种结合来自多个周围视图的信息的SuroundDepth方法来预测跨摄像机的深度图。具体地说，我们使用一个联合网络来处理周围的所有视图，并提出了一个交叉视图转换器来有效地融合来自多个视图的信息。我们应用交叉视点自我注意来有效地实现多摄像机特征地图之间的全局交互。与自监督单目深度估计不同，我们能够在给定多摄像机外部矩阵的情况下预测真实世界的尺度。为了达到这一目的，我们采用了两帧运动结构来提取尺度感知的伪深度来对模型进行预训练。此外，我们不是预测每个单独摄像机的自我运动，而是估计车辆的普遍自我运动，并将其传递到每个视角，以实现多视角的一致性。在实验中，我们的方法在具有挑战性的多摄像机深度估计数据集dda和nuScenes上达到了最先进的性能。

**[Paper URL](https://proceedings.mlr.press/v205/wei23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wei23a/wei23a.pdf)** 

# One-Shot Transfer of Affordance Regions? AffCorrs!
**题目:** 平价地区一次性转让？AffCorrs！

**作者:** Denis Hadjivelichkov, Sicelukwanda Zwane, Lourdes Agapito, Marc Peter Deisenroth, Dimitrios Kanoulas

**Abstract:** In this work, we tackle one-shot visual search of object parts.  Given a single reference image of an object with annotated affordance regions, we segment semantically corresponding parts within a target scene.  We propose AffCorrs, an unsupervised model that combines the properties of pre-trained DINO-ViT’s image descriptors and cyclic correspondences.  We use AffCorrs to find corresponding affordances both for intra- and inter-class one-shot part segmentation. This task is more difficult than supervised alternatives, but enables future work such as learning affordances via imitation and assisted teleoperation.

**摘要:** 在这项工作中，我们解决了对象部分的一次性视觉搜索。  给定具有注释示能区域的对象的单个参考图像，我们分割目标场景内的语义对应部分。  我们提出了AffCorrs，这是一种无监督模型，它结合了预训练的DINO-ViT图像描述符和循环对应关系的属性。  我们使用AffCorrs来为类内和类间一次性零件分割找到相应的可供性。这项任务比受监督的替代方案更困难，但可以实现未来的工作，例如通过模仿和辅助远程操作学习可供性。

**[Paper URL](https://proceedings.mlr.press/v205/hadjivelichkov23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/hadjivelichkov23a/hadjivelichkov23a.pdf)** 

# Lidar Line Selection with Spatially-Aware Shapley Value for Cost-Efficient Depth Completion
**题目:** 具有空间感知Shapley值的激光雷达测线选择，以实现经济高效的深度完工

**作者:** Kamil Adamczewski, Christos Sakaridis, Vaishakh Patil, Luc Van Gool

**Abstract:** Lidar is a vital sensor for estimating the depth of a scene. Typical spinning lidars emit pulses arranged in several horizontal lines and the monetary cost of the sensor increases with the number of these lines. In this work, we present the new problem of optimizing the positioning of lidar lines to find the most effective configuration for the depth completion task. We propose a solution to reduce the number of lines while retaining the up-to-the-mark quality of depth completion. Our method consists of two components, (1) line selection based on the marginal contribution of a line computed via the Shapley value and (2) incorporating line position spread to take into account its need to arrive at image-wide depth completion. Spatially-aware Shapley values (SaS) succeed in selecting line subsets that yield a depth accuracy comparable to the full lidar input while using just half of the lines.

**摘要:** 激光雷达是估计场景深度的重要传感器。典型的旋转激光雷达发射排列在几条水平线上的脉冲，传感器的货币成本随着这些线的数量而增加。在这项工作中，我们提出了优化激光雷达线定位的新问题，以找到深度完成任务的最有效配置。我们提出了一种解决方案，可以减少线条数量，同时保持深度完成的最高质量。我们的方法由两个部分组成，（1）基于通过Shapley值计算的线的边际贡献进行线选择;（2）合并线位置扩展以考虑其达到图像范围深度完成的需要。空间感知Shapley值（SaS）成功选择线子集，其深度准确度与完整激光雷达输入相当，同时仅使用一半线。

**[Paper URL](https://proceedings.mlr.press/v205/adamczewski23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/adamczewski23a/adamczewski23a.pdf)** 

# Iterative Interactive Modeling for Knotting Plastic Bags
**题目:** 打结塑料袋的迭代交互建模

**作者:** Chongkai Gao, Zekun Li, Haichuan Gao, Feng Chen

**Abstract:** Deformable object manipulation has great research significance for the robotic community and numerous applications in daily life. In this work, we study how to knot plastic bags that are randomly dropped from the air with a dual-arm robot based on image input. The complex initial configuration and terrible material properties of plastic bags pose challenges to reliable perception and planning. Directly knotting it from random initial states is difficult. To tackle this problem, we propose Iterative Interactive Modeling (IIM) to first adjust the plastic bag to a standing pose with imitation learning to establish a high-confidence keypoint skeleton model, then perform a set of learned motion primitives to knot it. We leverage spatial action maps to accomplish the iterative pick-and-place action and a graph convolutional network to evaluate the adjusted pose during the IIM process. In experiments, we achieve an 85.0% success rate in knotting 4 different plastic bags, including one with no demonstration.

**摘要:** 可变形物体的操纵对于机器人社区和日常生活中的众多应用都具有重要的研究意义。在这项工作中，我们研究了如何使用基于图像输入的双臂机器人来打结从空中随机掉下的塑料袋。塑料袋复杂的初始结构和糟糕的材料特性给可靠的感知和规划带来了挑战。从随机的初始状态直接将其打结是困难的。为了解决这一问题，我们提出了迭代交互建模(IIM)，首先通过模仿学习将塑料袋调整为站立姿势，建立高置信度的关键点骨架模型，然后执行一组学习的运动基元对其进行打结。我们利用空间动作地图来完成迭代的拾取和放置动作，并利用图形卷积网络来评估IIM过程中调整后的姿势。在实验中，我们获得了85.0%的成功率，4个不同的塑料袋，其中一个没有演示。

**[Paper URL](https://proceedings.mlr.press/v205/gao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/gao23a/gao23a.pdf)** 

# Tailoring Visual Object Representations to Human Requirements: A Case Study with a Recycling Robot
**题目:** 根据人类需求定制视觉对象表示：回收机器人的案例研究

**作者:** Debasmita Ghose, Michal Adam Lewkowicz, Kaleb Gezahegn, Julian Lee, Timothy Adamson, Marynel Vazquez, Brian Scassellati

**Abstract:** Robots are well-suited to alleviate the burden of repetitive and tedious manipulation tasks. In many applications though, a robot may be asked to interact with a wide variety of objects, making it hard or even impossible to pre-program visual object classifiers suitable for the task of interest. In this work, we study the problem of learning a classifier for visual objects based on a few examples provided by humans. We frame this problem from the perspective of learning a suitable visual object representation that allows us to distinguish the desired object category from others. Our proposed approach integrates human supervision into the representation learning process by combining contrastive learning with an additional loss function that brings the representations of human examples close to each other in the latent space. Our experiments show that our proposed method performs better than self-supervised and fully supervised learning methods in offline evaluations and can also be used in real-time by a robot in a simplified recycling domain, where recycling streams contain a variety of objects.

**摘要:** 机器人非常适合减轻重复性和繁琐的操作任务的负担。然而，在许多应用中，机器人可能被要求与各种各样的对象交互，这使得预编程适合感兴趣的任务的视觉对象分类器变得困难，甚至不可能。在这项工作中，我们研究了基于人类提供的几个例子的视觉对象的分类器的学习问题。我们从学习适当的视觉对象表示的角度来框架这个问题，该视觉对象表示允许我们将期望的对象类别与其他对象类别区分开来。我们提出的方法通过将对比学习和额外的损失函数相结合，将人类监督融入到表示学习过程中，该损失函数使人类示例的表示在潜在空间中彼此接近。实验表明，我们提出的方法在离线评估中的性能优于自监督和全监督学习方法，并且还可以在回收流包含各种对象的简化回收域中被机器人实时使用。

**[Paper URL](https://proceedings.mlr.press/v205/ghose23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ghose23a/ghose23a.pdf)** 

# DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation
**题目:** DexPoint：可推广的点云强化学习，用于模拟到真实的灵巧操纵

**作者:** Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Hao Su, Xiaolong Wang

**Abstract:** We propose a sim-to-real framework for dexterous manipulation which can generalize to new objects of the same category in the real world. The key of our framework is to train the manipulation policy with point cloud inputs and dexterous hands. We propose two new techniques to enable joint learning on multiple objects and sim-to-real generalization: (i) using imagined hand point clouds as augmented inputs; and (ii) designing novel contact-based rewards. We empirically evaluate our method using an Allegro Hand to grasp novel objects in both simulation and real world. To the best of our knowledge, this is the first policy learning-based framework that achieves such generalization results with dexterous hands. Our project page is available at https://yzqin.github.io/dexpoint.

**摘要:** 我们提出了一个用于灵巧操纵的从简单到真实的框架，该框架可以推广到现实世界中相同类别的新对象。我们框架的关键是通过点云输入和灵巧的手训练操纵策略。我们提出了两种新技术来实现对多个对象的联合学习和简单到真实的概括：（i）使用想象的手点云作为增强输入;（ii）设计新颖的基于接触的奖励。我们使用Allegro Hand在模拟和现实世界中抓取新对象的方法进行了经验评估。据我们所知，这是第一个通过灵巧的双手实现此类概括结果的基于政策学习的框架。我们的项目页面可访问https://yzqin.github.io/dexpoint。

**[Paper URL](https://proceedings.mlr.press/v205/qin23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/qin23a/qin23a.pdf)** 

# Interpretable Self-Aware Neural Networks for Robust Trajectory Prediction
**题目:** 用于鲁棒轨迹预测的可解释自感知神经网络

**作者:** Masha Itkina, Mykel Kochenderfer

**Abstract:** Although neural networks have seen tremendous success as predictive models in a variety of domains, they can be overly confident in their predictions on out-of-distribution (OOD) data. To be viable for safety-critical applications, like autonomous vehicles, neural networks must accurately estimate their epistemic or model uncertainty, achieving a level of system self-awareness. Techniques for epistemic uncertainty quantification often require OOD data during training or multiple neural network forward passes during inference. These approaches may not be suitable for real-time performance on high-dimensional inputs. Furthermore, existing methods lack interpretability of the estimated uncertainty, which limits their usefulness both to engineers for further system development and to downstream modules in the autonomy stack. We propose the use of evidential deep learning to estimate the epistemic uncertainty over a low-dimensional, interpretable latent space in a trajectory prediction setting. We introduce an interpretable paradigm for trajectory prediction that distributes the uncertainty among the semantic concepts: past agent behavior, road structure, and social context. We validate our approach on real-world autonomous driving data, demonstrating superior performance over state-of-the-art baselines.

**摘要:** 尽管神经网络作为预测模型在各个领域都取得了巨大的成功，但它们可能对自己对分布外(OOD)数据的预测过于自信。为了对自动驾驶汽车等安全关键型应用具有可行性，神经网络必须准确估计它们的认知或模型不确定性，实现一定程度的系统自我意识。认知不确定性量化技术通常在训练过程中需要OOD数据，或者在推理过程中需要多个神经网络前向传递。这些方法可能不适合高维输入的实时性能。此外，现有方法缺乏对估计的不确定性的可解释性，这限制了它们对工程师进一步系统开发和自治堆栈中的下游模块的有用性。我们建议使用证据深度学习来估计轨迹预测环境中低维的、可解释的潜在空间上的认知不确定性。我们引入了一种可解释的轨迹预测范式，该范式将不确定性分布在语义概念之间：过去的代理行为、道路结构和社会背景。我们在真实世界的自动驾驶数据上验证了我们的方法，展示了比最先进的基线更优越的性能。

**[Paper URL](https://proceedings.mlr.press/v205/itkina23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/itkina23a/itkina23a.pdf)** 

# Learning Generalizable Dexterous Manipulation from Human Grasp Affordance
**题目:** 从人类抓持能力中学习可推广的灵巧操纵

**作者:** Yueh-Hua Wu, Jiashun Wang, Xiaolong Wang

**Abstract:** Dexterous manipulation with a multi-finger hand is one of the most challenging problems in robotics. While recent progress in imitation learning has largely improved the sample efficiency compared to Reinforcement Learning, the learned policy can hardly generalize to manipulate novel objects, given limited expert demonstrations. In this paper, we propose to learn dexterous manipulation using large-scale demonstrations with diverse 3D objects in a category, which are generated from a human grasp affordance model. This generalizes the policy to novel object instances within the same category. To train the policy, we propose a novel imitation learning objective jointly with a geometric representation learning objective using our demonstrations. By experimenting with relocating diverse objects in simulation, we show that our approach outperforms baselines with a large margin when manipulating novel objects. We also ablate the importance of 3D object representation learning for manipulation. We include videos and code on the project website: https://kristery.github.io/ILAD/ .

**摘要:** 多指灵巧操作是机器人学中最具挑战性的问题之一。虽然与强化学习相比，模仿学习的最新进展在很大程度上提高了样本的效率，但由于有限的专家演示，学习的策略很难推广到操纵新对象。在这篇文章中，我们建议通过使用一个类别中不同3D对象的大规模演示来学习灵活的操作，这些对象是从人类抓取承受能力模型生成的。这将策略推广到同一类别中的新对象实例。为了训练策略，我们使用我们的演示提出了一个新的模仿学习目标和一个几何表示学习目标。通过在仿真中对不同对象进行重定位的实验，我们证明了在操作新对象时，我们的方法在很大程度上优于基线。我们还消除了3D对象表示学习对操作的重要性。我们在项目网站上提供了视频和代码：https://kristery.github.io/ILAD/。

**[Paper URL](https://proceedings.mlr.press/v205/wu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wu23a/wu23a.pdf)** 

# CADSim: Robust and Scalable in-the-wild 3D Reconstruction for Controllable Sensor Simulation
**题目:** CADS im：用于可控传感器模拟的稳健且可扩展的野外3D重建

**作者:** Jingkang Wang, Sivabalan Manivasagam, Yun Chen, Ze Yang, Ioan Andrei Bârsan, Anqi Joyce Yang, Wei-Chiu Ma, Raquel Urtasun

**Abstract:** Realistic simulation is key to enabling safe and scalable development of self-driving vehicles. A core component is simulating the sensors so that the entire autonomy system can be tested in simulation. Sensor simulation involves modeling traffic participants, such as vehicles, with high-quality appearance and articulated geometry, and rendering them in real-time. The self-driving industry has employed artists to build these assets. However, this is expensive, slow, and may not reflect reality. Instead, reconstructing assets automatically from sensor data collected in the wild would provide a better path to generating a diverse and large set that provides good real-world coverage. However, current reconstruction approaches struggle on in-the-wild sensor data, due to its sparsity and noise. To tackle these issues, we present CADSim which combines part-aware object-class priors via a small set of CAD models with differentiable rendering to automatically reconstruct vehicle geometry, including articulated wheels, with high-quality appearance. Our experiments show our approach recovers more accurate shape from sparse data compared to existing approaches. Importantly, it also trains and renders efficiently. We demonstrate our reconstructed vehicles in a wide range of applications, including accurate testing of autonomy perception systems.

**摘要:** 逼真的模拟是实现自动驾驶汽车安全和可扩展开发的关键。一个核心部件是模拟传感器，以便在仿真中对整个自主系统进行测试。传感器模拟包括对交通参与者(如车辆)进行建模，以高质量的外观和关节几何形状，并实时渲染它们。自动驾驶行业聘请了艺术家来建造这些资产。然而，这是昂贵的、缓慢的，而且可能不能反映现实。相反，从野外收集的传感器数据自动重建资产将提供一种更好的途径，以生成提供良好现实世界覆盖的多样化和大集合。然而，由于传感器数据的稀疏性和噪声，目前的重建方法很难依赖于野外传感器数据。为了解决这些问题，我们提出了CADSim，它通过一小组CAD模型和可区分的渲染来结合零件感知的对象类先验知识，以高质量的外观自动重建包括关节车轮在内的车辆几何形状。实验表明，与现有方法相比，我们的方法能够从稀疏数据中恢复出更准确的形状。重要的是，它还可以高效地训练和呈现。我们在广泛的应用中展示了我们改装的车辆，包括对自主感知系统的准确测试。

**[Paper URL](https://proceedings.mlr.press/v205/wang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wang23b/wang23b.pdf)** 

# Semantic Abstraction: Open-World 3D Scene Understanding from 2D Vision-Language Models
**题目:** 语义抽象：从2D视觉语言模型理解开放世界3D场景

**作者:** Huy Ha, Shuran Song

**Abstract:** We study open-world 3D scene understanding, a family of tasks that require agents to reason about their 3D environment with an open-set vocabulary and out-of-domain visual inputs – a critical skill for robots to operate in the unstructured 3D world. Towards this end, we propose Semantic Abstraction (SemAbs), a framework that equips 2D Vision-Language Models (VLMs) with new 3D spatial capabilities, while maintaining their zero-shot robustness. We achieve this abstraction using relevancy maps extracted from CLIP and learn 3D spatial and geometric reasoning skills on top of those abstractions in a semantic-agnostic manner. We demonstrate the usefulness of SemAbs on two open-world 3D scene understanding tasks: 1) completing partially observed objects and 2) localizing hidden objects from language descriptions. Experiments show that SemAbs can generalize to novel vocabulary, materials/lighting, classes, and domains (i.e., real-world scans) from training on limited 3D synthetic data.

**摘要:** 我们研究开放世界3D场景理解，这是一系列任务，需要代理通过开放式词汇和域外视觉输入来推理其3D环境--这是机器人在非结构化3D世界中操作的关键技能。为此，我们提出了语义抽象（SemAbs），这是一个为2D视觉语言模型（VLM）配备新的3D空间能力的框架，同时保持其零攻击鲁棒性。我们使用从CLIP中提取的相关性地图来实现这一抽象，并以语义不可知的方式在这些抽象之上学习3D空间和几何推理技能。我们展示了SemAbs在两项开放世界3D场景理解任务中的有用性：1）完成部分观察到的对象; 2）根据语言描述定位隐藏对象。实验表明，SemAbs可以推广到新颖的词汇、材料/照明、类别和领域（即现实世界扫描）来自有限的3D合成数据的训练。

**[Paper URL](https://proceedings.mlr.press/v205/ha23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ha23a/ha23a.pdf)** 

# VideoDex: Learning Dexterity from Internet Videos
**题目:** VideoDex：从互联网视频中学习灵活性

**作者:** Kenneth Shaw, Shikhar Bahl, Deepak Pathak

**Abstract:** To build general robotic agents that can operate in many environments, it is often imperative for the robot to collect experience in the real world.  However, this is often not feasible due to safety, time and hardware restrictions.  We thus propose leveraging the next best thing as real world experience: internet videos of humans using their hands.  Visual priors, such as visual features, are often learned from videos, but we believe that more information from videos can be utilized as a stronger prior.  We build a learning algorithm, Videodex, that leverages visual, action and physical priors from human video datasets to guide robot behavior.  These action and physical priors in the neural network dictate the typical human behavior for a particular robot task.   We test our approach on a robot arm and dexterous hand based system and show strong results on many different manipulation tasks, outperforming various state-of-the-art methods. For videos and supplemental material visit our website at https://video-dex.github.io.

**摘要:** 为了构建可以在许多环境中操作的通用机器人代理，机器人通常必须收集现实世界中的经验。然而，由于安全、时间和硬件的限制，这通常是不可行的。因此，我们建议利用第二好的东西作为现实世界的体验：人类使用手的互联网视频。视觉先验，例如视觉特征，通常是从视频中学习的，但我们相信，从视频中获得的更多信息可以作为更强的先验。我们建立了一个学习算法，Videodex，它利用人类视频数据集中的视觉、动作和物理先验来指导机器人的行为。神经网络中的这些动作和物理先验决定了特定机器人任务的典型人类行为。我们在基于机械臂和灵巧手的系统上测试了我们的方法，并在许多不同的操作任务上显示了很好的结果，表现出优于各种最先进的方法。有关视频和补充材料，请访问我们的网站https://video-dex.github.io.

**[Paper URL](https://proceedings.mlr.press/v205/shaw23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/shaw23a/shaw23a.pdf)** 

# Last-Mile Embodied Visual Navigation
**题目:** 最后一英里视觉导航

**作者:** Justin Wasserman, Karmesh Yadav, Girish Chowdhary, Abhinav Gupta, Unnat Jain

**Abstract:** Realistic long-horizon tasks like image-goal navigation involve exploratory and exploitative phases. Assigned with an image of the goal, an embodied agent must explore to discover the goal, i.e., search efficiently using learned priors. Once the goal is discovered, the agent must accurately calibrate the last-mile of navigation to the goal. As with any robust system, switches between exploratory goal discovery and exploitative last-mile navigation enable better recovery from errors. Following these intuitive guide rails, we propose SLING to improve the performance of existing image-goal navigation systems. Entirely complementing prior methods, we focus on last-mile navigation and leverage the underlying geometric structure of the problem with neural descriptors. With simple but effective switches, we can easily connect SLING with heuristic, reinforcement learning, and neural modular policies. On a standardized image-goal navigation benchmark (Hahn et al. 2021), we improve performance across policies, scenes, and episode complexity, raising the state-of-the-art from 45% to 55% success rate. Beyond photorealistic simulation, we conduct real-robot experiments in three physical scenes and find these improvements to transfer well to real environments.

**摘要:** 现实的长期任务，如图像目标导航，包括探索和开发阶段。被分配了目标的图像的具身代理必须探索以发现目标，即使用学习的先验有效地搜索。一旦发现目标，代理必须准确地校准到目标的最后一英里的导航。与任何强大的系统一样，探索性目标发现和开发性最后一英里导航之间的切换可以更好地从错误中恢复。根据这些直观的导轨，我们提出了Sling来提高现有图像目标导航系统的性能。与以前的方法完全互补，我们专注于最后一英里的导航，并利用神经描述符来利用问题的基本几何结构。通过简单但有效的开关，我们可以轻松地将吊索与启发式、强化学习和神经模块策略联系起来。关于标准化的图像目标导航基准(Hahn等人2021)，我们提高了策略、场景和剧集复杂性的性能，将最先进的成功率从45%提高到55%。除了真实感模拟，我们还在三个物理场景中进行了真实机器人实验，发现这些改进可以很好地移植到真实环境中。

**[Paper URL](https://proceedings.mlr.press/v205/wasserman23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wasserman23a/wasserman23a.pdf)** 

# Learning Preconditions of Hybrid Force-Velocity Controllers for Contact-Rich Manipulation
**题目:** 用于富接触操纵的混合力速度控制器的学习先决条件

**作者:** Jacky Liang, Xianyi Cheng, Oliver Kroemer

**Abstract:** Robots need to manipulate objects in constrained environments like shelves and cabinets when assisting humans in everyday settings like homes and offices. These constraints make manipulation difficult by reducing grasp accessibility, so robots need to use non-prehensile strategies that leverage object-environment contacts to perform manipulation tasks. To tackle the challenge of planning and controlling contact-rich behaviors in such settings, this work uses Hybrid Force-Velocity Controllers (HFVCs) as the skill representation and plans skill sequences with learned preconditions. While HFVCs naturally enable robust and compliant contact-rich behaviors, solvers that synthesize them have traditionally relied on precise object models and closed-loop feedback on object pose, which are difficult to obtain in constrained environments due to occlusions. We first relax HFVCs’ need for precise models and feedback with our HFVC synthesis framework, then learn a point-cloud-based precondition function to classify where HFVC executions will still be successful despite modeling inaccuracies. Finally, we use the learned precondition in a search-based task planner to complete contact-rich manipulation tasks in a shelf domain. Our method achieves a task success rate of $73.2%$, outperforming the $51.5%$ achieved by a baseline without the learned precondition. While the precondition function is trained in simulation, it can also transfer to a real-world setup without further fine-tuning. See supplementary materials and videos at https://sites.google.com/view/constrained-manipulation/.

**摘要:** 机器人在家庭和办公室等日常环境中帮助人类时，需要在货架和橱柜等受限环境中操纵物体。这些限制降低了抓取的可及性，从而使操作变得困难，因此机器人需要使用非抓取策略，利用对象与环境的接触来执行操作任务。为了应对在这样的环境中规划和控制丰富接触行为的挑战，本工作使用混合力-速度控制器(HFVC)作为技能表示，并使用学习的前提条件规划技能序列。虽然HFVC自然能够实现健壮且符合要求的接触丰富行为，但合成它们的求解器传统上依赖于精确的对象模型和对象姿态的闭环反馈，而在受限的环境中，由于遮挡，很难获得这些反馈。我们首先通过我们的HFVC合成框架放松了HFVC对精确模型和反馈的需求，然后学习基于点云的前提函数来分类尽管建模不准确，HFVC仍将在哪里成功执行。最后，我们在基于搜索的任务规划器中使用学习到的前提条件来完成架域中的联系人丰富的操作任务。我们的方法获得了$73.2%$的任务成功率，超过了没有学习前提条件的基线获得的$51.5%$的成功率。虽然前置条件函数是在模拟中训练的，但它也可以转换到真实世界的设置，而不需要进一步的微调。请访问https://sites.google.com/view/constrained-manipulation/.查看补充材料和视频

**[Paper URL](https://proceedings.mlr.press/v205/liang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/liang23a/liang23a.pdf)** 

# Cross-Domain Transfer via Semantic Skill Imitation
**题目:** 通过语义技能模仿实现跨领域转移

**作者:** Karl Pertsch, Ruta Desai, Vikash Kumar, Franziska Meier, Joseph J Lim, Dhruv Batra, Akshara Rai

**Abstract:** We propose an approach for semantic imitation, which uses demonstrations from a source domain, e.g. human videos, to accelerate reinforcement learning (RL) in a different target domain, e.g. a robotic manipulator in a simulated kitchen. Instead of imitating low-level actions like joint velocities, our approach imitates the sequence of demonstrated semantic skills like "opening the microwave" or "turning on the stove". This allows us to transfer demonstrations across environments (e.g. real-world to simulated kitchen) and agent embodiments (e.g. bimanual human demonstration to robotic arm).  We evaluate on three challenging cross-domain learning problems and match the performance of demonstration-accelerated RL approaches that require in-domain demonstrations. In a simulated kitchen environment, our approach learns long-horizon robot manipulation tasks, using less than 3 minutes of human video demonstrations from a real-world kitchen. This enables scaling robot learning via the reuse of demonstrations, e.g. collected as human videos, for learning in any number of target domains.

**摘要:** 我们提出了一种语义模拟方法，它使用来自源域的演示，例如人类视频，来加速不同目标域(例如模拟厨房中的机器人操作手)的强化学习(RL)。我们的方法不是模仿关节速度等低级别动作，而是模仿展示的语义技能序列，如“打开微波炉”或“打开炉子”。这允许我们在环境(例如，从真实世界到模拟厨房)和代理实施例(例如，双人演示到机械臂)之间转移演示。我们对三个具有挑战性的跨域学习问题进行了评估，并与需要域内演示的演示加速RL方法的性能进行了匹配。在模拟的厨房环境中，我们的方法学习长期机器人操作任务，使用不到3分钟的真实厨房中的人类视频演示。这使得能够通过重复使用例如作为人类视频收集的演示来扩展机器人学习，以便在任意数量的目标领域中进行学习。

**[Paper URL](https://proceedings.mlr.press/v205/pertsch23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/pertsch23a/pertsch23a.pdf)** 

# Learning Neuro-Symbolic Skills for Bilevel Planning
**题目:** 学习二层规划的神经符号技能

**作者:** Tom Silver, Ashay Athalye, Joshua B. Tenenbaum, Tomás Lozano-Pérez, Leslie Pack Kaelbling

**Abstract:** Decision-making is challenging in robotics environments with continuous object-centric states, continuous actions, long horizons, and sparse feedback. Hierarchical approaches, such as task and motion planning (TAMP), address these challenges by decomposing decision-making into two or more levels of abstraction. In a setting where demonstrations and symbolic predicates are given, prior work has shown how to learn symbolic operators and neural samplers for TAMP with manually designed parameterized policies. Our main contribution is a method for learning parameterized polices in combination with operators and samplers. These components are packaged into modular neuro-symbolic skills and sequenced together with search-then-sample TAMP to solve new tasks. In experiments in four robotics domains, we show that our approach — bilevel planning with neuro-symbolic skills — can solve a wide range of tasks with varying initial states, goals, and objects, outperforming six baselines and ablations.

**摘要:** 在具有连续以对象为中心的状态、连续动作、长视野和稀疏反馈的机器人环境中，决策具有挑战性。任务和运动规划（TAMP）等分层方法通过将决策分解为两个或更多抽象级别来解决这些挑战。在给出演示和符号断言的环境中，之前的工作展示了如何通过手动设计的参数化策略学习TAMP的符号运算符和神经采样器。我们的主要贡献是一种结合操作员和采样器学习参数化策略的方法。这些组件被打包成模块化神经符号技能，并与搜索然后采样TAMP一起排序以解决新任务。在四个机器人领域的实验中，我们表明，我们的方法--具有神经符号技能的两层规划--可以解决具有不同初始状态、目标和对象的广泛任务，优于六个基线和消融。

**[Paper URL](https://proceedings.mlr.press/v205/silver23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/silver23a/silver23a.pdf)** 

# MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare
**题目:** MegaPose：通过渲染和比较对新颖物体进行6D姿势估计

**作者:** Yann Labbé, Lucas Manuelli, Arsalan Mousavian, Stephen Tyree, Stan Birchfield, Jonathan Tremblay, Justin Carpentier, Mathieu Aubry, Dieter Fox, Josef Sivic

**Abstract:** We introduce MegaPose, a method to estimate the 6D pose of novel objects, that is, objects unseen during training. At inference time, the method only assumes knowledge of (i) a region of interest displaying the object in the image and (ii) a CAD model of the observed object. The contributions of this work are threefold. First, we present a 6D pose refiner based on a render&compare strategy which can be applied to novel objects. The shape and coordinate system of the novel object are provided as inputs to the network by rendering multiple synthetic views of the object’s CAD model. Second, we introduce a novel approach for coarse pose estimation which leverages a network trained to classify whether the pose error between a synthetic rendering and an observed image of the same object can be corrected by the refiner. Third, we introduce a large-scale synthetic dataset of photorealistic images of thousands of objects with diverse visual and shape properties and show that this diversity is crucial to obtain good generalization performance on novel objects. We train our approach on this large synthetic dataset and apply it without retraining to hundreds of novel objects in real images from several pose estimation benchmarks. Our approach achieves state-of-the-art performance on the ModelNet and YCB-Video datasets. An extensive evaluation on the 7 core datasets of the BOP challenge demonstrates that our approach achieves performance competitive with existing approaches that require access to the target objects during training. Code, dataset and trained models are available on the project page: https://megapose6d.github.io/.

**摘要:** 我们介绍了MegaPose，一种估计新物体6D姿态的方法，即训练过程中看不到的物体。在推断时，该方法仅假设(I)在图像中显示对象的感兴趣区域和(Ii)观察对象的CAD模型的知识。这项工作的贡献有三个方面。首先，我们提出了一种适用于新物体的基于渲染和比较策略的六维姿态精化器。通过呈现对象的CAD模型的多个合成视图，将新对象的形状和坐标系作为输入提供给网络。其次，我们提出了一种新的粗略姿态估计方法，该方法利用训练好的网络来分类同一物体的合成绘制和观测图像之间的姿态误差是否可以被精化器校正。第三，我们介绍了一个大规模的合成数据集，其中包含数千个具有不同视觉和形状属性的对象的照片真实感图像，并表明这种多样性对于获得对新对象的良好泛化性能至关重要。我们在这个大型合成数据集上训练我们的方法，并将其应用于来自几个姿势估计基准的数百个真实图像中的新对象。我们的方法在ModelNet和YCB-Video数据集上实现了最先进的性能。对BOP挑战的7个核心数据集的广泛评估表明，我们的方法实现了与现有方法的性能竞争，这些方法需要在训练期间访问目标对象。代码、数据集和经过训练的模型可在项目页面上找到：https://megapose6d.github.io/.

**[Paper URL](https://proceedings.mlr.press/v205/labbe23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf)** 

# Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer
**题目:** 使用可解释传感器融合Transformer提高安全性的自动驾驶

**作者:** Hao Shao, Letian Wang, Ruobing Chen, Hongsheng Li, Yu Liu

**Abstract:** Large-scale deployment of autonomous vehicles has been continually delayed due to safety concerns. On the one hand, comprehensive scene understanding is indispensable, a lack of which would result in vulnerability to rare but complex traffic situations, such as the sudden emergence of unknown objects. However, reasoning from a global context requires access to sensors of multiple types and adequate fusion of multi-modal sensor signals, which is difficult to achieve. On the other hand, the lack of interpretability in learning models also hampers the safety with unverifiable failure causes. In this paper, we propose a safety-enhanced autonomous driving framework, named Interpretable Sensor Fusion Transformer (InterFuser), to fully process and fuse information from multi-modal multi-view sensors for achieving comprehensive scene understanding and adversarial event detection. Besides, intermediate interpretable features are generated from our framework, which provide more semantics and are exploited to better constrain actions to be within the safe sets. We conducted extensive experiments on CARLA benchmarks, where our model outperforms prior methods, ranking the first on the public CARLA Leaderboard.

**摘要:** 出于安全考虑，自动驾驶汽车的大规模部署不断推迟。一方面，全面的场景理解是必不可少的，如果缺乏全面的场景理解，将导致对罕见但复杂的交通情况的脆弱性，例如未知对象的突然出现。然而，从全局角度进行推理需要获得多种类型的传感器，并对多模式传感器信号进行充分融合，这是很难实现的。另一方面，学习模型中缺乏可解释性也阻碍了无法验证失败原因的安全性。本文提出了一种安全增强的自主驾驶框架，称为可解释传感器融合转换器(InterFuser)，用于充分处理和融合来自多模式多视角传感器的信息，以实现全面的场景理解和敌对事件检测。此外，我们的框架还生成了中间可解释特征，这些特征提供了更多的语义，并被用来更好地将操作约束在安全集内。我们在Carla基准上进行了广泛的实验，我们的模型比以前的方法性能更好，在公共Carla排行榜上排名第一。

**[Paper URL](https://proceedings.mlr.press/v205/shao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/shao23a/shao23a.pdf)** 

# A Dual Representation Framework for Robot Learning with Human Guidance
**题目:** 人类引导机器人学习的双重表示框架

**作者:** Ruohan Zhang, Dhruva Bansal, Yilun Hao, Ayano Hiranaka, Jialu Gao, Chen Wang, Roberto Martín-Martín, Li Fei-Fei, Jiajun Wu

**Abstract:** The ability to interactively learn skills from human guidance and adjust behavior according to human preference is crucial to accelerating robot learning. But human guidance is an expensive resource, calling for methods that can learn efficiently. In this work, we argue that learning is more efficient if the agent is equipped with a high-level, symbolic representation. We propose a dual representation framework for robot learning from human guidance. The dual representation used by the robotic agent includes one for learning a sensorimotor control policy, and the other, in the form of a symbolic scene graph, for encoding the task-relevant information that motivates human input. We propose two novel learning algorithms based on this framework for learning from human evaluative feedback and from preference. In five continuous control tasks in simulation and in the real world, we demonstrate that our algorithms lead to significant improvement in task performance and learning speed. Additionally, these algorithms require less human effort and are qualitatively preferred by users.

**摘要:** 从人类的指导中交互地学习技能并根据人类的偏好调整行为的能力是加速机器人学习的关键。但人类的指导是一种昂贵的资源，需要能够有效学习的方法。在这项工作中，我们认为，如果代理配备了高级别的符号表示，学习会更有效。我们提出了一种用于机器人从人类引导学习的双重表示框架。机器人智能体使用的双重表示包括一个用于学习感觉运动控制策略的表示，另一个以符号场景图的形式用于编码激励人类输入的与任务相关的信息。基于这一框架，我们提出了两种新的学习算法，分别用于人类评价反馈学习和偏好学习。在五个连续的控制任务中，在仿真和真实世界中，我们的算法在任务性能和学习速度方面都有显著的改善。此外，这些算法需要较少的人力，并且在质量上受到用户的青睐。

**[Paper URL](https://proceedings.mlr.press/v205/zhang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zhang23a/zhang23a.pdf)** 

# Proactive slip control by learned slip model and trajectory adaptation
**题目:** 通过学习滑动模型和轨迹自适应进行主动滑动控制

**作者:** Kiyanoush Nazari, Willow Mandil, Amir Masoud Ghalamzan Esfahani

**Abstract:** This paper presents a novel control approach to dealing with a slip during robotic manipulative movements. Slip is a major cause of failure in many robotic grasping and manipulation tasks. Existing works use increased gripping forces to avoid/control slip. However, this may not be feasible, e.g., because (i) the robot cannot increase the gripping force– the max gripping force has already applied or (ii) an increased force yields a damaged grasped object, such as soft fruit. Moreover, the robot fixes the gripping force when it forms a stable grasp on the surface of an object, and changing the gripping force during manipulative movements in real-time may not be feasible, e.g., with the Franka robot. Hence, controlling the slip by changing gripping forces is not an effective control policy in many settings. We propose a novel control approach to slip avoidance including a learned action-conditioned slip predictor and a constrained optimizer avoiding an expected slip given the desired robot actions. We show the effectiveness of this receding horizon controller in a series of test cases in real robot experimentation. Our experimental results show our proposed data-driven predictive controller can control slip for objects unseen in training.

**摘要:** 本文提出了一种新的控制方法来处理机器人操作运动中的打滑。在许多机器人抓取和操纵任务中，滑移是失败的主要原因。现有工程使用增加的抓持力来避免/控制打滑。然而，这可能是不可行的，例如，因为(I)机器人不能增加抓取力--最大抓持力已经施加，或者(Ii)增加的力产生损坏的抓取对象，例如软果。此外，机器人在物体表面形成稳定抓取时固定抓取力，并且在操纵运动期间实时改变抓取力可能是不可行的，例如使用Franka机器人。因此，在许多情况下，通过改变抓持力来控制滑移不是一种有效的控制策略。我们提出了一种新的防滑控制方法，包括学习动作条件滑移预测器和约束优化器，以避免给定期望动作的预期滑移。在真实机器人实验中的一系列测试用例中，我们证明了该滚动域控制器的有效性。我们的实验结果表明，我们提出的数据驱动预测控制器可以控制训练中未见过的对象的滑移。

**[Paper URL](https://proceedings.mlr.press/v205/nazari23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/nazari23a/nazari23a.pdf)** 

# Transferring Hierarchical Structures with Dual Meta Imitation Learning
**题目:** 通过二元Meta模仿学习转移分层结构

**作者:** Chongkai Gao, Yizhou Jiang, Feng Chen

**Abstract:** Hierarchical Imitation Learning (HIL) is an effective way for robots to learn sub-skills from long-horizon unsegmented demonstrations. However, the learned hierarchical structure lacks the mechanism to transfer across multi-tasks or to new tasks, which makes them have to learn from scratch when facing a new situation. Transferring and reorganizing modular sub-skills require fast adaptation ability of the whole hierarchical structure. In this work, we propose Dual Meta Imitation Learning (DMIL), a hierarchical meta imitation learning method where the high-level network and sub-skills are iteratively meta-learned with model-agnostic meta-learning. DMIL uses the likelihood of state-action pairs from each sub-skill as the supervision for the high-level network adaptation and uses the adapted high-level network to determine different data set for each sub-skill adaptation. We theoretically prove the convergence of the iterative training process of DMIL and establish the connection between DMIL and Expectation-Maximization algorithm. Empirically, we achieve state-of-the-art few-shot imitation learning performance on the Meta-world benchmark and competitive results on long-horizon tasks in Kitchen environments.

**摘要:** 分层模仿学习(HIL)是机器人从长期不分段的演示中学习子技能的一种有效方法。然而，习得的层次结构缺乏跨多任务或向新任务转移的机制，这使得他们在面对新情况时不得不从头开始学习。组件子技能的迁移和重组需要整个层级结构的快速适应能力。在这项工作中，我们提出了双重元模仿学习(DMIL)，这是一种层次化的元模仿学习方法，其中高层网络和子技能使用模型不可知元学习进行迭代元学习。DMIL使用来自每个子技能的状态-动作对的可能性作为高级网络适应的监督，并使用适应的高级网络来确定每个子技能适应的不同数据集。从理论上证明了DMIL迭代训练过程的收敛性，并建立了DMIL与期望最大化算法之间的联系。在经验上，我们在元世界基准上获得了最先进的少量模仿学习性能，并在厨房环境中的长期任务中获得了具有竞争力的结果。

**[Paper URL](https://proceedings.mlr.press/v205/gao23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/gao23b/gao23b.pdf)** 

# Motion Style Transfer: Modular Low-Rank Adaptation for Deep Motion Forecasting
**题目:** 运动风格转移：用于深度运动预测的模块化低等级适应

**作者:** Parth Kothari, Danya Li, Yuejiang Liu, Alexandre Alahi

**Abstract:** Deep motion forecasting models have achieved great success when trained on a massive amount of data. Yet, they often perform poorly when training data is limited. To address this challenge, we propose a transfer learning approach for efficiently adapting pre-trained forecasting models to new domains, such as unseen agent types and scene contexts. Unlike the conventional fine-tuning approach that updates the whole encoder, our main idea is to reduce the amount of tunable parameters that can precisely account for the target domain-specific motion style. To this end, we introduce two components that exploit our prior knowledge of motion style shifts: (i) a low-rank motion style adapter that projects and adjusts the style features at a low-dimensional bottleneck; and (ii) a modular adapter strategy that disentangles the features of scene context and motion history to facilitate a fine-grained choice of adaptation layers. Through extensive experimentation, we show that our proposed adapter design, coined MoSA, outperforms prior methods on several forecasting benchmarks.

**摘要:** 深度运动预测模型在海量数据上的训练取得了很大的成功。然而，当训练数据有限时，它们的表现往往很差。为了应对这一挑战，我们提出了一种迁移学习方法，用于有效地使预先训练的预测模型适应新的领域，如看不见的代理类型和场景上下文。与更新整个编码器的传统微调方法不同，我们的主要思想是减少可调参数的数量，这些可调参数可以精确地解释目标领域特定的运动风格。为此，我们引入了两个利用我们对运动样式转换的先验知识的组件：(I)在低维瓶颈处投影和调整样式特征的低阶运动样式适配器；(Ii)将场景上下文和运动历史的特征分开的模块化适配器策略，以便于细粒度地选择适配层。通过广泛的实验，我们证明了我们提出的适配器设计，称为MOSA，在几个预测基准上优于以前的方法。

**[Paper URL](https://proceedings.mlr.press/v205/kothari23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/kothari23a/kothari23a.pdf)** 

# Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation
**题目:** 感知者-行动者：机器人操纵的多任务Transformer

**作者:** Mohit Shridhar, Lucas Manuelli, Dieter Fox

**Abstract:** Transformers have revolutionized vision and natural language processing with their ability to scale with large datasets. But in robotic manipulation, data is both limited and expensive. Can manipulation still benefit from Transformers with the right problem formulation? We investigate this question with PerAct, a language-conditioned behavior-cloning agent for multi-task 6-DoF manipulation. PerAct encodes language goals and RGB-D voxel observations with a Perceiver Transformer, and outputs discretized actions by “detecting the next best voxel action”. Unlike frameworks that operate on 2D images, the voxelized 3D observation and action space provides a strong structural prior for efficiently learning 6-DoF actions. With this formulation, we train a single multi-task Transformer for 18 RLBench tasks (with 249 variations) and 7 real-world tasks (with 18 variations) from just a few demonstrations per task. Our results show that PerAct significantly outperforms unstructured image-to-action agents and 3D ConvNet baselines for a wide range of tabletop tasks.

**摘要:** 变形金刚凭借其处理大数据集的能力彻底改变了视觉和自然语言处理。但在机器人操作中，数据既有限又昂贵。操控还能从正确的问题表达方式的变形金刚中受益吗？我们用PerAct来研究这个问题，PerAct是一种用于多任务6-DOF操作的语言条件行为克隆代理。PerAct使用感知器转换器对语言目标和RGB-D体素观测进行编码，并通过“检测下一个最佳体素动作”来输出离散化动作。与在2D图像上操作的框架不同，体素化的3D观察和动作空间为高效学习6-DOF动作提供了强大的结构先验。使用这个公式，我们从每个任务的几个演示中训练了一个多任务转换器，用于18个RLBtch任务(有249个变体)和7个真实任务(有18个变体)。我们的结果显示，在广泛的桌面任务中，PerAct的表现明显优于非结构化图像到动作代理和3D ConvNet基线。

**[Paper URL](https://proceedings.mlr.press/v205/shridhar23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/shridhar23a/shridhar23a.pdf)** 

# Synthesizing Adversarial Visual Scenarios for Model-Based Robotic Control
**题目:** 基于模型的机器人控制的对抗视觉场景合成

**作者:** Shubhankar Agarwal, Sandeep P. Chinchali

**Abstract:** Today’s robots often interface data-driven perception and planning models with classical model-predictive controllers (MPC). Often, such learned perception/planning models produce erroneous waypoint predictions on out-of-distribution (OoD) or even adversarial visual inputs, which increase control cost. However, today’s methods to train robust perception models are largely task-agnostic – they augment a dataset using random image transformations or adversarial examples targeted at the vision model in isolation. As such, they often introduce pixel perturbations that are ultimately benign for control. In contrast to prior work that synthesizes adversarial examples for single-step vision tasks, our key contribution is to synthesize adversarial scenarios tailored to multi-step, model-based control. To do so, we use differentiable MPC methods to calculate the sensitivity of a model-based controller to errors in state estimation. We show that re-training vision models on these adversarial datasets improves control performance on OoD test scenarios by up to 36.2% compared to standard task-agnostic data augmentation. We demonstrate our method on examples of robotic navigation, manipulation in RoboSuite, and control of an autonomous air vehicle.

**摘要:** 今天的机器人经常将数据驱动的感知和规划模型与经典的模型预测控制器(MPC)相结合。通常，这种学习的感知/规划模型在分布外(OOD)甚至对抗性的视觉输入上产生错误的路点预测，这增加了控制成本。然而，今天训练稳健感知模型的方法在很大程度上是与任务无关的-它们使用随机图像变换或孤立地针对视觉模型的对抗性示例来扩大数据集。因此，它们经常引入最终对控制有利的像素扰动。与之前为单步视觉任务合成对抗性示例的工作不同，我们的主要贡献是合成针对多步骤、基于模型的控制而定制的对抗性场景。为此，我们使用可微预测控制方法来计算基于模型的控制器对状态估计误差的灵敏度。我们表明，与标准的任务无关数据增强相比，在这些对抗性数据集上重新训练视觉模型可以提高OOD测试场景中的控制性能高达36.2%。我们在机器人导航、RoboSuite中的操作和自主飞行器控制的例子中演示了我们的方法。

**[Paper URL](https://proceedings.mlr.press/v205/agarwal23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/agarwal23b/agarwal23b.pdf)** 

# CausalAF: Causal Autoregressive Flow for Safety-Critical Driving Scenario Generation
**题目:** Cascycle AF：安全关键驾驶场景生成的因果自回归流

**作者:** Wenhao Ding, Haohong Lin, Bo Li, Ding Zhao

**Abstract:** Generating safety-critical scenarios, which are crucial yet difficult to collect, provides an effective way to evaluate the robustness of autonomous driving systems. However, the diversity of scenarios and efficiency of generation methods are heavily restricted by the rareness and structure of safety-critical scenarios. Therefore, existing generative models that only estimate distributions from observational data are not satisfying to solve this problem. In this paper, we integrate causality as a prior into the scenario generation and propose a flow-based generative framework, Causal Autoregressive Flow (CausalAF). CausalAF encourages the generative model to uncover and follow the causal relationship among generated objects via novel causal masking operations instead of searching the sample only from observational data. By learning the cause-and-effect mechanism of how the generated scenario causes risk situations rather than just learning correlations from data, CausalAF significantly improves learning efficiency. Extensive experiments on three heterogeneous traffic scenarios illustrate that CausalAF requires much fewer optimization resources to effectively generate safety-critical scenarios. We also show that using generated scenarios as additional training samples empirically improves the robustness of autonomous driving algorithms.

**摘要:** 生成安全关键场景是关键的，但很难收集，这为评估自动驾驶系统的健壮性提供了一种有效的方法。然而，安全关键场景的稀缺性和结构严重制约了场景的多样性和生成方法的效率。因此，现有的仅根据观测数据估计分布的生成模型不能很好地解决这一问题。本文将因果关系作为先验引入情景生成，提出了一个基于流的生成框架--因果自回归流(Causalaf)。CausalAF鼓励生成模型通过新颖的因果掩蔽操作来发现和跟踪生成对象之间的因果关系，而不是仅从观测数据中搜索样本。通过学习生成的情景如何导致风险情景的因果机制，而不仅仅是从数据中学习相关性，CausalAF显著提高了学习效率。在三种不同流量场景上的大量实验表明，CausalAF需要更少的优化资源才能有效地生成安全关键场景。我们还表明，使用生成的场景作为额外的训练样本，经验上提高了自动驾驶算法的稳健性。

**[Paper URL](https://proceedings.mlr.press/v205/ding23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ding23a/ding23a.pdf)** 

# Volumetric-based Contact Point Detection for 7-DoF Grasping
**题目:** 基于体积的接触点检测用于7-DoF抓取

**作者:** Junhao Cai, Jingcheng Su, Zida Zhou, Hui Cheng, Qifeng Chen, Michael Y Wang

**Abstract:** In this paper, we propose a novel grasp pipeline based on contact point detection on the truncated signed distance function (TSDF) volume to achieve closed-loop 7-degree-of-freedom (7-DoF) grasping on cluttered environments. The key aspects of our method are that 1) the proposed pipeline exploits the TSDF volume in terms of multi-view fusion, contact-point sampling and evaluation, and collision checking, which provides reliable and collision-free 7-DoF gripper poses with real-time performance; 2) the contact-based pose representation effectively eliminates the ambiguity introduced by the normal-based methods, which provides a more precise and flexible solution. Extensive simulated and real-robot experiments demonstrate that the proposed pipeline can select more antipodal and stable grasp poses and outperforms normal-based baselines in terms of the grasp success rate in both simulated and physical scenarios. Code and data are available at https://github.com/caijunhao/vcpd

**摘要:** 在本文中，我们提出了一种基于截断带符号距离函数（TSDF）体积上接触点检测的新型抓取管道，以在混乱环境中实现闭环7自由度（7-DoF）抓取。我们方法的关键方面是：1）拟议的管道在多视图融合、接触点采样和评估以及碰撞检查方面利用了TSDF体积，从而提供可靠且无碰撞的7-DoF夹持器姿势并具有实时性能; 2）基于接触的姿态表示有效地消除了基于常态的方法引入的模糊性，从而提供了更精确、更灵活的解决方案。广泛的模拟和真实机器人实验表明，拟议的管道可以选择更对足和稳定的抓取姿势，并且在模拟和物理场景中的抓取成功率方面优于基于正常的基线。代码和数据可访问https://github.com/caijunhao/vcpd

**[Paper URL](https://proceedings.mlr.press/v205/cai23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/cai23a/cai23a.pdf)** 

# SE(3)-Equivariant Relational Rearrangement with Neural Descriptor Fields
**题目:** SE（3）-神经描述符场的等变关系重排

**作者:** Anthony Simeonov, Yilun Du, Yen-Chen Lin, Alberto Rodriguez Garcia, Leslie Pack Kaelbling, Tomás Lozano-Pérez, Pulkit Agrawal

**Abstract:** We present a framework for specifying tasks involving spatial relations between objects using only 5-10 demonstrations and then executing such tasks given point cloud observations of a novel pair of objects in arbitrary initial poses. Our approach structures these rearrangement tasks by assigning a consistent local coordinate frame to the task-relevant object parts, localizing the corresponding coordinate frame on unseen object instances, and executing an action that brings these frames into alignment. We propose an optimization method that uses multiple Neural Descriptor Fields (NDFs) and a single annotated 3D keypoint to assign a set of consistent coordinate frames to the task-relevant object parts. We also propose an energy-based learning scheme to model the joint configuration of the objects that satisfies a desired relational task. We validate our pipeline on three multi-object rearrangement tasks in simulation and on a real robot. Results show that our method can infer relative transformations that satisfy the desired relation between novel objects in unseen initial poses using just a few demonstrations.

**摘要:** 我们提出了一个框架，仅使用5-10个演示来指定涉及对象之间的空间关系的任务，然后在给定任意初始姿势的一对新对象的点云观测的情况下执行这些任务。我们的方法通过为与任务相关的对象部分分配一致的局部坐标框架，在不可见的对象实例上定位对应的坐标框架，并执行使这些框架对齐的动作来构造这些重新排列的任务。我们提出了一种优化方法，它使用多个神经描述符场(NDF)和一个带注释的3D关键点来为与任务相关的对象部分分配一组一致的坐标系。我们还提出了一种基于能量的学习方案来模拟满足期望关系任务的对象的联合配置。我们在三个多目标重排任务上和在真实机器人上验证了我们的流水线。结果表明，我们的方法可以推断出满足未知初始姿势下新对象之间期望关系的相对变换，只需几个示例。

**[Paper URL](https://proceedings.mlr.press/v205/simeonov23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/simeonov23a/simeonov23a.pdf)** 

# Learning Multi-Objective Curricula for Robotic Policy Learning
**题目:** 学习机器人政策学习的多目标课程

**作者:** Jikun Kang, Miao Liu, Abhinav Gupta, Christopher Pal, Xue Liu, Jie Fu

**Abstract:** Various automatic curriculum learning (ACL) methods have been proposed to improve the sample efficiency and final performance of robots’ policies learning. They are designed to control how a robotic agent collects data, which is inspired by how humans gradually adapt their learning processes to their capabilities. In this paper, we propose a unified automatic curriculum learning framework to create multi-objective but coherent curricula that are generated by a set of parametric curriculum modules. Each curriculum module is instantiated as a neural network and is responsible for generating a particular curriculum. In order to coordinate those potentially conflicting modules in unified parameter space, we propose a multi-task hyper-net learning framework that uses a single hyper-net to parameterize all those curriculum modules. We evaluate our method on a series of robotic manipulation tasks and demonstrate its superiority over other state-of-the-art ACL methods in terms of sample efficiency and final performance.

**摘要:** 为了提高机器人策略学习的样本效率和最终性能，已经提出了各种自动课程学习方法。它们的设计目的是控制机器人代理收集数据的方式，这是受到人类如何逐渐根据自己的能力调整学习过程的启发。在本文中，我们提出了一个统一的自动课程学习框架，以创建由一组参数课程模块生成的多目标但连贯的课程。每个课程模块被实例化为一个神经网络，并负责生成特定的课程。为了在统一的参数空间中协调那些潜在冲突的模块，我们提出了一个多任务超网学习框架，该框架使用一个超网来对所有课程模块进行参数化。我们在一系列机器人操作任务上对我们的方法进行了评估，并在样本效率和最终性能方面证明了它比其他最先进的ACL方法的优越性。

**[Paper URL](https://proceedings.mlr.press/v205/kang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/kang23a/kang23a.pdf)** 

# Rethinking Sim2Real: Lower Fidelity Simulation Leads to Higher Sim2Real Transfer in Navigation
**题目:** 重新思考Sim2Real：较低的保真模拟导致导航中较高的Sim2Real传输

**作者:** Joanne Truong, Max Rudolph, Naoki Harrison Yokoyama, Sonia Chernova, Dhruv Batra, Akshara Rai

**Abstract:** If we want to train robots in simulation before deploying them in reality, it seems natural and almost self-evident to presume that reducing the sim2real gap involves creating simulators of increasing fidelity (since reality is what it is). We challenge this assumption and present a contrary hypothesis – sim2real transfer of robots may be improved with lower (not higher) fidelity simulation. We conduct a systematic large-scale evaluation of this hypothesis on the problem of visual navigation – in the real world, and on 2 different simulators (Habitat and iGibson) using 3 different robots (A1, AlienGo, Spot). Our results show that, contrary to expectation, adding fidelity does not help with learning; performance is poor due to slow simulation speed (preventing large-scale learning) and overfitting to inaccuracies in simulation physics. Instead, building simple models of the robot motion using real-world data can improve learning and generalization.

**摘要:** 如果我们想在现实中部署机器人之前先在模拟中训练它们，那么假设缩小sim 2 real差距涉及创建保真度不断增加的模拟器（因为现实就是现实），这似乎很自然，而且几乎是不言而喻的。我们挑战了这一假设并提出了一个相反的假设--机器人的sim 2 real转移可以通过较低（而不是较高）的保真度模拟来改善。我们使用3个不同的机器人（A1、AlienGo、Spot）在现实世界中以及2个不同的模拟器（Habitat和iGibson）上对视觉导航问题的这一假设进行了系统的大规模评估。我们的结果表明，与预期相反，增加保真度无助于学习;由于模拟速度慢（阻止大规模学习）和过度适应模拟物理中的不准确性，性能较差。相反，使用现实世界的数据构建简单的机器人运动模型可以提高学习和概括性。

**[Paper URL](https://proceedings.mlr.press/v205/truong23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/truong23a/truong23a.pdf)** 

# Learning Dense Visual Descriptors using Image Augmentations for Robot Manipulation Tasks
**题目:** 使用图像增强来学习机器人操纵任务的密集视觉描述符

**作者:** Christian Graf, David B. Adrian, Joshua Weil, Miroslav Gabriel, Philipp Schillinger, Markus Spies, Heiko Neumann, Andras Gabor Kupcsik

**Abstract:** We propose a self-supervised training approach for learning view-invariant dense visual descriptors using image augmentations. Unlike existing works, which often require complex datasets, such as registered RGBD sequences, we train on an unordered set of RGB images. This allows for learning from a single camera view, e.g., in an existing robotic cell with a fix-mounted camera. We create synthetic views and dense pixel correspondences using data augmentations. We find our descriptors are competitive to the existing methods, despite the simpler data recording and setup requirements. We show that training on synthetic correspondences provides descriptor consistency across a broad range of camera views. We compare against training with geometric correspondence from multiple views and provide ablation studies. We also show a robotic bin-picking experiment using descriptors learned from a fix-mounted camera for defining grasp preferences.

**摘要:** 我们提出了一种自监督训练方法，用于使用图像增强来学习视图不变的密集视觉描述符。与现有作品不同，现有作品通常需要复杂的数据集（例如注册的RGBD序列），我们在一组无序的RB图像上进行训练。这允许从单个相机视图进行学习，例如，在现有的带有固定安装摄像机的机器人单元中。我们使用数据增强创建合成视图和密集像素对应。我们发现我们的描述符与现有方法相比具有竞争力，尽管数据记录和设置要求更简单。我们表明，对合成对应的训练可以在广泛的相机视图中提供描述符一致性。我们与来自多个视图的几何对应的训练进行比较，并提供消融研究。我们还展示了一个机器人捡箱实验，使用从固定安装的摄像机学习的描述符来定义抓取偏好。

**[Paper URL](https://proceedings.mlr.press/v205/graf23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/graf23a/graf23a.pdf)** 

# Proactive Robot Assistance via Spatio-Temporal Object Modeling
**题目:** 通过时空对象建模的主动机器人辅助

**作者:** Maithili Patel, Sonia Chernova

**Abstract:** Proactive robot assistance enables a robot to anticipate and provide for a user’s needs without being explicitly asked. We formulate proactive assistance as the problem of the robot anticipating temporal patterns of object movements associated with everyday user routines, and proactively assisting the user by placing objects to adapt the environment to their needs. We introduce a generative graph neural network to learn a unified spatio-temporal predictive model of object dynamics from temporal sequences of object arrangements. We additionally contribute the Household Object Movements from Everyday Routines (HOMER) dataset, which tracks household objects associated with human activities of daily living across 50+ days for five simulated households. Our model outperforms the leading baseline in predicting object movement, correctly predicting locations for 11.1% more objects and wrongly predicting locations for 11.5% fewer objects used by the human user.

**摘要:** 主动式机器人协助使机器人能够预测并满足用户的需求，而无需明确询问。我们将主动协助定义为机器人预测与日常用户日常生活相关的对象移动的时间模式，并通过放置对象来主动协助用户使环境适应他们的需求。我们引入生成图神经网络，从对象排列的时间序列学习对象动态的统一时空预测模型。我们还提供了日常生活中的家庭对象移动（HOMER）数据集，该数据集跟踪五个模拟家庭在50多天内与人类日常生活活动相关的家庭对象。我们的模型在预测对象移动方面优于领先的基线，正确预测的对象位置多了11.1%，错误预测的对象位置少了11.5%。

**[Paper URL](https://proceedings.mlr.press/v205/patel23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/patel23a/patel23a.pdf)** 

# R3M: A Universal Visual Representation for Robot Manipulation
**题目:** R3 M：机器人操纵的通用视觉表示

**作者:** Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, Abhinav Gupta

**Abstract:** We study how visual representations pre-trained on diverse human video data can enable data-efficient learning of downstream robotic manipulation tasks. Concretely, we pre-train a visual representation using the Ego4D human video dataset using a combination of time-contrastive learning, video-language alignment, and an L1 penalty to encourage sparse and compact representations. The resulting representation, R3M, can be used as a frozen perception module for downstream policy learning. Across a suite of 12 simulated robot manipulation tasks, we find that R3M improves task success by over 20% compared to training from scratch and by over 10% compared to state-of-the-art visual representations like CLIP and MoCo. Furthermore, R3M enables a Franka Emika Panda arm to learn a range of manipulation tasks in a real, cluttered apartment given just 20 demonstrations.

**摘要:** 我们研究在不同人类视频数据上预训练的视觉表示如何能够实现下游机器人操纵任务的数据高效学习。具体来说，我们使用Ego 4D人类视频数据集预训练视觉表示，结合时间对比学习、视频语言对齐和L1罚分，以鼓励稀疏和紧凑的表示。由此产生的表示R3 M可以用作下游政策学习的冻结感知模块。在一系列12个模拟机器人操作任务中，我们发现与从头开始训练相比，R3 M将任务成功率提高了20%以上，与CLIP和MoCo等最先进的视觉表示相比，R3 M将任务成功率提高了10%以上。此外，只需进行20次演示，R3 M就能让Franka Talika Panda手臂在真实、凌乱的公寓中学习一系列操纵任务。

**[Paper URL](https://proceedings.mlr.press/v205/nair23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/nair23a/nair23a.pdf)** 

# Towards Capturing the Temporal Dynamics for Trajectory Prediction: a Coarse-to-Fine Approach
**题目:** 捕捉时间动态进行轨迹预测：从粗到细的方法

**作者:** Xiaosong Jia, Li Chen, Penghao Wu, Jia Zeng, Junchi Yan, Hongyang Li, Yu Qiao

**Abstract:** Trajectory prediction is one of the basic tasks in the autonomous driving field, which aims to predict the future position of other agents around the ego vehicle so that a safe yet efficient driving plan could be generated in the downstream module. Recently, deep learning based methods dominate the field. State-of-the-art (SOTA) methods usually follow an encoder-decoder paradigm. Specifically, the encoder is responsible for extracting information from agents’ history states and HD-Map and providing a representation vector for each agent. Taking these vectors as input, the decoder predicts multi-step future positions for each agent, which is usually accomplished by a single multi-layer perceptron (MLP) to directly output a Tx2 tensor. Though models with adoptation of MLP decoder have dominated the leaderboard of multiple datasets, ‘the elephant in the room is that the temporal correlation among future time-steps is ignored since there is no direct relation among output neurons of a MLP. In this work, we examine this design choice and investigate several ways to apply the temporal inductive bias into the generation of future trajectories on top of a SOTA encoder. We find that simply using autoregressive RNN to generate future positions would lead to significant performance drop even with techniques such as history highway and teacher forcing. Instead, taking scratch trajectories generated by MLP as input, an additional refinement module based on structures with temporal prior such as RNN or 1D-CNN could remarkably boost the accuracy. Furthermore, we examine several objective functions to  emphasize the temporal priors. By the combination of aforementioned techniques to introduce the temporal prior, we improve the top-ranked method’s performance by a large margin and achieve SOTA result on the Waymo Open Motion Challenge.

**摘要:** 轨迹预测是自动驾驶领域的一项基本任务，其目的是预测EGO车辆周围其他智能体的未来位置，以便在下游模块生成安全而高效的驾驶计划。近年来，基于深度学习的方法在该领域占据主导地位。最先进的(SOTA)方法通常遵循编解码器范例。具体来说，编码器负责从代理的历史状态和HD-Map中提取信息，并为每个代理提供表示向量。解码器将这些向量作为输入，预测每个代理的多步未来位置，这通常通过单个多层感知器(MLP)直接输出Tx2张量来完成。虽然采用MLP解码器的模型在多个数据集的排行榜上占据了主导地位，但最大的问题是，由于MLP的输出神经元之间没有直接关系，未来时间步长之间的时间相关性被忽略了。在这项工作中，我们研究了这种设计选择，并研究了几种将时间感应偏差应用于在SOTA编码器上生成未来轨迹的方法。我们发现，简单地使用自回归RNN来生成未来职位会导致显著的性能下降，即使使用历史高速公路和教师强迫等技术也是如此。取而代之的是，以MLP生成的划痕轨迹为输入，基于RNN或1D-CNN等具有时间先验结构的额外求精模块可以显著提高精度。此外，我们还检查了几个目标函数以强调时间先验。通过上述技术的结合引入时间先验，我们大幅提高了排名靠前的方法的性能，并在Waymo公开运动挑战赛上取得了SOTA结果。

**[Paper URL](https://proceedings.mlr.press/v205/jia23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/jia23a/jia23a.pdf)** 

# Human-Robot Commensality: Bite Timing Prediction for Robot-Assisted Feeding in Groups
**题目:** 人与机器人的共同性：机器人辅助群体喂食的咬合时机预测

**作者:** Jan Ondras, Abrar Anwar, Tong Wu, Fanjun Bu, Malte Jung, Jorge Jose Ortiz, Tapomayukh Bhattacharjee

**Abstract:** We develop data-driven models to predict when a robot should feed during social dining scenarios. Being able to eat independently with friends and family is considered one of the most memorable and important activities for people with mobility limitations. While existing robotic systems for feeding people with mobility limitations focus on solitary dining, commensality, the act of eating together, is often the practice of choice. Sharing meals with others introduces the problem of socially appropriate bite timing for a robot, i.e. the appropriate timing for the robot to feed without disrupting the social dynamics of a shared meal. Our key insight is that bite timing strategies that take into account the delicate balance of social cues can lead to seamless interactions during robot-assisted feeding in a social dining scenario. We approach this problem by collecting a Human-Human Commensality Dataset (HHCD) containing 30 groups of three people eating together. We use this dataset to analyze human-human commensality behaviors and develop bite timing prediction models in social dining scenarios. We also transfer these models to human-robot commensality scenarios. Our user studies show that prediction improves when our algorithm uses multimodal social signaling cues between diners to model bite timing. The HHCD dataset, videos of user studies, and code are available at https://emprise.cs.cornell.edu/hrcom/

**摘要:** 我们开发数据驱动模型来预测机器人在社交用餐场景中何时应该进食。对于行动不便的人来说，能够与朋友和家人单独用餐被认为是最令人难忘和最重要的活动之一。虽然现有的为行动不便的人提供食物的机器人系统专注于单独用餐，但集体就餐，即一起吃饭的行为，往往是一种选择。与他人分享食物给机器人带来了社交上合适的咬合时机的问题，即机器人在不破坏分享食物的社交动态的情况下进食的适当时机。我们的关键见解是，在社交进餐场景中，考虑到社交线索的微妙平衡的咬合时机策略可以在机器人辅助进食期间产生无缝互动。我们通过收集人类-人类共生关系数据集(HHCD)来解决这个问题，该数据集包含30组三个人一起吃饭的数据。我们使用这个数据集来分析人与人之间的共生行为，并在社交就餐场景中开发出咬人时间预测模型。我们还将这些模型转换为人-机器人共生场景。我们的用户研究表明，当我们的算法使用食客之间的多通道社交信号提示来模拟就餐时间时，预测会有所改善。HHCD数据集、用户研究视频和代码可在https://emprise.cs.cornell.edu/hrcom/上获得

**[Paper URL](https://proceedings.mlr.press/v205/ondras23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ondras23a/ondras23a.pdf)** 

# Learning Control Admissibility Models with Graph Neural Networks for Multi-Agent Navigation
**题目:** 多智能体导航的图神经网络学习控制允许性模型

**作者:** Chenning Yu, Hongzhan Yu, Sicun Gao

**Abstract:** Deep reinforcement learning in continuous domains focuses on learning control policies that map states to distributions over actions that ideally concentrate on the optimal choices in each step. In multi-agent navigation problems, the optimal actions depend heavily on the agents’ density. Their interaction patterns grow exponentially with respect to such density, making it hard for learning-based methods to generalize. We propose to switch the learning objectives from predicting the optimal actions to predicting sets of admissible actions, which we call control admissibility models (CAMs), such that they can be easily composed and used for online inference for an arbitrary number of agents. We design CAMs using graph neural networks and develop training methods that optimize the CAMs in the standard model-free setting, with the additional benefit of eliminating the need for reward engineering typically required to balance collision avoidance and goal-reaching requirements. We evaluate the proposed approach in multi-agent navigation environments. We show that the CAM models can be trained in environments with only a few agents and be easily composed for deployment in dense environments with hundreds of agents, achieving better performance than state-of-the-art methods.

**摘要:** 连续域中的深度强化学习专注于学习控制策略，这些策略将状态映射到动作上的分布，理想地集中在每个步骤的最优选择上。在多智能体导航问题中，最优行为在很大程度上依赖于智能体的密度。它们的交互模式相对于这种密度呈指数级增长，这使得基于学习的方法很难推广。我们建议将学习目标从预测最优动作转换为预测允许动作集，我们称之为控制可容许性模型(CAM)，这样它们就可以很容易地组合在一起，并用于任意数量的代理的在线推理。我们使用图形神经网络设计凸轮，并开发在标准无模型设置下优化凸轮的培训方法，同时消除了通常需要平衡碰撞避免和达到目标要求的奖励工程的额外好处。我们在多智能体导航环境中对该方法进行了评估。我们表明，CAM模型可以在只有几个代理的环境中进行训练，并且可以很容易地组合在包含数百个代理的密集环境中部署，获得了比最先进的方法更好的性能。

**[Paper URL](https://proceedings.mlr.press/v205/yu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/yu23a/yu23a.pdf)** 

# Socially-Attentive Policy Optimization in Multi-Agent Self-Driving System
**题目:** 多智能体自动驾驶系统中的社会关注政策优化

**作者:** Zipeng Dai, Tianze Zhou, Kun Shao, David Henry Mguni, Bin Wang, Jianye HAO

**Abstract:** As increasing numbers of autonomous vehicles (AVs) are being deployed, it is important to construct a multi-agent self-driving (MASD) system for navigating traffic flows of AVs. In an MASD system, AVs not only navigate themselves to pursue their own goals, but also interact with each other to prevent congestion or collision, especially in scenarios like intersection or lane merging. Multi-agent reinforcement learning (MARL) provides an appealing alternative to generate safe and efficient actions for multiple AVs. However, current MARL methods are limited to describe scenarios where agents interact in either a cooperative of competitive fashion within one episode. Ordinarily, the agents’ objectives are defined with a global or team reward function, which fail to deal with the dynamic social preferences (SPs) and mixed motives like human drivers in traffic interactions. To this end, we propose a novel MARL method called Socially-Attentive Policy Optimization (SAPO), which incorporates: (a) a self-attention module to select the most interactive traffic participant for each AV, and (b) a social-aware integration mechanism to integrate objectives of interacting AVs by estimating the dynamic social preferences from their observations. SAPO solves the problem of how to improve the safety and efficiency of MASD systems, by enabling AVs to learn socially-compatible behaviors. Simulation experiments show that SAPO can successfully capture and utilize the variation of the SPs of AVs to achieve superior performance, compared with baselines in MASD scenarios.

**摘要:** 随着越来越多的自动驾驶车辆的部署，构建一个多智能体自动驾驶系统来导航自动驾驶车辆的交通流量变得非常重要。在MASD系统中，自动车辆不仅导航以追求自己的目标，而且还相互作用以防止拥堵或碰撞，特别是在交叉路口或车道合并等场景中。多智能体强化学习(MAIL)为多个AVs生成安全有效的动作提供了一种有吸引力的替代方案。然而，目前的Marl方法仅限于描述代理在一集内以竞争方式或合作方式交互的场景。通常，智能体的目标定义为全局或团队奖励函数，无法处理交通交互中的动态社会偏好(SP)和驾驶员等混合动机。为此，我们提出了一种新的MAIL方法，称为社会注意力策略优化(SAPO)，它包括：(A)自我注意模块，为每个AV选择最具互动性的交通参与者；(B)社交感知整合机制，通过估计互动AVs的动态社会偏好来整合他们的目标。SAPO通过使AVS能够学习社会兼容行为，解决了如何提高MASD系统的安全性和效率的问题。仿真实验表明，SAPO能够成功地捕获和利用AVS的SPS的变化，获得比MASD场景中的基线更好的性能。

**[Paper URL](https://proceedings.mlr.press/v205/dai23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/dai23a/dai23a.pdf)** 

# Reciprocal MIND MELD: Improving Learning From Demonstration via Personalized, Reciprocal Teaching
**题目:** 互惠MIND MELD：通过个性化、互惠教学改善演示学习

**作者:** Mariah L Schrum, Erin Hedlund-Botti, Matthew Gombolay

**Abstract:** Endowing robots with the ability to learn novel tasks via demonstrations will increase the accessibility of robots for non-expert, non-roboticists. However, research has shown that humans can be poor teachers, making it difficult for robots to effectively learn from humans. If the robot could instruct humans how to provide better demonstrations, then humans might be able to effectively teach a broader range of novel, out-of-distribution tasks. In this work, we introduce Reciprocal MIND MELD, a framework in which the robot learns the way in which a demonstrator is suboptimal and utilizes this information to provide feedback to the demonstrator to improve upon their demonstrations. We additionally develop an Embedding Predictor Network which learns to predict the demonstrator’s suboptimality online without the need for optimal labels. In a series of human-subject experiments in a driving simulator domain, we demonstrate that robotic feedback can effectively improve human demonstrations in two dimensions of suboptimality (p < .001) and that robotic feedback translates into better learning outcomes for a robotic agent on novel tasks (p = .045).

**摘要:** 通过演示赋予机器人学习新任务的能力，将增加非专家、非机器人专家对机器人的可及性。然而，研究表明，人类可能是糟糕的教师，这使得机器人很难有效地向人类学习。如果机器人可以指导人类如何提供更好的演示，那么人类可能能够有效地教授更广泛的新奇、分配外的任务。在这项工作中，我们引入了对等思维融合，这是一个框架，在这个框架中，机器人学习演示者是次优的方式，并利用这些信息向演示者提供反馈，以改进他们的演示。此外，我们还开发了一个嵌入预测网络，它学习在线预测演示者的次优程度，而不需要最优标签。在驾驶模拟器领域的一系列人类受试者实验中，我们证明了机器人反馈可以有效地在二个次优维度上改善人类演示(p<.001)，并且机器人反馈转化为机器人主体在新任务中更好的学习结果(p=.045)。

**[Paper URL](https://proceedings.mlr.press/v205/schrum23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/schrum23a/schrum23a.pdf)** 

# Motion Policy Networks
**题目:** 运动政策网络

**作者:** Adam Fishman, Adithyavairavan Murali, Clemens Eppner, Bryan Peele, Byron Boots, Dieter Fox

**Abstract:** Collision-free motion generation in unknown environments is a core building block for robot manipulation. Generating such motions is challenging due to multiple objectives; not only should the solutions be optimal, the motion generator itself must be fast enough for real-time performance and reliable enough for practical deployment. A wide variety of methods have been proposed ranging from local controllers to global planners, often being combined to offset their shortcomings. We present an end-to-end neural model called Motion Policy Networks (M$\pi$Nets) to generate collision-free, smooth motion from just a single depth camera observation. M$\pi$Nets are trained on over 3 million motion planning problems in more than 500,000 environments. Our experiments show that M$\pi$Nets are significantly faster than global planners while exhibiting the reactivity needed to deal with dynamic scenes. They are 46% better than prior neural planners and more robust than local control policies. Despite being only trained in simulation, M$\pi$Nets transfer well to the real robot with noisy partial point clouds. Videos and code are available at https://mpinets.github.io

**摘要:** 未知环境下的无碰撞运动生成是机器人操作的核心组成部分。由于多个目标，生成这样的运动是具有挑战性的；不仅要有最优的解决方案，而且运动生成器本身必须足够快以满足实时性能，并且对于实际部署来说足够可靠。已经提出了各种各样的方法，从本地控制器到全球规划者，通常是为了弥补它们的缺点而组合在一起的。我们提出了一种端到端的神经模型，称为运动策略网络(M$\pi$Nets)，它可以从单个深度的摄像机观测中生成无碰撞、平滑的运动。M$\pi$Net在500,000多个环境中接受了300多万个运动规划问题的培训。我们的实验表明，M$\pi$Nets比全局规划器更快，同时表现出处理动态场景所需的反应性。它们比以前的神经规划者好46%，比局部控制政策更健壮。尽管只接受了模拟训练，但M$\pi$Net可以很好地转换为带有噪声的局部点云的真实机器人。有关视频和代码，请访问https://mpinets.github.io

**[Paper URL](https://proceedings.mlr.press/v205/fishman23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/fishman23a/fishman23a.pdf)** 

# Decentralized Data Collection for Robotic Fleet Learning: A Game-Theoretic Approach
**题目:** 机器人舰队学习的分散数据收集：游戏理论方法

**作者:** Oguzhan Akcin, Po-han Li, Shubhankar Agarwal, Sandeep P. Chinchali

**Abstract:** Fleets of networked autonomous vehicles (AVs) collect terabytes of sensory data, which is often transmitted to central servers (the “cloud”) for training machine learning (ML) models. Ideally, these fleets should upload all their data, especially from rare operating contexts, in order to train robust ML models. However, this is infeasible due to prohibitive network bandwidth and data labeling costs. Instead, we propose a cooperative data sampling strategy where geo-distributed AVs collaborate to collect a diverse ML training dataset in the cloud. Since the AVs have a shared objective but minimal information about each other’s local data distribution and perception model, we can naturally cast cooperative data collection as an $N$-player mathematical game. We show that our cooperative sampling strategy uses minimal information to converge to a centralized oracle policy with complete information about all AVs. Moreover, we theoretically characterize the performance benefits of our game-theoretic strategy compared to greedy sampling. Finally, we experimentally demonstrate that our method outperforms standard benchmarks by up to $21.9%$ on 4 perception datasets, including for autonomous driving in adverse weather conditions. Crucially, our experimental results on real-world datasets closely align with our theoretical guarantees.

**摘要:** 联网自动驾驶车辆(AV)的车队收集数TB的感觉数据，这些数据通常被传输到中央服务器(“云”)，用于训练机器学习(ML)模型。理想情况下，这些舰队应该上传他们的所有数据，特别是来自罕见的操作环境的数据，以训练稳健的ML模型。然而，由于高昂的网络带宽和数据标签成本，这是不可行的。相反，我们提出了一种协作数据采样策略，其中地理上分布的AVs协作在云中收集不同的ML训练数据集。由于AVS拥有共同的目标，但关于彼此的本地数据分布和感知模型的信息却很少，我们自然可以将合作数据收集视为一场$N$玩家的数学游戏。我们证明了我们的协作抽样策略使用最少的信息来收敛到一个关于所有AVs的完全信息的集中式Oracle策略。此外，我们从理论上刻画了与贪婪抽样相比，我们的博弈论策略的性能优势。最后，我们在4个感知数据集上实验证明，我们的方法比标准基准高出21.9%$，包括在恶劣天气条件下的自动驾驶。至关重要的是，我们在真实世界数据集上的实验结果与我们的理论保证密切一致。

**[Paper URL](https://proceedings.mlr.press/v205/akcin23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/akcin23a/akcin23a.pdf)** 

# CoBEVT: Cooperative Bird’s Eye View Semantic Segmentation with Sparse Transformers
**题目:** CoBEVT：使用稀疏变形器进行合作鸟瞰语义分割

**作者:** Runsheng Xu, Zhengzhong Tu, Hao Xiang, Wei Shao, Bolei Zhou, Jiaqi Ma

**Abstract:** Bird’s eye view (BEV) semantic segmentation plays a crucial role in spatial sensing for autonomous driving. Although recent literature has made significant progress on BEV map understanding, they are all based on single-agent camera-based systems. These solutions sometimes have difficulty handling occlusions or detecting distant objects in complex traffic scenes. Vehicle-to-Vehicle (V2V) communication technologies have enabled autonomous vehicles to share sensing information, dramatically improving the perception performance and range compared to single-agent systems. In this paper, we propose CoBEVT, the first generic multi-agent multi-camera perception framework that can cooperatively generate BEV map predictions. To efficiently fuse camera features from multi-view and multi-agent data in an underlying Transformer architecture, we design a fused axial attention module (FAX), which captures sparsely local and global spatial interactions across views and agents. The extensive experiments on the V2V perception dataset, OPV2V, demonstrate that CoBEVT achieves state-of-the-art performance for cooperative BEV semantic segmentation. Moreover, CoBEVT is shown to be generalizable to other tasks, including 1) BEV segmentation with single-agent multi-camera and 2) 3D object detection with multi-agent LiDAR systems, achieving state-of-the-art performance with real-time inference speed. The code is available at https://github.com/DerrickXuNu/CoBEVT.

**摘要:** 鸟瞰(BEV)语义分割在自动驾驶空间感知中起着至关重要的作用。虽然最近的文献在理解Bev地图方面取得了很大的进展，但它们都是基于单代理摄像机系统的。这些解决方案有时难以处理遮挡或在复杂的交通场景中检测远处的对象。车对车(V2V)通信技术使自动驾驶车辆能够共享传感信息，与单代理系统相比，显著提高了感知性能和覆盖范围。在本文中，我们提出了CoBEVT，这是第一个通用的多智能体多摄像机感知框架，可以协同生成Bev MAP预测。为了在Transformer底层架构中有效地融合多视点和多代理数据中的摄像机特征，我们设计了一个融合的轴向注意力模块(FAX)，该模块能够捕获视点和代理之间稀疏的局部和全局空间交互。在V2V感知数据集OPV2V上的大量实验表明，CoBEVT在协作BEV语义分割方面取得了最好的性能。此外，CoBEVT还可以推广到其他任务，包括1)单智能体多摄像机的Bev分割和2)多智能体LiDAR系统的3D目标检测，获得了最先进的性能和实时推理速度。代码可在https://github.com/DerrickXuNu/CoBEVT.上获得

**[Paper URL](https://proceedings.mlr.press/v205/xu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/xu23a/xu23a.pdf)** 

# Selective Object Rearrangement in Clutter
**题目:** 杂乱中的选择性对象重新排列

**作者:** Bingjie Tang, Gaurav S. Sukhatme

**Abstract:** We propose an image-based, learned method for selective tabletop object rearrangement in clutter using a parallel jaw gripper. Our method consists of three stages: graph-based object sequencing (which object to move), feature-based action selection (whether to push or grasp, and at what position and orientation) and a visual correspondence-based placement policy (where to place a grasped object). Experiments show that this decomposition works well in challenging settings requiring the robot to begin with an initially cluttered scene, selecting only the objects that need to be rearranged while discarding others, and dealing with cases where the goal location for an object is already occupied – making it the first system to address all these concurrently in a purely image-based setting. We also achieve an $\sim$ 8% improvement in task success rate over the previously best reported result that handles both translation and orientation in less restrictive (un-cluttered, non-selective) settings. We demonstrate zero-shot transfer of our system solely trained in simulation to a real robot selectively rearranging up to everyday objects, many unseen during learning, on a crowded tabletop. Videos:https://sites.google.com/view/selective-rearrangement

**摘要:** 我们提出了一种基于图像的学习方法，用于在杂乱的桌面上使用平行抓爪进行选择性桌面对象重排。我们的方法包括三个阶段：基于图形的对象排序(移动哪个对象)，基于特征的动作选择(是推还是抓，以及在什么位置和方向)，以及基于视觉对应的放置策略(将抓取的对象放置在哪里)。实验表明，这种分解在挑战性的环境中工作得很好，要求机器人从最初混乱的场景开始，只选择需要重新排列的对象，而丢弃其他对象，并处理对象的目标位置已经被占用的情况-使其成为第一个在纯粹基于图像的环境中同时处理所有这些问题的系统。我们还在任务成功率方面实现了$\sim$8%的改进，这比之前最好的报告结果更好地处理了限制较少(整洁、非选择性)的环境中的翻译和定向。我们演示了将我们的系统以零射击的方式转移到一个真实的机器人上，该机器人在拥挤的桌面上选择性地重新排列日常物品，其中许多物品在学习过程中是看不到的。Videos:https://sites.google.com/view/selective-rearrangement

**[Paper URL](https://proceedings.mlr.press/v205/tang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/tang23a/tang23a.pdf)** 

# Transformers Are Adaptable Task Planners
**题目:** 变形金刚是适应性的任务规划者

**作者:** Vidhi Jain, Yixin Lin, Eric Undersander, Yonatan Bisk, Akshara Rai

**Abstract:** Every home is different, and every person likes things done in their particular way. Therefore, home robots of the future need to both reason about the sequential nature of day-to-day tasks and generalize to user’s preferences. To this end, we propose a Transformer Task Planner (TTP) that learns high-level actions from demonstrations by leveraging object attribute-based representations. TTP can be pre-trained on multiple preferences and shows generalization to unseen preferences using a single demonstration as a prompt in a simulated dishwasher loading task. Further, we demonstrate real-world dish rearrangement using TTP with a Franka Panda robotic arm, prompted using a single human demonstration.

**摘要:** 每个家庭都是不同的，每个人都喜欢以自己特定的方式做事。因此，未来的家用机器人既需要推理日常任务的顺序性质，又需要根据用户的偏好进行概括。为此，我们提出了一个Transformer TaskPlanner（TTP），它通过利用基于对象属性的表示来从演示中学习高级动作。TTP可以对多个偏好进行预训练，并在模拟洗碗机加载任务中使用单个演示作为提示来显示对未见偏好的概括。此外，我们使用TTP和Franka Panda机械臂演示了现实世界的菜肴重新排列，并由单个人类演示提示。

**[Paper URL](https://proceedings.mlr.press/v205/jain23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/jain23a/jain23a.pdf)** 

# ToolFlowNet: Robotic Manipulation with Tools via Predicting Tool Flow from Point Clouds
**题目:** Tools FlowNet：通过从点云预测工具流来使用工具进行机器人操纵

**作者:** Daniel Seita, Yufei Wang, Sarthak J Shetty, Edward Yao Li, Zackory Erickson, David Held

**Abstract:** Point clouds are a widely available and canonical data modality which convey the 3D geometry of a scene. Despite significant progress in classification and segmentation from point clouds, policy learning from such a modality remains challenging, and most prior works in imitation learning focus on learning policies from images or state information. In this paper, we propose a novel framework for learning policies from point clouds for robotic manipulation with tools. We use a novel neural network, ToolFlowNet, which predicts dense per-point flow on the tool that the robot controls, and then uses the flow to derive the transformation that the robot should execute. We apply this framework to imitation learning of challenging deformable object manipulation tasks with continuous movement of tools, including scooping and pouring, and demonstrate significantly improved performance over baselines which do not use flow. We perform physical scooping experiments with ToolFlowNet and find that we can attain 82% scooping success. See https://sites.google.com/view/point-cloud-policy/home for supplementary material.

**摘要:** 点云是一种广泛可用的规范数据形式，它传达了场景的3D几何图形。尽管在点云分类和分割方面取得了重大进展，但从这种通道学习策略仍然具有挑战性，以前的大多数模仿学习工作都集中在从图像或状态信息学习策略上。在本文中，我们提出了一种新的框架，用于从点云中学习策略，用于机器人使用工具进行操作。我们使用了一种新的神经网络--ToolFlowNet，它预测机器人控制的工具上的密集逐点流，然后使用该流来推导出机器人应该执行的变换。我们将这个框架应用于通过连续移动工具(包括铲和浇注)来挑战可变形对象操纵任务的模拟学习，并显示出比不使用流的基线显著提高的性能。我们使用ToolFlowNet进行了物理挖掘实验，发现我们可以获得82%的挖掘成功率。有关补充材料，请参阅https://sites.google.com/view/point-cloud-policy/home。

**[Paper URL](https://proceedings.mlr.press/v205/seita23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/seita23a/seita23a.pdf)** 

# Fusing Priori and Posteriori Metrics for Automatic Dataset Annotation of Planar Grasping
**题目:** 融合先验和后验概率进行平面抓取自动数据集注释

**作者:** Hao Sha, Lai Qianen, Hongxiang Yu, Rong Xiong, Yue Wang

**Abstract:** Grasp detection based on deep learning has been a research hot spot in recent years.  The performance of grasping detection models relies on high-quality, large-scale grasp datasets. Taking comprehensive consideration of quality, extendability, and annotation cost, metric-based simulation methodology is the most promising way to generate grasp annotation. As experts in grasping, human intuitively tends to make grasp decision based both on priori and posteriori knowledge. Inspired by that, a combination of priori and posteriori grasp metrics is intuitively helpful to improve annotation quality. In this paper, we build a hybrid metric group involving both priori and posteriori metrics and propose a grasp evaluator to merge those metrics to approximate human grasp decision capability. Centered on the evaluator, we have constructed an automatic grasp annotation framework, through which a large-scale, high-quality, low annotation cost planar grasp dataset GMD is automatically generated.

**摘要:** 基于深度学习的抓取检测一直是近年来的研究热点。  抓取检测模型的性能依赖于高质量、大规模抓取数据集。综合考虑质量、可扩展性和注释成本，基于度量的模拟方法是生成抓取注释最有希望的方法。作为抓取专家，人类直觉上倾向于基于先验和后验知识做出抓取决策。受此启发，先验和后验抓取指标的结合直观地有助于提高注释质量。在本文中，我们构建了一个涉及先验和后验指标的混合指标组，并提出了一个抓取评估器来合并这些指标以逼近人类抓取决策能力。以评估器为中心，构建了自动抓取注释框架，通过该框架自动生成大规模、高质量、低注释成本的平面抓取数据集GMD。

**[Paper URL](https://proceedings.mlr.press/v205/sha23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/sha23a/sha23a.pdf)** 

# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
**题目:** PoET：用于单视图、多对象6D姿势估计的姿势估计Transformer

**作者:** Thomas Georg Jantos, Mohamed Amin Hamdad, Wolfgang Granig, Stephan Weiss, Jan Steinbrener

**Abstract:** Accurate 6D object pose estimation is an important task for a variety of robotic applications such as grasping or localization. It is a challenging task due to object symmetries, clutter and occlusion, but it becomes more challenging when additional information, such as depth and 3D models, is not provided. We present a transformer-based approach that takes an RGB image as input and predicts a 6D pose for each object in the image. Besides the image, our network does not require any additional information such as depth maps or 3D object models. First, the image is passed through an object detector to generate feature maps and to detect objects. Then, the feature maps are fed into a transformer with the detected bounding boxes as additional information. Afterwards, the output object queries are processed by a separate translation and rotation head. We achieve state-of-the-art results for RGB-only approaches on the challenging YCB-V dataset. We illustrate the suitability of the resulting model as pose sensor for a 6-DoF state estimation task. Code is available at https://github.com/aau-cns/poet.

**摘要:** 准确的六维物体位姿估计是抓取或定位等各种机器人应用的重要任务。由于物体的对称性、杂乱和遮挡，这是一项具有挑战性的任务，但当不提供其他信息(如深度和3D模型)时，它变得更具挑战性。我们提出了一种基于变换的方法，该方法将一幅RGB图像作为输入，并预测图像中每个对象的6D姿势。除了图像，我们的网络不需要任何额外的信息，如深度图或3D对象模型。首先，图像通过目标检测器以生成特征地图并检测目标。然后，将特征地图以检测到的包围盒作为附加信息馈送到变压器中。然后，输出对象查询由单独的平移和旋转头处理。我们在具有挑战性的YCB-V数据集上实现了仅限RGB方法的最先进结果。我们演示了所得到的模型作为姿态传感器用于6-DOF状态估计任务的适用性。代码可在https://github.com/aau-cns/poet.上找到

**[Paper URL](https://proceedings.mlr.press/v205/jantos23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/jantos23a/jantos23a.pdf)** 

# Out-of-Dynamics Imitation Learning from Multimodal Demonstrations
**题目:** 来自多模式演示的非动态模仿学习

**作者:** Yiwen Qiu, Jialong Wu, Zhangjie Cao, Mingsheng Long

**Abstract:** Existing imitation learning works mainly assume that the demonstrator who collects demonstrations shares the same dynamics as the imitator. However, the assumption limits the usage of imitation learning, especially when collecting demonstrations for the imitator is difficult. In this paper, we study out-of-dynamics imitation learning (OOD-IL), which relaxes the assumption to that the demonstrator and the imitator have the same state spaces but could have different action spaces and dynamics. OOD-IL enables imitation learning to utilize demonstrations from a wide range of demonstrators but introduces a new challenge: some demonstrations cannot be achieved by the imitator due to the different dynamics. Prior works try to filter out such demonstrations by feasibility measurements, but ignore the fact that the demonstrations exhibit a multimodal distribution since the different demonstrators may take different policies in different dynamics. We develop a better transferability measurement to tackle this newly-emerged challenge. We firstly design a novel sequence-based contrastive clustering algorithm to cluster demonstrations from the same mode to avoid the mutual interference of demonstrations from different modes, and then learn the transferability of each demonstration with an adversarial-learning based algorithm in each cluster. Experiment results on several MuJoCo environments, a driving environment, and a simulated robot environment show that the proposed transferability measurement more accurately finds and down-weights non-transferable demonstrations and outperforms prior works on the final imitation learning performance. We show the videos of our experiment results on our website.

**摘要:** 现有的模仿学习工作主要假设收集示范的演示者与模仿者具有相同的动力。然而，这一假设限制了模仿学习的使用，特别是在为模仿者收集演示很困难的情况下。本文研究了动力学外模拟学习(OOD-IL)，它放宽了演示者和模仿者具有相同的状态空间，但可以具有不同的动作空间和动力学的假设。OOD-IL使模仿学习能够利用来自广泛示威者的演示，但引入了一个新的挑战：由于动态的不同，模仿者无法实现一些示范。以往的工作试图通过可行性度量来过滤这类示范，但忽略了示范呈现多模式分布的事实，因为不同的示范者可能在不同的动态中采取不同的政策。我们开发了一种更好的可转移性衡量标准来应对这一新出现的挑战。我们首先设计了一种新的基于序列的对比聚类算法来对同一模式的演示进行聚类，以避免不同模式的演示之间的相互干扰，然后在每个聚类中使用基于对抗性学习的算法来学习每个演示的可转移性。在几个MuJoCo环境、一个驾驶环境和一个模拟机器人环境上的实验结果表明，所提出的可转移性度量能够更准确地发现和降低不可转移演示的权重，并且在最终的模仿学习性能上优于已有的工作。我们在网站上展示了我们实验结果的视频。

**[Paper URL](https://proceedings.mlr.press/v205/qiu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/qiu23a/qiu23a.pdf)** 

# Domain Adaptation and Generalization: A Low-Complexity Approach
**题目:** 领域适应和概括：一种低复杂性的方法

**作者:** Joshua Niemeijer, Jörg Peter Schäfer

**Abstract:** Well-performing deep learning methods are essential in today’s perception of robotic systems such as autonomous driving vehicles. Ongoing research is due to the real-life demands for robust deep learning models against numerous domain changes and cheap training processes to avoid costly manual-labeling efforts. These requirements are addressed by unsupervised domain adaptation methods, in particular for synthetic to real-world domain changes. Recent top-performing approaches are hybrids consisting of multiple adaptation technologies and complex training processes.   In contrast, this work proposes EasyAdap, a simple and easy-to-use unsupervised domain adaptation method achieving near state-of-the-art performance on the synthetic to real-world domain change. Our evaluation consists of a comparison to numerous top-performing methods, and it shows the competitiveness and further potential of domain adaptation and domain generalization capabilities of our method. We contribute and focus on an extensive discussion revealing possible reasons for domain generalization capabilities, which is necessary to satisfy real-life application’s demands.

**摘要:** 良好的深度学习方法对于当今人们对自动驾驶汽车等机器人系统的认知至关重要。正在进行的研究是由于现实生活中对稳健的深度学习模型的需求，以应对大量的领域变化和廉价的培训过程，以避免昂贵的手动标记工作。这些要求通过非监督域自适应方法来解决，特别是对于合成的到真实世界的域改变。最近表现最好的方法是由多种适应技术和复杂的培训过程组成的混合方法。相反，这项工作提出了EasyAdap，一种简单易用的无监督领域自适应方法，在合成到现实世界的领域变化上取得了接近最先进的性能。我们的评估包括与许多性能最好的方法的比较，它表明了我们方法的领域适应和领域泛化能力的竞争力和进一步的潜力。我们贡献并集中于广泛的讨论，揭示了域泛化能力的可能原因，这是满足现实生活应用程序需求所必需的。

**[Paper URL](https://proceedings.mlr.press/v205/niemeijer23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/niemeijer23a/niemeijer23a.pdf)** 

# COACH: Cooperative Robot Teaching
**题目:** 教练：机器人合作教学

**作者:** Cunjun Yu, Yiqing Xu, Linfeng Li, David Hsu

**Abstract:** Knowledge and skills can transfer from human teachers to human students. However, such direct transfer is often not scalable for physical tasks, as they require one-to-one interaction, and human teachers are not available in sufficient numbers.  Machine learning enables robots to become experts and play the role of teachers to help in this situation.  In this work, we formalize cooperative robot teaching as a Markov game, consisting of four key elements: the target task, the student model, the teacher model, and the interactive teaching-learning process.  Under a moderate assumption, the Markov game reduces to a partially observable Markov decision process, with an efficient approximate solution. We illustrate our approach on two cooperative tasks, one in a simulated video game and one with a real robot.

**摘要:** 知识和技能可以从人类教师转移到人类学生。然而，这种直接转移通常无法扩展到体力任务，因为它们需要一对一的互动，而且人类教师的数量不够。  机器学习使机器人能够成为专家，并扮演教师的角色，在这种情况下提供帮助。  在这项工作中，我们将机器人合作教学形式化为马尔科夫游戏，由四个关键元素组成：目标任务、学生模型、教师模型和交互式教学过程。  在适度的假设下，Markov博弈简化为部分可观察的Markov决策过程，并具有有效的近似解。我们说明了我们在两项合作任务中的方法，一项是在模拟视频游戏中，另一项是在真实的机器人中。

**[Paper URL](https://proceedings.mlr.press/v205/yu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/yu23b/yu23b.pdf)** 

# TrackletMapper: Ground Surface Segmentation and Mapping from Traffic Participant Trajectories
**题目:** TrackletMapper：根据交通参与者轨迹绘制地面分割和地图

**作者:** Jannik Zürn, Sebastian Weber, Wolfram Burgard

**Abstract:** Robustly classifying ground infrastructure such as roads and street crossings is an essential task for mobile robots operating alongside pedestrians. While many semantic segmentation datasets are available for autonomous vehicles, models trained on such datasets exhibit a large domain gap when deployed on robots operating in pedestrian spaces. Manually annotating images recorded from pedestrian viewpoints is both expensive and time-consuming. To overcome this challenge, we propose \textit{TrackletMapper}, a framework for annotating ground surface types such as sidewalks, roads, and street crossings from object tracklets without requiring human-annotated data. To this end, we project the robot ego-trajectory and the paths of other traffic participants into the ego-view camera images, creating sparse semantic annotations for multiple types of ground surfaces from which a ground segmentation model can be trained. We further show that the model can be self-distilled for additional performance benefits by aggregating a ground surface map and projecting it into the camera images, creating a denser set of training annotations compared to the sparse tracklet annotations. We qualitatively and quantitatively attest our findings on a novel large-scale dataset for mobile robots operating in pedestrian areas. Code and dataset will be made available upon acceptance of the manuscript.

**摘要:** 对道路和十字路口等地面基础设施进行强有力的分类，是移动机器人与行人并肩作战的一项基本任务。虽然许多语义分割数据集可用于自动驾驶车辆，但在这些数据集上训练的模型在部署在行人空间操作的机器人上时显示出很大的领域缺口。手动注释从行人视点记录的图像既昂贵又耗时。为了克服这一挑战，我们提出了一个框架，用于从对象轨迹中注释地面类型，如人行道、道路和街道交叉口，而不需要人工注释的数据。为此，我们将机器人的自我轨迹和其他交通参与者的路径投影到自我视角的摄像机图像中，为多种类型的地面创建稀疏语义标注，从而可以训练地面分割模型。我们进一步表明，该模型可以通过聚合地面地图并将其投影到相机图像中来自我提取以获得额外的性能优势，从而创建比稀疏轨迹注释更密集的训练注释集。我们在一个新的大规模数据集上定性和定量地证明了我们的发现，该数据集用于在步行区操作的移动机器人。代码和数据集将在手稿被接受后提供。

**[Paper URL](https://proceedings.mlr.press/v205/zurn23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zurn23a/zurn23a.pdf)** 

# HUM3DIL: Semi-supervised Multi-modal 3D HumanPose Estimation for Autonomous Driving
**题目:** HUM 3DIL：用于自动驾驶的半监督多模式3D人体姿势估计

**作者:** Andrei Zanfir, Mihai Zanfir, Alex Gorban, Jingwei Ji, Yin Zhou, Dragomir Anguelov, Cristian Sminchisescu

**Abstract:** Autonomous driving is an exciting new industry, posing important research questions. Within the perception module, 3D human pose estimation is an emerging technology, which can enable the autonomous vehicle to perceive and understand the subtle and complex behaviors of pedestrians. While hardware systems and sensors have dramatically improved over the decades – with cars potentially boasting complex LiDAR and vision systems and with a growing expansion of the available body of dedicated datasets for this newly available information – not much work has been done to harness these novel signals for the core problem of 3D human pose estimation. Our method, which we coin HUM3DIL (HUMan 3D from Images and LiDAR), efficiently uses of these complementary signals, in a semi-supervised fashion and outperforms existing methods with a large margin. It is a fast and compact model for onboard deployment. Specifically, we embed LiDAR points into pixel-aligned multi-modal features, which we pass through a sequence of Transformer refinement stages. Quantitative experiments on the Waymo Open Dataset support these claims, where we achieve state-of-the-art results on the task of 3D pose estimation.

**摘要:** 自动驾驶是一个令人兴奋的新行业，提出了重要的研究问题。在感知模块中，三维人体姿态估计是一项新兴的技术，它可以使自主车辆感知和理解行人的微妙和复杂的行为。虽然硬件系统和传感器在过去几十年里有了显著的改进-汽车可能拥有复杂的激光雷达和视觉系统，并且可用于这些新获得的信息的专用数据集的可用主体不断扩大-但在利用这些新信号来解决3D人体姿势估计的核心问题方面，还没有做太多的工作。我们的方法被称为HUM3DIL(Human 3D From Images And LiDAR)，它以半监督的方式有效地利用了这些互补信号，并在很大程度上超过了现有的方法。它是一种快速、紧凑的机型，适用于机载部署。具体地说，我们将LiDAR点嵌入到像素对齐的多模式特征中，并通过一系列Transformer细化阶段进行这些特征。在Waymo Open数据集上的定量实验支持这些说法，其中我们在3D姿势估计任务上获得了最先进的结果。

**[Paper URL](https://proceedings.mlr.press/v205/zanfir23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zanfir23a/zanfir23a.pdf)** 

# Laplace Approximation Based Epistemic Uncertainty Estimation in 3D Object Detection
**题目:** 3D物体检测中基于拉普拉斯逼近的认识不确定性估计

**作者:** Peng Yun, Ming Liu

**Abstract:** Understanding the uncertainty of predictions is a desirable feature for perceptual modules in critical robotic applications. 3D object detectors are neural networks with high-dimensional output space. It suffers from poor calibration in classification and lacks reliable uncertainty estimation in regression. To provide a reliable epistemic uncertainty estimation, we tailor Laplace approximation for 3D object detectors, and propose an Uncertainty Separation and Aggregation pipeline for Bayesian inference. The proposed Laplace-approximation approach can easily convert a deterministic 3D object detector into a Bayesian neural network capable of estimating epistemic uncertainty. The experiment results on the KITTI dataset empirically validate the effectiveness of our proposed methods, and demonstrate that Laplace approximation performs better uncertainty quality than Monte-Carlo Dropout, DeepEnsembles, and deterministic models.

**摘要:** 了解预测的不确定性是关键机器人应用中感知模块的理想功能。3D对象检测器是具有多维输出空间的神经网络。它的分类校准较差，回归时缺乏可靠的不确定性估计。为了提供可靠的认识不确定性估计，我们为3D对象检测器定制了拉普拉斯逼近，并提出了一种用于Bayesian推理的不确定性分离和聚集管道。提出的拉普拉斯逼近方法可以轻松地将确定性3D对象检测器转换为能够估计认识不确定性的Bayesian神经网络。KITTI数据集的实验结果从经验上验证了我们提出的方法的有效性，并证明拉普拉斯逼近比Monte-Carlo Dropout、DeepEnsembles和确定性模型具有更好的不确定性质量。

**[Paper URL](https://proceedings.mlr.press/v205/yun23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/yun23a/yun23a.pdf)** 

# Towards Online 3D Bin Packing: Learning Synergies between Packing and Unpacking via DRL
**题目:** 走向在线3D垃圾箱包装：通过DRL学习包装和拆包之间的协同作用

**作者:** Shuai Song, Shuo Yang, Ran Song, Shilei Chu, yibin Li, Wei Zhang

**Abstract:** There is an emerging research interest in addressing the online 3D bin packing problem (3D-BPP), which has a wide range of applications in logistics industry. However, neither heuristic methods nor those based on deep reinforcement learning (DRL) outperform human packers in real logistics scenarios. One important reason is that humans can make corrections after performing inappropriate packing actions by unpacking incorrectly packed items. Inspired by such an unpacking mechanism, we present a DRL-based packing-and-unpacking network (PUN) to learn the synergies between the two actions for the online 3D-BPP. Experimental results demonstrate that PUN achieves the state-of-the-art performance and the supplementary video shows that the system based on PUN can reliably complete the online 3D bin packing task in the real world.

**摘要:** 解决在线3D垃圾箱装箱问题（3D-BPP）的研究兴趣正在兴起，该问题在物流行业中具有广泛的应用。然而，在现实物流场景中，启发式方法和基于深度强化学习（DRL）的方法都不如人类包装工。一个重要原因是，人类可以在执行不当包装动作后通过拆开包装错误的物品来进行纠正。受这种拆包机制的启发，我们提出了一个基于DRL的打包拆包网络（PUN），以了解在线3D-BPP两个动作之间的协同效应。实验结果表明，PUN达到了最先进的性能，补充视频表明，基于PUN的系统可以可靠地完成现实世界中的在线3D垃圾箱装箱任务。

**[Paper URL](https://proceedings.mlr.press/v205/song23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/song23a/song23a.pdf)** 

# Reinforcement learning with Demonstrations from Mismatched Task under Sparse Reward
**题目:** 稀疏奖励下的不匹配任务演示的强化学习

**作者:** Yanjiang Guo, Jingyue Gao, Zheng Wu, Chengming Shi, Jianyu Chen

**Abstract:** Reinforcement learning often suffer from the sparse reward issue in real-world robotics problems. Learning from demonstration (LfD) is an effective way to eliminate this problem, which leverages collected expert data to aid online learning. Prior works often assume that the learning agent and the expert aim to accomplish the same task, which requires collecting new data for every new task. In this paper, we consider the case where the target task is mismatched from but similar with that of the expert. Such setting can be challenging and we found existing LfD methods may encounter a phenomenon called reward signal backward propagation blockages so that the agent cannot be effectively guided by the demonstrations from mismatched task. We propose conservative reward shaping from demonstration (CRSfD), which shapes the sparse rewards using estimated expert value function. To accelerate learning processes, CRSfD guides the agent to conservatively explore around demonstrations. Experimental results of robot manipulation tasks show that our approach outperforms baseline LfD methods when transferring demonstrations collected in a single task to other different but similar tasks.

**摘要:** 在现实机器人问题中，强化学习常常会受到稀疏奖励问题的困扰。从演示中学习(LFD)是消除这一问题的有效方法，它利用收集的专家数据来帮助在线学习。以往的工作往往假设学习代理和专家的目标是完成相同的任务，这需要为每个新任务收集新的数据。在本文中，我们考虑目标任务与专家任务不匹配但与专家任务相似的情况。这样的设置可能是具有挑战性的，我们发现现有的LFD方法可能会遇到奖励信号反向传播阻塞的现象，从而无法有效地指导来自不匹配任务的演示。提出了基于论证的保守型报酬整形方法(CRSfD)，该方法利用估计的专家价值函数对稀疏报酬进行整形。为了加快学习过程，CRSfD引导代理保守地探索演示。机器人操作任务的实验结果表明，当将单个任务中收集的演示转移到其他不同但相似的任务时，我们的方法优于基线LFD方法。

**[Paper URL](https://proceedings.mlr.press/v205/guo23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/guo23a/guo23a.pdf)** 

# Graph network simulators can learn discontinuous, rigid contact dynamics
**题目:** 图形网络模拟器可以学习不连续、刚性接触动力学

**作者:** Kelsey R Allen, Tatiana Lopez Guevara, Yulia Rubanova, Kim Stachenfeld, Alvaro Sanchez-Gonzalez, Peter Battaglia, Tobias Pfaff

**Abstract:** Recent years have seen a rise in techniques for modeling discontinuous dynamics, such as rigid contact or switching motion modes, using deep learning. A common claim is that deep networks are incapable of accurately modeling rigid-body dynamics without explicit modules for handling contacts, due to the continuous nature of how deep networks are parameterized. Here we investigate this claim with experiments on established real and simulated datasets and show that general-purpose graph network simulators, with no contact-specific assumptions, can learn and predict contact discontinuities. Furthermore, contact dynamics learned by graph network simulators capture real-world cube tossing trajectories more accurately than highly engineered robotics simulators, even when provided with only 8 – 16 trajectories. Overall, this suggests that rigid-body dynamics do not pose a fundamental challenge for deep networks with the appropriate general architecture and parameterization.  Instead, our work opens new directions for considering when deep learning-based models might be preferable to traditional simulation environments for accurately modeling real-world contact dynamics.

**摘要:** 近年来，使用深度学习对不连续动力学建模的技术有所上升，例如刚性接触或切换运动模式。一种常见的说法是，由于深层网络被参数化的连续性质，如果没有用于处理接触的显式模块，深层网络无法准确地模拟刚体动力学。在这里，我们通过在已建立的真实和模拟数据集上的实验来研究这一说法，并表明通用的图形网络模拟器，没有特定于接触的假设，可以学习和预测接触不连续。此外，通过图形网络仿真器学习的接触动力学比高度工程化的机器人仿真器更准确地捕捉真实世界的立方体抛掷轨迹，即使只提供8-16个轨迹。总体而言，这表明刚体动力学不会对具有适当的通用体系结构和参数化的深层网络构成根本挑战。相反，我们的工作为考虑何时基于深度学习的模型可能比传统模拟环境更适合准确建模真实世界的接触动力学开辟了新的方向。

**[Paper URL](https://proceedings.mlr.press/v205/allen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/allen23a/allen23a.pdf)** 

# Particle-Based Score Estimation for State Space Model Learning in Autonomous Driving
**题目:** 自动驾驶状态空间模型学习的基于粒子的得分估计

**作者:** Angad Singh, Omar Makhlouf, Maximilian Igl, Joao Messias, Arnaud Doucet, Shimon Whiteson

**Abstract:** Multi-object state estimation is a fundamental problem for robotic applications where a robot must interact with other moving objects. Typically, other objects’ relevant state features are not directly observable, and must instead be inferred from observations. Particle filtering can perform such inference given approximate transition and observation models. However, these models are often unknown a priori, yielding a difficult parameter estimation problem since observations jointly carry transition and observation noise. In this work, we consider learning maximum-likelihood parameters using particle methods. Recent methods addressing this problem typically differentiate through time in a particle filter, which requires workarounds to the non-differentiable resampling step, that yield biased or high variance gradient estimates. By contrast, we exploit Fisher’s identity to obtain a particle-based approximation of the score function (the gradient of the log likelihood) that yields a low variance estimate while only requiring stepwise differentiation through the transition and observation models. We apply our method to real data collected from autonomous vehicles (AVs) and show that it learns better models than existing techniques and is more stable in training, yielding an effective smoother for tracking the trajectories of vehicles around an AV.

**摘要:** 多目标状态估计是机器人应用中的一个基本问题，机器人必须与其他移动对象进行交互。通常，其他物体的相关状态特征不能直接观察到，而必须从观测中推断出来。粒子滤波可以在给定近似的过渡和观测模型的情况下进行这样的推理。然而，这些模型通常是先验未知的，这就产生了一个困难的参数估计问题，因为观测值同时带有过渡和观测噪声。在这项工作中，我们考虑使用粒子方法学习最大似然参数。最近解决这一问题的方法通常是在粒子过滤器中随时间进行区分，这需要解决到不可微重采样步骤，即产生偏向或高方差梯度估计。相反，我们利用Fisher恒等式来获得分数函数(对数似然的梯度)的基于粒子的近似，该函数产生低方差估计，而只需要通过过渡和观察模型逐步区分。我们将我们的方法应用于从自动驾驶车辆(AVs)收集的真实数据，结果表明，它比现有技术学习了更好的模型，在训练中更稳定，为跟踪自动驾驶车辆周围的车辆轨迹产生了有效的平滑。

**[Paper URL](https://proceedings.mlr.press/v205/singh23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/singh23a/singh23a.pdf)** 

# Learning with Muscles: Benefits for Data-Efficiency and Robustness in Anthropomorphic Tasks
**题目:** 用肌肉学习：拟人任务中数据效率和鲁棒性的好处

**作者:** Isabell Wochner, Pierre Schumacher, Georg Martius, Dieter Büchler, Syn Schmitt, Daniel Haeufle

**Abstract:** Humans are able to outperform robots in terms of robustness, versatility, and learning of new tasks in a wide variety of movements. We hypothesize that highly nonlinear muscle dynamics play a large role in providing inherent stability, which is favorable to learning. While recent advances have been made in applying modern learning techniques to muscle-actuated systems both in simulation as well as in robotics, so far, no detailed analysis has been performed to show the benefits of muscles in this setting. Our study closes this gap by investigating core robotics challenges and comparing the performance of different actuator morphologies in terms of data-efficiency, hyperparameter sensitivity, and robustness.

**摘要:** 人类在鲁棒性、多功能性以及在各种动作中学习新任务方面的表现优于机器人。我们假设高度非线性的肌肉动力学在提供有利于学习的固有稳定性方面发挥着重要作用。虽然最近在模拟和机器人技术中将现代学习技术应用于肌肉驱动系统方面取得了进展，但到目前为止，还没有进行详细的分析来显示肌肉在这种环境中的好处。我们的研究通过调查核心机器人挑战并比较不同致动器形态在数据效率、超参数敏感性和稳健性方面的性能来缩小这一差距。

**[Paper URL](https://proceedings.mlr.press/v205/wochner23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wochner23a/wochner23a.pdf)** 

# Bayesian Reinforcement Learning for Single-Episode Missions in Partially Unknown Environments
**题目:** 部分未知环境中单集任务的Bayesian强化学习

**作者:** Matthew Budd, Paul Duckworth, Nick Hawes, Bruno Lacerda

**Abstract:** We consider planning for mobile robots conducting missions in real-world domains where a priori unknown dynamics affect the robot’s costs and transitions. We study single-episode missions where it is crucial that the robot appropriately trades off exploration and exploitation, such that the learning of the environment dynamics is just enough to effectively complete the mission. Thus, we propose modelling unknown dynamics using Gaussian processes, which provide a principled Bayesian framework for incorporating online observations made by the robot, and using them to predict the dynamics in unexplored areas. We then formulate the problem of mission planning in Markov decision processes under Gaussian process predictions as Bayesian model-based reinforcement learning. This allows us to employ solution techniques that plan more efficiently than previous Gaussian process planning methods are able to. We empirically evaluate the benefits of our formulation in an underwater autonomous vehicle navigation task and robot mission planning in a realistic simulation of a nuclear environment.

**摘要:** 我们考虑规划在真实世界领域执行任务的移动机器人，其中先验未知的动态影响机器人的成本和过渡。我们研究单集任务，其中机器人适当地权衡探索和开发是至关重要的，这样对环境动力学的学习就足以有效地完成任务。因此，我们建议使用高斯过程来建模未知的动力学，它提供了一个原则性的贝叶斯框架来结合机器人进行的在线观察，并使用它们来预测未知区域的动力学。然后将高斯过程预测下的马尔可夫决策过程中的任务规划问题描述为基于贝叶斯模型的强化学习。这使我们能够使用比以前的高斯工艺规划方法更有效地规划的解决方案技术。我们在真实的核环境模拟中，对我们的公式在水下自主航行器导航任务和机器人任务规划中的好处进行了经验性的评估。

**[Paper URL](https://proceedings.mlr.press/v205/budd23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/budd23a/budd23a.pdf)** 

# VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors
**题目:** VIOLA：具有对象提议先验的基于视觉操纵的模仿学习

**作者:** Yifeng Zhu, Abhishek Joshi, Peter Stone, Yuke Zhu

**Abstract:** We introduce VIOLA, an object-centric imitation learning approach to learning closed-loop visuomotor policies for robot manipulation. Our approach constructs object-centric representations based on general object proposals from a pre-trained vision model. VIOLA uses a transformer-based policy to reason over these representations and attend to the task-relevant visual factors for action prediction. Such object-based structural priors improve deep imitation learning algorithm’s robustness against object variations and environmental perturbations. We quantitatively evaluate VIOLA in simulation and on real robots. VIOLA outperforms the state-of-the-art imitation learning methods by 45.8% in success rate. It has also been deployed successfully on a physical robot to solve challenging long-horizon tasks, such as dining table arrangement and coffee making. More videos and model details can be found in supplementary material and the project website:  https://ut-austin-rpl.github.io/VIOLA/.

**摘要:** 我们引入了VIOLA，这是一种以对象为中心的模仿学习方法，用于学习机器人操纵的闭环可视化策略。我们的方法基于来自预先训练的视觉模型的一般对象提议来构建以对象为中心的表示。VIOLA使用基于转换器的政策来推理这些表示，并关注与任务相关的视觉因素来进行动作预测。这种基于对象的结构先验提高了深度模仿学习算法对对象变化和环境扰动的鲁棒性。我们在模拟和真实机器人上定量评估VIOLA。VIOLA的成功率优于最先进的模仿学习方法45.8%。它还成功部署在物理机器人上，以解决具有挑战性的长期任务，例如餐桌布置和咖啡制作。更多视频和模型详细信息请参阅补充材料和项目网站：https://ut-austin-rpl.github.io/VIOLA/。

**[Paper URL](https://proceedings.mlr.press/v205/zhu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zhu23a/zhu23a.pdf)** 

# Learning Riemannian Stable Dynamical Systems via Diffeomorphisms
**题目:** 通过互胚学习Riemann稳定动态系统

**作者:** Jiechao Zhang, Hadi Beik Mohammadi, Leonel Rozo

**Abstract:** Dexterous and autonomous robots should be capable of executing elaborated dynamical motions skillfully. Learning techniques may be leveraged to build models of such dynamic skills. To accomplish this, the learning model needs to encode a stable vector field that resembles the desired motion dynamics. This is challenging as the robot state does not evolve on a Euclidean space, and therefore the stability guarantees and vector field encoding need to account for the geometry arising from, for example, the orientation representation. To tackle this problem, we propose learning Riemannian stable dynamical systems (RSDS) from demonstrations, allowing us to account for different geometric constraints resulting from the dynamical system state representation. Our approach provides Lyapunov-stability guarantees on Riemannian manifolds that are enforced on the desired motion dynamics via diffeomorphisms built on neural manifold ODEs. We show that our Riemannian approach makes it possible to learn stable dynamical systems displaying complicated vector fields on both illustrative examples and real-world manipulation tasks, where Euclidean approximations fail.

**摘要:** 灵巧的自主机器人应该能够熟练地执行精心设计的动态动作。可以利用学习技术来构建这种动态技能的模型。要做到这一点，学习模型需要编码一个稳定的矢量场，该矢量场类似于所需的运动动力学。这是具有挑战性的，因为机器人状态不是在欧几里德空间上进化的，因此稳定性保证和矢量场编码需要考虑到例如由方位表示产生的几何。为了解决这个问题，我们建议从演示中学习黎曼稳定动力系统(RSD)，允许我们考虑由动态系统状态表示产生的不同几何约束。我们的方法在黎曼流形上提供了Lyapunov稳定性保证，这些流形通过建立在神经流形上的微分同胚来强制于期望的运动动力学。我们表明，我们的黎曼方法使学习稳定的动力系统成为可能，无论是在示例性例子上还是在欧几里德近似失败的真实操作任务中，都显示出复杂的矢量场。

**[Paper URL](https://proceedings.mlr.press/v205/zhang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zhang23b/zhang23b.pdf)** 

# Learning Robust Real-World Dexterous Grasping Policies via Implicit Shape Augmentation
**题目:** 通过隐式形状增强学习稳健的现实世界灵巧抓取策略

**作者:** Qiuyu Chen, Karl Van Wyk, Yu-Wei Chao, Wei Yang, Arsalan Mousavian, Abhishek Gupta, Dieter Fox

**Abstract:** Dexterous robotic hands have the capability to interact with a wide variety of household objects. However, learning robust real world grasping policies for arbitrary objects has proven challenging due to the difficulty of generating high quality training data. In this work, we propose a learning system (\emph{ISAGrasp}) for leveraging a small number of human demonstrations to bootstrap the generation of a much larger dataset containing successful grasps on a variety of novel objects.  Our key insight is to use a correspondence-aware implicit generative model to deform object meshes and demonstrated human grasps in order to create a diverse dataset for supervised learning, while maintaining semantic realism. We use this dataset to train a robust grasping policy in simulation which can be deployed in the real world. We demonstrate grasping performance with a four-fingered Allegro hand in both simulation and the real world, and show this method can handle entirely new semantic classes and achieve a 79% success rate on grasping unseen objects in the real world.

**摘要:** 灵巧的机械手能够与各种各样的家用物品互动。然而，由于难以生成高质量的训练数据，学习针对任意对象的稳健的真实世界抓取策略被证明是具有挑战性的。在这项工作中，我们提出了一个学习系统(\emph{ISAGrasp})，用于利用少量的人类演示来引导生成包含对各种新对象的成功抓取的更大的数据集。我们的关键见解是使用一种对应感知的隐式生成模型来变形对象网格和演示人类的抓取，以便在保持语义真实感的同时为监督学习创建一个多样化的数据集。我们使用这个数据集在仿真中训练一种健壮的抓取策略，可以部署在真实世界中。我们在仿真和真实世界中演示了四指快板手的抓取性能，并表明该方法可以处理全新的语义类，对现实世界中看不见的物体的抓取成功率为79%。

**[Paper URL](https://proceedings.mlr.press/v205/chen23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/chen23b/chen23b.pdf)** 

# Learning Visualization Policies of Augmented Reality for Human-Robot Collaboration
**题目:** 人机协作的增强现实可视化策略学习

**作者:** Kishan Dhananjay Chandan, Jack Albertson, Shiqi Zhang

**Abstract:** In human-robot collaboration domains, augmented reality (AR) technologies have enabled people to visualize the state of robots. Current AR-based visualization policies are designed manually, which requires a lot of human efforts and domain knowledge. When too little information is visualized, human users find the AR interface not useful; when too much information is visualized, they find it difficult to process the visualized information. In this paper, we develop an intelligent AR agent that learns visualization policies (what to visualize, when, and how) from demonstrations. We created a Unity-based platform for simulating warehouse environments where human-robot teammates work on collaborative delivery tasks. We have collected a dataset that includes demonstrations of visualizing robots’ current and planned behaviors. Our results from experiments with real human participants show that, compared with competitive baselines from the literature, our learned visualization strategies significantly increase the efficiency of human-robot teams in delivery tasks, while reducing the distraction level of human users.

**摘要:** 在人-机器人协作领域，增强现实(AR)技术使人们能够可视化机器人的状态。目前基于AR的可视化策略都是人工设计的，需要大量的人力和领域知识。当可视化的信息太少时，人类用户发现AR界面没有用处；当可视化的信息太多时，他们发现很难处理可视化的信息。在本文中，我们开发了一个智能AR代理，它从演示中学习可视化策略(可视化什么、何时和如何)。我们创建了一个基于Unity的平台，用于模拟仓库环境，在该环境中，人-机器人团队成员在协作交付任务中工作。我们收集了一个数据集，其中包括可视化机器人当前和计划的行为的演示。我们对真实人类参与者的实验结果表明，与文献中的竞争基线相比，我们学习的可视化策略显著提高了人类-机器人团队在交付任务中的效率，同时降低了人类用户的分心程度。

**[Paper URL](https://proceedings.mlr.press/v205/chandan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/chandan23a/chandan23a.pdf)** 

# Deep Black-Box Reinforcement Learning with Movement Primitives
**题目:** 使用运动基元的深度黑匣子强化学习

**作者:** Fabian Otto, Onur Celik, Hongyi Zhou, Hanna Ziesche, Vien Anh Ngo, Gerhard Neumann

**Abstract:** Episode-based reinforcement learning (ERL) algorithms treat reinforcement learning (RL) as a black-box optimization problem where we learn to select a parameter vector of a controller, often represented as a movement primitive, for a given task descriptor called a context. ERL offers several distinct benefits in comparison to step-based RL. It generates smooth control trajectories, can handle non-Markovian reward definitions, and the resulting exploration in parameter space is well suited for solving sparse reward settings. Yet, the high dimensionality of the movement primitive parameters has so far hampered the effective use of deep RL methods. In this paper, we present a new algorithm for deep ERL. It is based on differentiable trust region layers, a successful on-policy deep RL algorithm. These layers allow us to specify trust regions for the policy update that are solved exactly for each state using convex optimization, which enables policies learning with the high precision required for the ERL. We compare our ERL algorithm to state-of-the-art step-based algorithms in many complex simulated robotic control tasks. In doing so, we investigate different reward formulations - dense, sparse, and non-Markovian. While step-based algorithms perform well only on dense rewards, ERL performs favorably on sparse and non-Markovian rewards. Moreover, our results show that the sparse and the non-Markovian rewards are also often better suited to define the desired behavior, allowing us to obtain considerably higher quality policies compared to step-based RL.

**摘要:** 基于情节的强化学习(ERL)算法将强化学习(RL)视为一个黑箱优化问题，其中我们学习为给定的任务描述符(称为上下文)选择控制器的参数向量，通常表示为运动基元。与基于STEP的RL相比，ERL提供了几个明显的优势。它生成平滑的控制轨迹，可以处理非马尔可夫报酬定义，所产生的参数空间探索非常适合于求解稀疏报酬设置。然而，到目前为止，运动基元参数的高维性阻碍了深度RL方法的有效使用。在本文中，我们提出了一种新的深度ERL算法。它基于可区分信任域层，是一种成功的基于策略的深度RL算法。这些层允许我们为策略更新指定信任域，这些信任域使用凸优化精确地为每个状态求解，这使得策略学习能够以ERL所需的高精度进行。在许多复杂的模拟机器人控制任务中，我们将我们的ERL算法与最先进的基于步骤的算法进行了比较。为此，我们研究了不同的奖励公式--稠密的、稀疏的和非马尔科夫的。虽然基于步骤的算法只在密集奖励上表现良好，但ERL在稀疏和非马尔科夫奖励上表现良好。此外，我们的结果表明，稀疏和非马尔可夫报酬通常也更适合于定义期望的行为，使我们能够获得比基于步骤的RL高得多的质量策略。

**[Paper URL](https://proceedings.mlr.press/v205/otto23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/otto23a/otto23a.pdf)** 

# Discriminator-Guided Model-Based Offline Imitation Learning
**题目:** 鉴别器引导的基于模型的离线模仿学习

**作者:** Wenjia Zhang, Haoran Xu, Haoyi Niu, Peng Cheng, Ming Li, Heming Zhang, Guyue Zhou, Xianyuan Zhan

**Abstract:** Offline imitation learning (IL) is a powerful method to solve decision-making problems from expert demonstrations without reward labels. Existing offline IL methods suffer from severe performance degeneration under limited expert data. Including a learned dynamics model can potentially improve the state-action space coverage of expert data, however, it also faces challenging issues like model approximation/generalization errors and suboptimality of rollout data. In this paper, we propose the Discriminator-guided Model-based offline Imitation Learning (DMIL) framework, which introduces a discriminator to simultaneously distinguish the dynamics correctness and sub-optimality of model rollout data against real expert demonstrations. DMIL adopts a novel cooperative-yet-adversarial learning strategy, which uses the discriminator to guide and couple the learning process of the policy and dynamics model, resulting in improved model performance and robustness. Our framework can also be extended to the case when demonstrations contain a large proportion of suboptimal data. Experimental results show that DMIL and its extension achieve superior performance and robustness compared to state-of-the-art offline IL methods under small datasets.

**摘要:** 离线模仿学习(IL)是解决无奖励标签专家演示决策问题的一种有效方法。现有的离线IL方法在有限的专家数据下存在严重的性能退化问题。包含学习的动力学模型可以潜在地提高专家数据的状态-动作空间覆盖率，然而，它也面临诸如模型近似/泛化误差和推出数据的次优等挑战性问题。在本文中，我们提出了鉴别器引导的基于模型的离线模仿学习(DMIL)框架，它引入了一个鉴别器来同时区分模型推出数据的动力学正确性和次优性与真实的专家演示。DMIL采用了一种新颖的合作-对抗学习策略，该策略使用鉴别器来指导和耦合策略模型和动态模型的学习过程，从而提高了模型的性能和鲁棒性。我们的框架也可以扩展到演示包含较大比例的次优数据的情况。实验结果表明，在小数据集下，DMIL及其扩展在性能和稳健性上优于目前最先进的离线IL方法。

**[Paper URL](https://proceedings.mlr.press/v205/zhang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zhang23c/zhang23c.pdf)** 

# Representation Learning for Object Detection from Unlabeled Point Cloud Sequences
**题目:** 从未标记点云序列中检测对象的表示学习

**作者:** Xiangru Huang, Yue Wang, Vitor Campagnolo Guizilini, Rares Andrei Ambrus, Adrien Gaidon, Justin Solomon

**Abstract:** Although unlabeled 3D data is easy to collect, state-of-the-art machine learning techniques for 3D object detection still rely on difficult-to-obtain manual annotations. To reduce dependence on the expensive and error-prone process of manual labeling, we propose a technique for representation learning from unlabeled LiDAR point cloud sequences. Our key insight is that moving objects can be reliably detected from point cloud sequences without the need for human-labeled 3D bounding boxes. In a single LiDAR frame extracted from a sequence, the set of moving objects provides sufficient supervision for single-frame object detection. By designing appropriate pretext tasks, we learn point cloud features that generalize to both moving and static unseen objects. We apply these features to object detection, achieving strong performance on self-supervised representation learning and unsupervised object detection tasks.

**摘要:** 尽管未标记的3D数据很容易收集，但用于3D对象检测的最先进机器学习技术仍然依赖于难以获取的手动注释。为了减少对昂贵且容易出错的手动标记过程的依赖，我们提出了一种从未标记的LiDART点云序列进行表示学习的技术。我们的主要见解是，可以从点云序列中可靠地检测移动对象，而无需人工标记的3D边界框。在从序列提取的单个LiDART帧中，运动对象集为单帧对象检测提供了充分的监督。通过设计适当的借口任务，我们学习了推广到移动和静态未见对象的点云特征。我们将这些功能应用于对象检测，在自我监督表示学习和无监督对象检测任务中实现了出色的性能。

**[Paper URL](https://proceedings.mlr.press/v205/huang23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/huang23b/huang23b.pdf)** 

# Learning Interpretable BEV Based VIO without Deep Neural Networks
**题目:** 无需深度神经网络即可学习可解释的基于BEV的VIO

**作者:** Zexi Chen, Haozhe Du, Xuecheng XU, Rong Xiong, Yiyi Liao, Yue Wang

**Abstract:** Monocular visual-inertial odometry (VIO) is a critical problem in robotics and autonomous driving. Traditional methods solve this problem based on filtering or optimization. While being fully interpretable, they rely on manual interference and empirical parameter tuning. On the other hand, learning-based approaches allow for end-to-end training but require a large number of training data to learn millions of parameters. However, the non-interpretable and heavy models hinder the generalization ability. In this paper, we propose a fully differentiable, and interpretable, bird-eye-view (BEV) based VIO model for robots with local planar motion that can be trained without deep neural networks. Specifically, we first adopt Unscented Kalman Filter as a differentiable layer to predict the pitch and roll, where the covariance matrices of noise are learned to filter out the noise of the IMU raw data.  Second, the refined pitch and roll are adopted to retrieve a gravity-aligned BEV image of each frame using differentiable camera projection. Finally, a differentiable pose estimator is utilized to estimate the remaining 3 DoF poses between the BEV frames: leading to a 5 DoF pose estimation. Our method allows for learning the covariance matrices end-to-end supervised by the pose estimation loss, demonstrating superior performance to empirical baselines. Experimental results on synthetic and real-world datasets demonstrate that our simple approach is competitive with state-of-the-art methods and generalizes well on unseen scenes.

**摘要:** 单目视觉惯性里程计(VIO)是机器人学和自动驾驶中的一个关键问题。传统的方法基于过滤或优化来解决这一问题。在完全可解释的同时，它们依赖于人工干预和经验参数调整。另一方面，基于学习的方法允许端到端的训练，但需要大量的训练数据来学习数百万个参数。然而，不可解释的模型和繁重的模型阻碍了泛化能力。本文提出了一种完全可微的、可解释的基于鸟瞰(BEV)的VIO模型，该模型适用于具有局部平面运动的机器人，无需深度神经网络即可进行训练。具体地说，我们首先采用Unscented卡尔曼滤波作为预测俯仰和横摇的可微层，学习噪声的协方差矩阵来滤除IMU原始数据中的噪声。其次，采用细化的俯仰和滚动，利用可微摄像机投影恢复每一帧的重力对齐的Bev图像。最后，利用一个可微的姿态估计器来估计BEV帧之间剩余的3个DOF姿态：从而得到5个DOF姿态估计。我们的方法允许端到端地学习由姿态估计损失监督的协方差矩阵，表现出优于经验基线的性能。在人工合成和真实世界数据集上的实验结果表明，我们的简单方法与最先进的方法具有竞争力，并且在未知场景下具有很好的泛化能力。

**[Paper URL](https://proceedings.mlr.press/v205/chen23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/chen23c/chen23c.pdf)** 

# Solving Complex Manipulation Tasks with Model-Assisted Model-Free Reinforcement Learning
**题目:** 利用模型辅助无模型强化学习解决复杂操纵任务

**作者:** Jianshu Hu, Paul Weng

**Abstract:** In this paper, we propose a novel deep reinforcement learning approach for improving the sample efficiency of a model-free actor-critic method by using a learned model to encourage exploration. The basic idea consists in generating artificial transitions with noisy actions, which can be used to update the critic. To counteract the model bias, we introduce a high initialization for the critic and two filters for the artificial transitions. Finally, we evaluate our approach with the TD3 algorithm on different robotic tasks and demonstrate that it achieves a better performance with higher sample efficiency than several other model-based and model-free methods.

**摘要:** 在本文中，我们提出了一种新型的深度强化学习方法，通过使用学习的模型来鼓励探索，提高无模型的行为者-评论家方法的样本效率。基本想法包括用有噪音的动作生成人工过渡，可用于更新批评者。为了抵消模型偏差，我们为评论者引入了高度初始化，并为人为转换引入了两个过滤器。最后，我们在不同的机器人任务上使用TD 3算法评估了我们的方法，并证明它比其他几种基于模型和无模型的方法实现了更好的性能和更高的样本效率。

**[Paper URL](https://proceedings.mlr.press/v205/hu23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/hu23a/hu23a.pdf)** 

# Robustness Certification of Visual Perception Models via Camera Motion Smoothing
**题目:** 通过摄像机运动平滑对视觉感知模型进行稳健性认证

**作者:** Hanjiang Hu, Zuxin Liu, Linyi Li, Jiacheng Zhu, Ding Zhao

**Abstract:** A vast literature shows that the learning-based visual perception model is sensitive to adversarial noises, but few works consider the robustness of robotic perception models under widely-existing camera motion perturbations. To this end, we study the robustness of the visual perception model under camera motion perturbations to investigate the influence of camera motion on robotic perception. Specifically, we propose a motion smoothing technique for arbitrary image classification models, whose robustness under camera motion perturbations could be certified. The proposed robustness certification framework based on camera motion smoothing provides effective and scalable robustness guarantees for visual perception modules so that they are applicable to wide robotic applications. As far as we are aware, this is the first work to provide robustness certification for the deep perception module against camera motions, which improves the trustworthiness of robotic perception. A realistic indoor robotic dataset with a dense point cloud map for the entire room, MetaRoom, is introduced for the challenging certifiable robust perception task. We conduct extensive experiments to validate the certification approach via motion smoothing against camera motion perturbations. Our framework guarantees the certified accuracy of 81.7% against camera translation perturbation along depth direction within -0.1m   0.1m. We also validate the effectiveness of our method on the real-world robot by conducting hardware experiments on the robotic arm with an eye-in-hand camera. The code is available at https://github.com/HanjiangHu/camera-motion-smoothing.

**摘要:** 大量文献表明，基于学习的视觉感知模型对对抗性噪声很敏感，但很少有文献考虑机器人感知模型在广泛存在的摄像机运动扰动下的稳健性。为此，我们研究了视觉感知模型在摄像机运动扰动下的稳健性，以考察摄像机运动对机器人感知的影响。具体地说，我们提出了一种适用于任意图像分类模型的运动平滑技术，其在摄像机运动扰动下的健壮性可以得到验证。提出的基于摄像机运动平滑的健壮性认证框架，为视觉感知模块提供了有效的、可扩展的健壮性保证，适用于广泛的机器人应用。据我们所知，这是第一次为深度感知模块提供针对摄像头运动的鲁棒性认证，从而提高了机器人感知的可信性。引入了一个真实的室内机器人数据集，其中包含整个房间MetaRoom的密集点云地图，用于完成具有挑战性的可验证的稳健感知任务。我们进行了大量的实验，通过对摄像机运动扰动进行运动平滑来验证该认证方法。我们的框架保证了在-0.1m到0.1m范围内相机沿深度方向平移扰动的认证准确率为81.7%。通过在手持眼摄像机的机械臂上进行硬件实验，验证了该方法在真实机器人上的有效性。代码可在https://github.com/HanjiangHu/camera-motion-smoothing.上获得

**[Paper URL](https://proceedings.mlr.press/v205/hu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/hu23b/hu23b.pdf)** 

# GLSO: Grammar-guided Latent Space Optimization for Sample-efficient Robot Design Automation
**题目:** GLSO：文法引导的潜在空间优化，实现样本高效的机器人设计自动化

**作者:** Jiaheng Hu, Julian Whitman, Howie Choset

**Abstract:** Robots have been used in all sorts of automation, and yet the design of robots remains mainly a manual task. We seek to provide design tools to automate the design of robots themselves. An important challenge in robot design automation is the large and complex design search space which grows exponentially with the number of components, making optimization difficult and sample inefficient. In this work, we present Grammar-guided Latent Space Optimization (GLSO), a framework that transforms design automation into a low-dimensional continuous optimization problem by training a graph variational autoencoder (VAE) to learn a mapping between the graph-structured design space and a continuous latent space. This transformation allows optimization to be conducted in a continuous latent space, where sample efficiency can be significantly boosted by applying algorithms such as Bayesian Optimization. GLSO guides training of the VAE using graph grammar rules and robot world space features, such that the learned latent space focus on valid robots and is easier for the optimization algorithm to explore. Importantly, the trained VAE can be reused to search for designs specialized to multiple different tasks without retraining. We evaluate GLSO by designing robots for a set of locomotion tasks in simulation, and demonstrate that our method outperforms related state-of-the-art robot design automation methods.

**摘要:** 机器人已经被用于各种自动化，但机器人的设计仍然主要是手工任务。我们寻求提供设计工具来自动化机器人本身的设计。机器人设计自动化中的一个重要挑战是庞大而复杂的设计搜索空间，它随着部件的数量呈指数级增长，使得优化变得困难，样本效率低下。在这项工作中，我们提出了语法制导的潜在空间优化(GLSO)框架，它通过训练图变分自动编码器(VAE)来学习图结构设计空间与连续潜在空间之间的映射，从而将设计自动化转化为低维连续优化问题。这种变换允许在连续的潜在空间中进行优化，其中通过应用贝叶斯优化等算法可以显著提高样本效率。GLSO使用图文法规则和机器人世界空间特征来指导VAE的训练，使学习的潜在空间集中在有效的机器人上，便于优化算法的探索。重要的是，经过训练的VAE可以重新用于搜索专门用于多个不同任务的设计，而无需重新培训。在仿真中，我们通过设计一组运动任务的机器人来评估GLSO，并证明了我们的方法比相关的最先进的机器人设计自动化方法要好。

**[Paper URL](https://proceedings.mlr.press/v205/hu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/hu23c/hu23c.pdf)** 

# Masked World Models for Visual Control
**题目:** 视觉控制的掩蔽世界模型

**作者:** Younggyo Seo, Danijar Hafner, Hao Liu, Fangchen Liu, Stephen James, Kimin Lee, Pieter Abbeel

**Abstract:** Visual model-based reinforcement learning (RL) has the potential to enable sample-efficient robot learning from visual observations. Yet the current approaches typically train a single model end-to-end for learning both visual representations and dynamics, making it difficult to accurately model the interaction between robots and small objects. In this work, we introduce a visual model-based RL framework that decouples visual representation learning and dynamics learning. Specifically, we train an autoencoder with convolutional layers and vision transformers (ViT) to reconstruct pixels given masked convolutional features, and learn a latent dynamics model that operates on the representations from the autoencoder. Moreover, to encode task-relevant information, we introduce an auxiliary reward prediction objective for the autoencoder. We continually update both autoencoder and dynamics model using online samples collected from environment interaction. We demonstrate that our decoupling approach achieves state-of-the-art performance on a variety of visual robotic tasks from Meta-world and RLBench, e.g., we achieve 81.7% success rate on 50 visual robotic manipulation tasks from Meta-world, while the baseline achieves 67.9%. Code is available on the project website: https://sites.google.com/view/mwm-rl.

**摘要:** 基于视觉模型的强化学习(RL)有可能使机器人从视觉观察中进行样本高效学习。然而，目前的方法通常是端到端地训练单个模型来学习视觉表示和动力学，这使得很难准确地对机器人和小对象之间的交互进行建模。在这项工作中，我们介绍了一种基于视觉模型的RL框架，该框架将视觉表征学习和动态学习解耦。具体地说，我们用卷积层和视觉转换器(VIT)训练自动编码器，以重建给定掩蔽卷积特征的像素，并学习对自动编码器的表示进行操作的潜在动力学模型。此外，为了编码与任务相关的信息，我们为自动编码器引入了一个辅助奖励预测目标。我们使用从环境交互中收集的在线样本不断更新自动编码器和动力学模型。实验结果表明，在Meta-World和RLB边的各种视觉机器人任务上，我们的解耦方法都达到了最好的性能，例如，我们在Meta-World的50个视觉机器人操作任务上获得了81.7%的成功率，而基线达到了67.9%。代码可在项目网站上找到：https://sites.google.com/view/mwm-rl.

**[Paper URL](https://proceedings.mlr.press/v205/seo23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/seo23a/seo23a.pdf)** 

# On-Robot Learning With Equivariant Models
**题目:** 使用等变模型的机器人学习

**作者:** Dian Wang, Mingxi Jia, Xupeng Zhu, Robin Walters, Robert Platt

**Abstract:** Recently, equivariant neural network models have been shown to improve sample efficiency for tasks in computer vision and reinforcement learning. This paper explores this idea in the context of on-robot policy learning in which a policy must be learned entirely on a physical robotic system without reference to a model, a simulator, or an offline dataset. We focus on applications of Equivariant SAC to robotic manipulation and explore a number of variations of the algorithm. Ultimately, we demonstrate the ability to learn several non-trivial manipulation tasks completely through on-robot experiences in less than an hour or two of wall clock time.

**摘要:** 最近，等变神经网络模型已被证明可以提高计算机视觉和强化学习任务的样本效率。本文在机器人政策学习的背景下探讨了这一想法，其中政策必须完全在物理机器人系统上学习，而无需参考模型、模拟器或离线数据集。我们专注于等变SAC在机器人操纵中的应用，并探索该算法的多种变体。最终，我们展示了在不到一两个小时的时间内通过机器人体验完全学习几项重要的操纵任务的能力。

**[Paper URL](https://proceedings.mlr.press/v205/wang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wang23c/wang23c.pdf)** 

# Neural Geometric Fabrics: Efficiently Learning High-Dimensional Policies from Demonstration
**题目:** 神经几何结构：从演示中有效学习多维策略

**作者:** Mandy Xie, Ankur Handa, Stephen Tyree, Dieter Fox, Harish Ravichandar, Nathan D. Ratliff, Karl Van Wyk

**Abstract:** Learning dexterous manipulation policies for multi-fingered robots has been a long-standing challenge in robotics. Existing methods either limit themselves to highly constrained problems and smaller models to achieve extreme sample efficiency or sacrifice sample efficiency to gain capacity to solve more complex tasks with deep neural networks. In this work, we develop a structured approach to sample-efficient learning of dexterous manipulation skills from demonstrations by leveraging recent advances in robot motion generation and control. Specifically, our policy structure is induced by Geometric Fabrics - a recent framework that generalizes classical mechanical systems to allow for flexible design of expressive robot motions. To avoid the cumbersome manual design required by existing motion generators, we introduce Neural Geometric Fabric (NGF) - a framework that learns Geometric Fabric-based policies from data. NGF policies are provably stable and capable of encoding speed-invariant geometries of complex motions in multiple task spaces simultaneously. We demonstrate that NGFs can learn to perform a variety of dexterous manipulation tasks on a 23-DoF hand-arm physical robotic platform purely from demonstrations. Results from comprehensive comparative and ablative experiments show that NGF’s structure and action spaces help learn acceleration-based policies that consistently outperform state-of-the-art baselines like Riemannian Motion Policies (RMPs), and other commonly used networks, such as feed-forward and recurrent neural networks. More importantly, we demonstrate that NGFs do not rely on often-used and expertly-designed operational-space controllers, promoting an advancement towards efficiently learning safe, stable, and high-dimensional controllers.

**摘要:** 学习多指机器人的灵活操作策略一直是机器人学中的一个长期挑战。现有的方法要么局限于高度受限的问题和较小的模型，以达到极高的样本效率，要么牺牲样本效率，以获得利用深度神经网络解决更复杂任务的能力。在这项工作中，我们开发了一种结构化的方法，通过利用机器人运动生成和控制方面的最新进展，从演示中有效地学习灵活的操作技能。具体地说，我们的策略结构是由几何结构引起的-这是一个最近的框架，它概括了经典的机械系统，允许灵活地设计富有表现力的机器人运动。为了避免现有运动生成器所需的繁琐的手动设计，我们引入了神经几何结构(NGF)-一个从数据中学习基于几何结构的策略的框架。NGF策略被证明是稳定的，并且能够同时编码多个任务空间中复杂运动的速度不变的几何形状。我们演示了NGF可以纯粹通过演示在23个自由度的手臂物理机器人平台上学习执行各种灵活的操作任务。综合比较和烧蚀实验的结果表明，NGF的结构和动作空间有助于学习基于加速度的策略，这些策略的性能始终优于最先进的基线，如黎曼运动策略(RMP)，以及其他常用的网络，如前馈和递归神经网络。更重要的是，我们证明了NGF不依赖于常用的和专业设计的操作空间控制器，促进了高效学习安全、稳定和高维控制器的进步。

**[Paper URL](https://proceedings.mlr.press/v205/xie23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/xie23a/xie23a.pdf)** 

# See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation
**题目:** 看、听和感觉：机器人操纵的智能感官融合

**作者:** Hao Li, Yizhi Zhang, Junzhe Zhu, Shaoxiong Wang, Michelle A Lee, Huazhe Xu, Edward Adelson, Li Fei-Fei, Ruohan Gao, Jiajun Wu

**Abstract:** Humans use all of their senses to accomplish different tasks in everyday activities. In contrast, existing work on robotic manipulation mostly relies on one, or occasionally two modalities, such as vision and touch. In this work, we systematically study how visual, auditory, and tactile perception can jointly help robots to solve complex manipulation tasks. We build a robot system that can see with a camera, hear with a contact microphone, and feel with a vision-based tactile sensor, with all three sensory modalities fused with a self-attention model. Results on two challenging tasks, dense packing and pouring, demonstrate the necessity and power of multisensory perception for robotic manipulation: vision displays the global status of the robot but can often suffer from occlusion, audio provides immediate feedback of key moments that are even invisible, and touch offers precise local geometry for decision making. Leveraging all three modalities, our robotic system significantly outperforms prior methods.

**摘要:** 在日常活动中，人类用所有的感官来完成不同的任务。相比之下，现有的机器人操纵工作大多依赖于一种，甚至偶尔两种模式，如视觉和触觉。在这项工作中，我们系统地研究了视觉、听觉和触觉如何共同帮助机器人解决复杂的操作任务。我们建立了一个机器人系统，它可以用摄像头看，用接触式麦克风听，用基于视觉的触觉传感器感觉，所有三种感觉模式都与自我注意模型相融合。在两项具有挑战性的任务(密集包装和浇注)上的结果表明，多感官感知对于机器人操作是必要的和强大的：视觉显示机器人的全局状态，但经常会受到咬合的影响，音频提供甚至看不见的关键时刻的即时反馈，而触摸为决策提供精确的局部几何图形。利用所有这三种模式，我们的机器人系统显著优于以前的方法。

**[Paper URL](https://proceedings.mlr.press/v205/li23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/li23c/li23c.pdf)** 

# Inferring Versatile Behavior from Demonstrations by Matching Geometric Descriptors
**题目:** 通过匹配几何描述符从演示中推断多功能行为

**作者:** Niklas Freymuth, Nicolas Schreiber, Aleksandar Taranovic, Philipp Becker, Gerhard Neumann

**Abstract:** Humans intuitively solve tasks in versatile ways, varying their behavior in terms of trajectory-based planning and for individual steps. Thus, they can easily generalize and adapt to new and changing environments. Current Imitation Learning algorithms often only consider unimodal expert demonstrations and act in a state-action-based setting, making it difficult for them to imitate human behavior in case of versatile demonstrations. Instead, we combine a mixture of movement primitives with a distribution matching objective to learn versatile behaviors that match the expert’s behavior and versatility. To facilitate generalization to novel task configurations, we do not directly match the agent’s and expert’s trajectory distributions but rather work with concise geometric descriptors which generalize well to unseen task configurations. We empirically validate our method on various robot tasks using versatile human demonstrations and compare to imitation learning algorithms in a state-action setting as well as a trajectory-based setting. We find that the geometric descriptors greatly help in generalizing to new task configurations and that combining them with our distribution-matching objective is crucial for representing and reproducing versatile behavior.

**摘要:** 人类以多种方式直观地解决任务，在基于轨迹的规划和个人步骤方面改变他们的行为。因此，它们可以很容易地概括和适应新的和不断变化的环境。目前的模仿学习算法往往只考虑单峰的专家演示，并且基于状态-动作的设置，使得它们很难在多用途演示的情况下模仿人类的行为。相反，我们将运动基元与分布匹配目标相结合，以学习与专家的行为和多功能性相匹配的多才多艺的行为。为了便于对新任务配置的泛化，我们不直接匹配代理和专家的轨迹分布，而是使用简洁的几何描述符，它很好地泛化到看不见的任务配置。在不同的机器人任务上，我们使用多种人类演示对我们的方法进行了经验验证，并与状态-动作设置以及基于轨迹的设置下的模仿学习算法进行了比较。我们发现，几何描述符在很大程度上有助于推广到新的任务配置，并且将它们与我们的分布匹配目标相结合对于表示和再现多样化的行为至关重要。

**[Paper URL](https://proceedings.mlr.press/v205/freymuth23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/freymuth23a/freymuth23a.pdf)** 

# Generative Category-Level Shape and Pose Estimation with Semantic Primitives
**题目:** 使用语义基元的生成性类别级形状和姿势估计

**作者:** Guanglin Li, Yifeng Li, Zhichao Ye, Qihang Zhang, Tao Kong, Zhaopeng Cui, Guofeng Zhang

**Abstract:** Empowering autonomous agents with 3D understanding for daily objects is a grand challenge in robotics applications. When exploring in an unknown environment, existing methods for object pose estimation are still not satisfactory due to the diversity of object shapes. In this paper, we propose a novel framework for category-level object shape and pose estimation from a single RGB-D image. To handle the intra-category variation, we adopt a semantic primitive representation that encodes diverse shapes into a unified latent space, which is the key to establish reliable correspondences between observed point clouds and estimated shapes. Then, by using a SIM(3)-invariant shape descriptor, we gracefully decouple the shape and pose of an object, thus supporting latent shape optimization of target objects in arbitrary poses. Extensive experiments show that the proposed method achieves SOTA pose estimation performance and better generalization in the real-world dataset. Code and video are available at https://zju3dv.github.io/gCasp.

**摘要:** 在机器人应用中，使自主代理能够理解日常对象的3D是一个巨大的挑战。当在未知环境中进行探索时，由于目标形状的多样性，现有的目标姿态估计方法仍然不能令人满意。本文提出了一种基于单幅RGB-D图像的类别级目标形状和姿态估计框架。为了处理类内变化，我们采用了一种语义基元表示，将不同的形状编码到一个统一的潜在空间中，这是在观测到的点云和估计的形状之间建立可靠的对应关系的关键。然后，通过使用SIM(3)不变形状描述子，将物体的形状和姿态优雅地解耦，从而支持目标物体在任意姿势下的潜在形状优化。大量实验表明，该方法在真实数据集上取得了较好的SOTA姿态估计性能和较好的泛化能力。代码和视频可在https://zju3dv.github.io/gCasp.上获得

**[Paper URL](https://proceedings.mlr.press/v205/li23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/li23d/li23d.pdf)** 

# Learning Goal-Conditioned Policies Offline with Self-Supervised Reward Shaping
**题目:** 通过自我监督的奖励塑造离线学习目标条件政策

**作者:** Lina Mezghani, Sainbayar Sukhbaatar, Piotr Bojanowski, Alessandro Lazaric, Karteek Alahari

**Abstract:** Developing agents that can execute multiple skills by learning from pre-collected datasets is an important problem in robotics, where online interaction with the environment is extremely time-consuming. Moreover, manually designing reward functions for every single desired skill is prohibitive. Prior works targeted these challenges by learning goal-conditioned policies from offline datasets without manually specified rewards, through hindsight relabeling. These methods suffer from the issue of sparsity of rewards, and fail at long-horizon tasks. In this work, we propose a novel self-supervised learning phase on the pre-collected dataset to understand the structure and the dynamics of the model, and shape a dense reward function for learning policies offline. We evaluate our method on three continuous control tasks, and show that our model significantly outperforms existing approaches, especially on tasks that involve long-term planning.

**摘要:** 开发可以通过从预先收集的数据集中学习来执行多种技能的代理是机器人技术中的一个重要问题，其中与环境的在线交互极其耗时。此外，为每一项所需技能手动设计奖励功能是令人望而却步的。之前的作品通过事后诸葛亮重新标记，从离线数据集中学习目标条件政策，无需手动指定奖励。这些方法存在回报稀缺的问题，并且在长期任务中失败。在这项工作中，我们在预先收集的数据集上提出了一个新颖的自我监督学习阶段，以了解模型的结构和动态，并为离线学习策略塑造密集的奖励函数。我们在三个连续控制任务上评估了我们的方法，并表明我们的模型显着优于现有方法，尤其是在涉及长期规划的任务上。

**[Paper URL](https://proceedings.mlr.press/v205/mezghani23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/mezghani23a/mezghani23a.pdf)** 

# ROS-PyBullet Interface: A Framework for Reliable Contact Simulation and Human-Robot Interaction
**题目:** ROS-PyBullet接口：可靠接触模拟和人机交互的框架

**作者:** Christopher Mower, Theodoros Stouraitis, João Moura, Christian Rauch, Lei Yan, Nazanin Zamani Behabadi, Michael Gienger, Tom Vercauteren, Christos Bergeles, Sethu Vijayakumar

**Abstract:** Reliable contact simulation plays a key role in the development of (semi-)autonomous robots, especially when dealing with contact-rich manipulation scenarios, an active robotics research topic. Besides simulation, components such as sensing, perception, data collection, robot hardware control, human interfaces, etc. are all key enablers towards applying machine learning algorithms or model-based approaches in real world systems. However, there is a lack of software connecting reliable contact simulation with the larger robotics ecosystem (i.e. ROS, Orocos), for a more seamless application of novel approaches, found in the literature, to existing robotic hardware. In this paper, we present the ROS-PyBullet Interface, a framework that provides a bridge between the reliable contact/impact simulator PyBullet and the Robot Operating System (ROS). Furthermore, we provide additional utilities for facilitating Human-Robot Interaction (HRI) in the simulated environment. We also present several use-cases that highlight the capabilities and usefulness of our framework. Our code base is open source and can be found at https://github.com/ros-pybullet/ros_pybullet_interface.

**摘要:** 可靠的接触模拟在(半)自主机器人的发展中起着关键作用，特别是在处理接触丰富的操作场景时，这是一个活跃的机器人学研究课题。除了仿真，传感、感知、数据采集、机器人硬件控制、人机界面等组件都是将机器学习算法或基于模型的方法应用于现实世界系统的关键使能。然而，缺乏将可靠的接触模拟与更大的机器人生态系统(即，ROS，Orocos)连接起来的软件，以便将文献中发现的新方法更无缝地应用于现有的机器人硬件。在本文中，我们提出了ROS-PyBullet接口，这是一个在可靠的接触/碰撞模拟器PyBullet和机器人操作系统(ROS)之间提供桥梁的框架。此外，我们还为促进模拟环境中的人-机器人交互(HRI)提供了额外的实用程序。我们还提供了几个用例，这些用例突出了我们框架的功能和实用性。我们的代码库是开源的，可以在https://github.com/ros-pybullet/ros_pybullet_interface.上找到

**[Paper URL](https://proceedings.mlr.press/v205/mower23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/mower23a/mower23a.pdf)** 

# PLATO: Predicting Latent Affordances Through Object-Centric Play
**题目:** PLATO：通过以对象为中心的游戏预测潜在的功能

**作者:** Suneel Belkhale, Dorsa Sadigh

**Abstract:** Constructing a diverse repertoire of manipulation skills in a scalable fashion remains an unsolved challenge in robotics. One way to address this challenge is with unstructured human play, where humans operate freely in an environment to reach unspecified goals. Play is a simple and cheap method for collecting diverse user demonstrations with broad state and goal coverage over an environment. Due to this diverse coverage, existing approaches for learning from play are more robust to online policy deviations from the offline data distribution. However, these methods often struggle to learn under scene variation and on challenging manipulation primitives, due in part to improperly associating complex behaviors to the scene changes they induce. Our insight is that an object-centric view of play data can help link human behaviors and the resulting changes in the environment, and thus improve multi-task policy learning. In this work, we construct a latent space to model object \textit{affordances} – properties of an object that define its uses – in the environment, and then learn a policy to achieve the desired affordances. By modeling and predicting the desired affordance across variable horizon tasks, our method, Predicting Latent Affordances Through Object-Centric Play (PLATO), outperforms existing methods on complex manipulation tasks in both 2D and 3D object manipulation simulation and real world environments for diverse types of interactions. Videos can be found on our website: https://sites.google.com/view/plato-corl22/home.

**摘要:** 在机器人学中，以可扩展的方式构建多样化的操作技能体系仍然是一个悬而未决的挑战。解决这一挑战的一种方法是无组织的人类游戏，即人类在一个环境中自由操作，以实现未指定的目标。Play是一种简单而廉价的方法，用于收集各种用户演示，并在环境中覆盖广泛的状态和目标。由于覆盖范围的多样性，现有的从游戏中学习的方法对在线政策与离线数据分布的偏差更加稳健。然而，这些方法经常在场景变化和挑战操纵基元方面难以学习，部分原因是不正确地将复杂行为与它们引起的场景变化相关联。我们的见解是，以对象为中心的游戏数据视图可以帮助将人类行为与由此导致的环境变化联系起来，从而改善多任务策略学习。在这项工作中，我们构建了一个潜在空间来对环境中的对象\文本(定义其用途的对象的属性)进行建模，然后学习实现期望的启示的策略。通过对可变水平任务的期望启示进行建模和预测，我们的方法，通过以对象为中心的游戏(柏拉图)预测潜在的平等，在2D和3D对象操纵模拟和现实世界环境中针对不同类型的交互的复杂操纵任务上的性能优于现有方法。视频可在我们的网站上找到：https://sites.google.com/view/plato-corl22/home.

**[Paper URL](https://proceedings.mlr.press/v205/belkhale23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/belkhale23a/belkhale23a.pdf)** 

# You Only Look at One: Category-Level Object Representations for Pose Estimation From a Single Example
**题目:** 您只看一个：从单个示例中进行姿势估计的类别级对象表示

**作者:** Walter Goodwin, Ioannis Havoutis, Ingmar Posner

**Abstract:** In order to meaningfully interact with the world, robot manipulators must be able to interpret objects they encounter. A critical aspect of this interpretation is pose estimation: inferring quantities that describe the position and orientation of an object in 3D space. Most existing approaches to pose estimation make limiting assumptions, often working only for specific, known object instances, or at best generalising to an object category using large pose-labelled datasets. In this work, we present a method for achieving category-level pose estimation by inspection of just a single object from a desired category. We show that we can subsequently perform accurate pose estimation for unseen objects from an inspected category, and considerably outperform prior work by exploiting multi-view correspondences. We demonstrate that our method runs in real-time, enabling a robot manipulator to rearrange previously unseen objects faithfully in terms of placement and orientation. Finally, we showcase our method in a continual learning setting, with a robot able to determine whether objects belong to known categories, and if not, use active perception to produce a one-shot category representation for subsequent pose estimation

**摘要:** 为了与世界有意义地互动，机器人操作员必须能够解释他们遇到的物体。这种解释的一个关键方面是姿势估计：推断描述物体在3D空间中的位置和方向的量。大多数现有的姿势估计方法都做了有限的假设，通常只对特定的已知对象实例起作用，或者充其量使用大量的姿势标签数据集来概括对象类别。在这项工作中，我们提出了一种方法，通过检查期望类别中的单个对象来实现类别级别的姿势估计。我们表明，我们可以随后对来自检查类别的不可见对象执行准确的姿态估计，并且通过利用多视角对应关系显著地优于先前的工作。我们证明了我们的方法是实时运行的，使机器人操作手能够在位置和方向方面忠实地重新排列以前看不到的对象。最后，我们在一个持续学习的环境中展示了我们的方法，机器人能够确定对象是否属于已知类别，如果不属于，则使用主动感知来产生一次类别表示，用于后续的姿势估计

**[Paper URL](https://proceedings.mlr.press/v205/goodwin23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/goodwin23a/goodwin23a.pdf)** 

# ARC - Actor Residual Critic for Adversarial Imitation Learning
**题目:** ARC -对抗模仿学习的演员残余批评者

**作者:** Ankur Deka, Changliu Liu, Katia P. Sycara

**Abstract:** Adversarial Imitation Learning (AIL) is a class  of popular state-of-the-art Imitation Learning algorithms commonly used in robotics. In AIL, an artificial adversary’s misclassification is used as a reward signal that is optimized by any standard Reinforcement Learning (RL) algorithm. Unlike most RL settings, the reward in AIL is $differentiable$ but current model-free RL algorithms do not make use of this property to train a policy. The reward is AIL is also $shaped$ since it comes from an adversary. We leverage the differentiability property of the shaped AIL reward function and formulate a class of Actor Residual Critic (ARC) RL algorithms. ARC algorithms draw a parallel to the standard Actor-Critic (AC) algorithms in RL literature and uses a residual critic, $C$ function (instead of the standard $Q$ function) to approximate only the discounted future return (excluding the immediate reward). ARC algorithms have similar convergence properties as the standard AC algorithms with the additional advantage that the gradient through the immediate reward is exact. For the discrete (tabular) case with finite states, actions, and known dynamics, we prove that policy iteration with $C$ function converges to an optimal policy. In the continuous case with function approximation and unknown dynamics, we experimentally show that ARC aided AIL outperforms standard AIL in simulated continuous-control and real robotic manipulation tasks. ARC algorithms are simple to implement and can be incorporated into any existing AIL implementation with an AC algorithm.

**摘要:** 对抗性模仿学习(AIL)是机器人学中常用的一类流行的模仿学习算法。总之，人工对手的错误分类被用作奖励信号，任何标准的强化学习(RL)算法都会对其进行优化。与大多数RL设置不同的是，AIL中的奖励是$Differential$，但当前的无模型RL算法没有利用这一性质来训练策略。奖励是All也是$形的$，因为它来自对手。利用赋形AIL奖励函数的可微性，构造了一类ARC RL算法。ARC算法与RL文献中的标准Actor-Critic(AC)算法相似，并使用剩余批评者$C$函数(而不是标准的$Q$函数)来仅近似贴现的未来回报(不包括即时奖励)。ARC算法具有与标准AC算法类似的收敛特性，另外一个优点是通过即时奖励的梯度是精确的。对于具有有限状态、动作和已知动态的离散(表)情形，我们证明了使用$C$函数的策略迭代收敛到最优策略。在函数逼近和动力学未知的连续情况下，实验表明ARC辅助AIL在模拟连续控制和真实机器人操作任务中的性能优于标准AIL。ARC算法易于实现，并且可以结合到任何现有的带有AC算法的AIL实现中。

**[Paper URL](https://proceedings.mlr.press/v205/deka23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/deka23a/deka23a.pdf)** 

# JFP: Joint Future Prediction with Interactive Multi-Agent Modeling for Autonomous Driving
**题目:** JFP：自动驾驶的交互式多智能体建模联合未来预测

**作者:** Wenjie Luo, Cheol Park, Andre Cornman, Benjamin Sapp, Dragomir Anguelov

**Abstract:** We propose \textit{JFP}, a Joint Future Prediction model that can learn to generate accurate and consistent multi-agent future trajectories. For this task, many different methods have been proposed to capture social interactions in the encoding part of the model, however, considerably less focus has been placed on representing interactions in the decoder and output stages. As a result, the predicted trajectories are not necessarily consistent with each other, and often result in unrealistic trajectory overlaps. In contrast, we propose an end-to-end trainable model that learns directly the interaction between pairs of agents in a structured, graphical model formulation in order to generate consistent future trajectories. It sets new state-of-the-art results on Waymo Open Motion Dataset (WOMD) for the interactive setting. We also investigate a more complex multi-agent setting for both WOMD and a larger internal dataset, where our approach improves significantly on the trajectory overlap metrics while obtaining on-par or better performance on single-agent trajectory metrics.

**摘要:** 我们提出了一种联合未来预测模型-联合未来预测模型，该模型能够学习生成准确、一致的多智能体未来轨迹。对于这项任务，已经提出了许多不同的方法来捕获模型编码部分的社交交互，然而，对表示解码和输出阶段的交互的关注要少得多。因此，预测的轨迹不一定彼此一致，并且经常导致不切实际的轨迹重叠。相反，我们提出了一个端到端的可训练模型，该模型直接学习结构化、图形模型公式中代理对之间的相互作用，以便生成一致的未来轨迹。它在Waymo开放式运动数据集(WOMD)上为交互设置设置了新的最先进的结果。我们还研究了WOMD和更大的内部数据集的更复杂的多代理设置，其中我们的方法在轨迹重叠度量上有显著改进，而在单代理轨迹度量上获得了相当或更好的性能。

**[Paper URL](https://proceedings.mlr.press/v205/luo23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/luo23a/luo23a.pdf)** 

# Online Inverse Reinforcement Learning with Learned Observation Model
**题目:** 具有习得观察模型的在线反向强化学习

**作者:** Saurabh Arora, Prashant Doshi, Bikramjit Banerjee

**Abstract:** With the motivation of extending incremental inverse reinforcement learning (I2RL) to real-world robotics applications with noisy observations as well as an unknown observation model, we introduce a new method (RIMEO) that approximates the observation model in order to best estimate the noise-free ground truth underlying the observations. It learns a maximum entropy distribution over the observation features governing the perception process, and then uses the inferred observation model to learn the reward function. Experimental evaluation is performed in two robotics tasks: (1) post-harvest vegetable sorting with a Sawyer arm based on human demonstration, and (2) breaching a perimeter patrol by two Turtlebots. Our experiments reveal that RIMEO learns a more accurate policy compared to (a) a state-of-the-art IRL method that does not directly learn an observation model, and (b) a custom baseline that learns a less sophisticated observation model. Furthermore, we show that RIMEO admits formal guarantees of monotonic convergence and a sample complexity bound.

**摘要:** 为了将增量式逆强化学习(I2RL)扩展到含有噪声观测和未知观测模型的真实机器人应用中，我们引入了一种新的方法(RIMEO)来逼近观测模型，以便最好地估计观测背后的无噪声地面真实情况。它学习支配感知过程的观测特征上的最大熵分布，然后使用推断的观测模型来学习奖励函数。在两个机器人任务中进行了实验评估：(1)基于人类演示的Sawyer手臂采摘后的蔬菜分拣；(2)由两个Turtlebot突破周边巡逻。我们的实验表明，与(A)不直接学习观测模型的最先进的IRL方法和(B)学习不太复杂的观测模型的定制基线相比，RIMEO学习更准确的策略。此外，我们还证明了RIMEO具有形式上的单调收敛保证和一个样本复杂性界。

**[Paper URL](https://proceedings.mlr.press/v205/arora23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/arora23a/arora23a.pdf)** 

# Hypernetworks in Meta-Reinforcement Learning
**题目:** 元强化学习中的超网络

**作者:** Jacob Beck, Matthew Thomas Jackson, Risto Vuorio, Shimon Whiteson

**Abstract:** Training a reinforcement learning (RL) agent on a real-world robotics task remains generally impractical due to sample inefficiency. Multi-task RL and meta-RL aim to improve sample efficiency by generalizing over a distribution of related tasks. However, doing so is difficult in practice: In multi-task RL, state of the art methods often fail to outperform a degenerate solution that simply learns each task separately. Hypernetworks are a promising path forward since they replicate the separate policies of the degenerate solution while also allowing for generalization across tasks, and are applicable to meta-RL. However, evidence from supervised learning suggests hypernetwork performance is highly sensitive to the initialization. In this paper, we 1) show that hypernetwork initialization is also a critical factor in meta-RL, and that naive initializations yield poor performance; 2) propose a novel hypernetwork initialization scheme that matches or exceeds the performance of a state-of-the-art approach proposed for supervised settings, as well as being simpler and more general; and 3) use this method to show that hypernetworks can improve performance in meta-RL by evaluating on multiple simulated robotics benchmarks.

**摘要:** 由于样本效率低下，在真实机器人任务中训练强化学习(RL)代理通常是不切实际的。多任务RL和Meta-RL旨在通过对相关任务的分布进行泛化来提高样本效率。然而，这样做在实践中是困难的：在多任务RL中，最先进的方法通常无法超越简单地分别学习每个任务的退化解决方案。超级网络是一条很有前途的前进道路，因为它们复制了退化解决方案的单独策略，同时还允许跨任务的泛化，并且适用于META-RL。然而，来自监督学习的证据表明，超网络的性能对初始化高度敏感。在本文中，我们1)证明了超网络初始化也是Meta-RL中的一个关键因素，而简单的初始化会导致较差的性能；2)提出了一种新的超网络初始化方案，其性能达到或超过了针对有监督设置提出的最新方法，并且更简单、更通用；3)使用该方法通过在多个模拟机器人基准上的评估来证明超网络可以提高Meta-RL的性能。

**[Paper URL](https://proceedings.mlr.press/v205/beck23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/beck23a/beck23a.pdf)** 

# Efficient Tactile Simulation with Differentiability for Robotic Manipulation
**题目:** 机器人操纵的高效可区分触觉模拟

**作者:** Jie Xu, Sangwoon Kim, Tao Chen, Alberto Rodriguez Garcia, Pulkit Agrawal, Wojciech Matusik, Shinjiro Sueda

**Abstract:** Efficient simulation of tactile sensors can unlock new opportunities for learning tactile-based manipulation policies in simulation and then transferring the learned policy to real systems, but fast and reliable simulators for dense tactile normal and shear force fields are still under-explored. We present a novel approach for efficiently simulating both the normal and shear tactile force field covering the entire contact surface with an arbitrary tactile sensor spatial layout. Our simulator also provides analytical gradients of the tactile forces to accelerate policy learning. We conduct extensive simulation experiments to showcase our approach and demonstrate successful zero-shot sim-to-real transfer for a high-precision peg-insertion task with high-resolution vision-based GelSlim tactile sensors.

**摘要:** 触觉传感器的高效模拟可以为在模拟中学习基于喷嘴的操纵政策，然后将学习到的政策转移到真实系统中打开新的机会，但用于密集触觉法向和剪切力场的快速可靠的模拟器仍然没有得到充分的开发。我们提出了一种新颖的方法，可以有效地模拟覆盖整个接触表面的法向和剪切触觉力场，并具有任意触觉传感器空间布局。我们的模拟器还提供触觉力的分析梯度，以加速政策学习。我们进行了广泛的模拟实验，以展示我们的方法，并演示使用高分辨率基于视觉的GelSlim触觉传感器，成功实现高精度钉插入任务的零镜头实时转换。

**[Paper URL](https://proceedings.mlr.press/v205/xu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/xu23b/xu23b.pdf)** 

# Exploring with Sticky Mittens: Reinforcement Learning with Expert Interventions via Option Templates
**题目:** 使用粘指手套进行探索：通过选项模板进行专家干预的强化学习

**作者:** Souradeep Dutta, Kaustubh Sridhar, Osbert Bastani, Edgar Dobriban, James Weimer, Insup Lee, Julia Parish-Morris

**Abstract:** Long horizon robot learning tasks with sparse rewards pose a significant challenge for current reinforcement learning algorithms. A key feature enabling humans to learn challenging control tasks is that they often receive expert intervention that enables them to understand the high-level structure of the task before mastering low-level control actions. We propose a framework for leveraging expert intervention to solve long-horizon reinforcement learning tasks. We consider \emph{option templates}, which are specifications encoding a potential option that can be trained using reinforcement learning. We formulate expert intervention as allowing the agent to execute option templates  before learning an implementation. This enables them to use an option, before committing costly resources to learning it. We evaluate our approach on three challenging reinforcement learning problems, showing that it outperforms state-of-the-art approaches by two orders of magnitude. Videos of trained agents and our code can be found at: https://sites.google.com/view/stickymittens

**摘要:** 稀疏报酬下的长时间机器人学习任务对现有的强化学习算法提出了很大的挑战。使人类能够学习具有挑战性的控制任务的一个关键特征是，他们经常接受专家干预，使他们能够在掌握低级别控制动作之前了解任务的高级结构。我们提出了一个利用专家干预来解决长时间强化学习任务的框架。我们考虑\emph{选项模板}，它是编码潜在选项的规范，可以使用强化学习进行训练。我们将专家干预定义为允许代理在学习实现之前执行选项模板。这使他们能够在投入昂贵的资源学习之前使用一种选择。我们在三个具有挑战性的强化学习问题上对我们的方法进行了评估，结果表明它的性能比最先进的方法高出两个数量级。有关训练有素的特工的视频和我们的代码，请访问：https://sites.google.com/view/stickymittens

**[Paper URL](https://proceedings.mlr.press/v205/dutta23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/dutta23a/dutta23a.pdf)** 

# Learning Bimanual Scooping Policies for Food Acquisition
**题目:** 学习两次铲食政策以获取食物

**作者:** Jennifer Grannen, Yilin Wu, Suneel Belkhale, Dorsa Sadigh

**Abstract:** A robotic feeding system must be able to acquire a variety of foods. Prior bite acquisition works consider single-arm spoon scooping or fork skewering, which do not generalize to foods with complex geometries and deformabilities. For example, when acquiring a group of peas, skewering could smoosh the peas while scooping without a barrier could result in chasing the peas on the plate. In order to acquire foods with such diverse properties, we propose stabilizing food items during scooping using a second arm, for example, by pushing peas against the spoon with a flat surface to prevent dispersion. The addition of this second stabilizing arm can lead to a new set of challenges. Critically, these strategies should stabilize the food scene without interfering with the acquisition motion, which is especially difficult for easily breakable high-risk food items, such as tofu. These high-risk foods can break between the pusher and spoon during scooping, which can lead to food waste falling onto the plate or out of the workspace. We propose a general bimanual scooping primitive and an adaptive stabilization strategy that enables successful acquisition of a diverse set of food geometries and physical properties. Our approach, CARBS: Coordinated Acquisition with Reactive Bimanual Scooping, learns to stabilize without impeding task progress by identifying high-risk foods and robustly scooping them using closed-loop visual feedback. We find that CARBS is able to generalize across food shape, size, and deformability and is additionally able to manipulate multiple food items simultaneously. CARBS achieves 87.0% success on scooping rigid foods, which is 25.8% more successful than a single-arm baseline, and reduces food breakage by 16.2% compared to an analytical baseline. Videos can be found on our website at https://sites.google.com/view/bimanualscoop-corl22/home.

**摘要:** 机器人喂食系统必须能够获取各种食物。以前的咬合获取工作考虑单臂勺子或叉子串，这不适用于具有复杂几何形状和变形能力的食物。例如，当获得一组豌豆时，串可以使豌豆变得光滑，而没有障碍的铲可能会导致追逐盘子上的豌豆。为了获得具有这种不同特性的食物，我们建议在用第二只手臂铲取食物时稳定食物，例如，用平坦的表面将豌豆推到勺子上，以防止分散。增加这第二个稳定的手臂可能会带来一系列新的挑战。关键的是，这些策略应该在不干扰收购动议的情况下稳定食品场景，而收购动议对于豆腐等容易破碎的高风险食品来说尤其困难。这些高危食物在铲出过程中可能会在推进器和勺子之间破裂，这可能会导致食物垃圾掉到盘子里或掉出工作空间。我们提出了一种通用的双手铲原语和一种自适应稳定策略，使得能够成功地获取一组不同的食物几何和物理属性。我们的方法，碳水化合物：协调收购与反应性双手铲，通过识别高风险食物并使用闭环视觉反馈强有力地铲起它们，学习在不阻碍任务进度的情况下保持稳定。我们发现，碳水化合物能够概括食物的形状、大小和变形性，另外还能够同时操作多种食物。碳水化合物在挖取硬质食物方面的成功率为87.0%，比单臂基线的成功率高25.8%，与分析基线相比，食物破碎减少了16.2%。视频可以在我们的网站上找到，网址是https://sites.google.com/view/bimanualscoop-corl22/home.

**[Paper URL](https://proceedings.mlr.press/v205/grannen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/grannen23a/grannen23a.pdf)** 

# SE(3)-Equivariant Point Cloud-Based Place Recognition
**题目:** SE（3）-等变点基于云的地点识别

**作者:** Chien Erh Lin, Jingwei Song, Ray Zhang, Minghan Zhu, Maani Ghaffari

**Abstract:** This paper reports on a new 3D point cloud-based place recognition framework that uses SE(3)-equivariant networks to learn SE(3)-invariant global descriptors. We discover that, unlike existing methods, learned SE(3)-invariant global descriptors are more robust to matching inaccuracy and failure in severe rotation and translation configurations. Mobile robots undergo arbitrary rotational and translational movements. The SE(3)-invariant property ensures that the learned descriptors are robust to the rotation and translation changes in the robot pose and can represent the intrinsic geometric information of the scene. Furthermore, we have discovered that the attention module aids in the enhancement of performance while allowing significant downsampling. We evaluate the performance of the proposed framework on real-world data sets. The experimental results show that the proposed framework outperforms state-of-the-art baselines in various metrics, leading to a reliable point cloud-based place recognition network. We have open-sourced our code at: https://github.com/UMich-CURLY/se3_equivariant_place_recognition.

**摘要:** 本文提出了一种新的基于三维点云的位置识别框架，该框架使用SE(3)等变网络学习SE(3)不变全局描述符。我们发现，与现有方法不同，学习的SE(3)不变全局描述符在严重的旋转和平移配置中对匹配不准确和失败具有更强的鲁棒性。移动机器人可以进行任意的旋转和平移运动。SE(3)-不变性保证了学习的描述子对机器人姿态的旋转和平移变化具有较强的鲁棒性，并且能够表示场景的内在几何信息。此外，我们还发现，注意模块有助于提高性能，同时允许显著的下采样。我们在真实数据集上对该框架的性能进行了评估。实验结果表明，该框架在各种度量指标上都优于最新的基线，从而形成了一个可靠的基于点云的位置识别网络。我们已将代码开源，网址为：https://github.com/UMich-CURLY/se3_equivariant_place_recognition.

**[Paper URL](https://proceedings.mlr.press/v205/lin23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/lin23a/lin23a.pdf)** 

# Leveraging Language for Accelerated Learning of Tool Manipulation
**题目:** 利用语言加速工具操作学习

**作者:** Allen Z. Ren, Bharat Govil, Tsung-Yen Yang, Karthik R Narasimhan, Anirudha Majumdar

**Abstract:** Robust and generalized tool manipulation requires an understanding of the properties and affordances of different tools. We investigate whether linguistic information about a tool (e.g., its geometry, common uses) can help control policies adapt faster to new tools for a given task. We obtain diverse descriptions of various tools in natural language and use pre-trained language models to generate their feature representations. We then perform language-conditioned meta-learning to learn policies that can efficiently adapt to new tools given their corresponding text descriptions. Our results demonstrate that combining linguistic information and meta-learning significantly accelerates tool learning in several manipulation tasks including pushing, lifting, sweeping, and hammering.

**摘要:** 稳健且通用的工具操作需要了解不同工具的属性和启示。我们调查有关工具的语言信息（例如，其几何形状、常见用途）可以帮助控制策略更快地适应特定任务的新工具。我们以自然语言获得各种工具的多样化描述，并使用预先训练的语言模型来生成它们的特征表示。然后，我们执行语言条件化的元学习，以学习可以有效地适应新工具的策略，给定相应的文本描述。我们的结果表明，将语言信息和元学习相结合可以显着加速推动、举起、扫掠和敲打等多个操纵任务中的工具学习。

**[Paper URL](https://proceedings.mlr.press/v205/ren23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ren23a/ren23a.pdf)** 

# Adapting Neural Models with Sequential Monte Carlo Dropout
**题目:** 通过顺序蒙特卡罗辍学调整神经模型

**作者:** Pamela Carreno, Dana Kulic, Michael Burke

**Abstract:** The ability to adapt to changing environments and settings is essential for robots acting in dynamic and unstructured environments or working alongside humans with varied abilities or preferences. This work introduces an extremely simple and effective approach to adapting neural models in response to changing settings, without requiring any specialised meta-learning strategies. We first train a standard network using dropout, which is analogous to learning an ensemble of predictive models or distribution over predictions. At run-time, we use a particle filter to maintain a distribution over dropout masks to adapt the neural model to changing settings in an online manner. Experimental results show improved performance in control problems requiring both online and look-ahead prediction, and showcase the interpretability of the inferred masks in a human behaviour modelling task for drone tele-operation.

**摘要:** 适应不断变化的环境和设置的能力对于在动态和非结构化环境中行动或与具有不同能力或偏好的人类一起工作的机器人至关重要。这项工作引入了一种极其简单有效的方法来适应神经模型以响应不断变化的设置，而不需要任何专门的元学习策略。我们首先使用dropout训练标准网络，这类似于学习预测模型或预测分布的集合。在运行时，我们使用粒子过滤器来维持辍学面具上的分布，以使神经模型适应在线方式变化的设置。实验结果表明，在需要在线和前瞻预测的控制问题中，性能得到了改善，并展示了无人机远程操作人类行为建模任务中推断出的面具的可解释性。

**[Paper URL](https://proceedings.mlr.press/v205/carreno23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/carreno23a/carreno23a.pdf)** 

# Do we use the Right Measure? Challenges in Evaluating Reward Learning Algorithms
**题目:** 我们使用正确的措施吗？评估奖励学习算法的挑战

**作者:** Nils Wilde, Javier Alonso-Mora

**Abstract:** Reward learning is a highly active area of research in human-robot interaction (HRI), allowing a broad range of users to specify complex robot behaviour. Experiments with simulated user input play a major role in the development and evaluation of reward learning algorithms due to the availability of a ground truth. In this paper, we review measures for evaluating reward learning algorithms used in HRI, most of which fall into two classes. In a theoretical worst case analysis and several examples, we show that both classes of measures can fail to effectively indicate how good the learned robot behaviour is. Thus, our work contributes to the characterization of sim-to-real gaps of reward learning in HRI.

**摘要:** 奖励学习是人与机器人交互（HRI）中一个高度活跃的研究领域，允许广泛的用户指定复杂的机器人行为。由于地面事实的可用性，模拟用户输入的实验在奖励学习算法的开发和评估中发挥着重要作用。在本文中，我们回顾了HRI中使用的奖励学习算法的评估措施，其中大部分分为两类。在理论上最坏情况分析和几个例子中，我们表明这两类措施都无法有效地表明习得的机器人行为有多好。因此，我们的工作有助于描述HRI中奖励学习的简单与真实差距。

**[Paper URL](https://proceedings.mlr.press/v205/wilde23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wilde23a/wilde23a.pdf)** 

# Bayesian Object Models for Robotic Interaction with Differentiable Probabilistic Programming
**题目:** 基于可微概率规划的机器人交互的Bayesian对象模型

**作者:** Krishna Murthy Jatavallabhula, Miles Macklin, Dieter Fox, Animesh Garg, Fabio Ramos

**Abstract:** A hallmark of human intelligence is the ability to build rich mental models of previously unseen objects from very few interactions. To achieve true, continuous autonomy, robots too must possess this ability. Importantly, to integrate with the probabilistic robotics software stack, such models must encapsulate the uncertainty (resulting from noisy dynamics and observation models) in a prescriptive manner. We present Bayesian Object Models (BOMs): generative (probabilistic) models that encode both the structural and kinodynamic attributes of an object. BOMs are implemented in the form of a differentiable probabilistic program that models latent scene structure, object dynamics, and observation models. This allows for efficient and automated Bayesian inference – samples (object trajectories) drawn from the BOM are compared with a small set of real-world observations and used to compute a likelihood function. Our model comprises a differentiable tree structure sampler and a differentiable physics engine, enabling gradient computation through this likelihood function. This enables gradient-based Bayesian inference to efficiently update the distributional parameters of our model. BOMs outperform several recent approaches, including differentiable physics-based, gradient-free, and neural inference schemes. Further information at: https://bayesianobjects.github.io/

**摘要:** 人类智能的一个标志是，能够通过极少的互动，为以前未见过的物体建立丰富的心理模型。要实现真正的、持续的自主性，机器人也必须拥有这种能力。重要的是，为了与概率机器人软件堆栈集成，这些模型必须以规定的方式封装不确定性(由噪声动力学和观测模型产生)。我们提出了贝叶斯对象模型(BOM)：既编码对象的结构属性又编码对象的运动学属性的生成性(概率)模型。BOM是以可区分概率程序的形式实现的，该程序对潜在的场景结构、对象动力学和观测模型进行建模。这允许高效和自动的贝叶斯推断-从BOM中提取的样本(对象轨迹)与一小部分真实世界的观察结果进行比较，并用于计算似然函数。我们的模型包括一个可微树结构采样器和一个可微物理引擎，通过该似然函数实现梯度计算。这使得基于梯度的贝叶斯推理能够有效地更新模型的分布参数。BOM的表现优于最近的几种方法，包括基于可微物理的、无梯度的和神经推理方案。欲了解更多信息，请访问：https://bayesianobjects.github.io/。

**[Paper URL](https://proceedings.mlr.press/v205/jatavallabhula23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/jatavallabhula23a/jatavallabhula23a.pdf)** 

# Deep Projective Rotation Estimation through Relative Supervision
**题目:** 通过相对监督的深度投影旋转估计

**作者:** Brian Okorn, Chuer Pan, Martial Hebert, David Held

**Abstract:** Orientation estimation is the core to a variety of vision and robotics tasks such as camera and object pose estimation. Deep learning has offered a way to develop image-based orientation estimators; however, such estimators often require training on a large labeled dataset, which can be time-intensive to collect. In this work, we explore whether self-supervised learning from unlabeled data can be used to alleviate this issue.  Specifically, we assume access to estimates of the relative orientation between neighboring poses, such that can be obtained via a local alignment method. While self-supervised learning has been used successfully for translational object keypoints, in this work, we show that naively applying relative supervision to the rotational group $SO(3)$ will often fail to converge due to the non-convexity of the rotational space. To tackle this challenge, we propose a new algorithm for self-supervised orientation estimation which utilizes Modified Rodrigues Parameters to stereographically project the closed manifold of $SO(3)$ to the open manifold of $\mathbb{R}^{3}$, allowing the optimization to be done in an open Euclidean space. We empirically validate the benefits of the proposed algorithm for rotational averaging problem in two settings: (1) direct optimization on rotation parameters, and (2) optimization of parameters of a convolutional neural network that predicts object orientations from images. In both settings, we demonstrate that our proposed algorithm is able to converge to a consistent relative orientation frame much faster than algorithms that purely operate in the $SO(3)$ space. Additional information can be found at https://sites.google.com/view/deep-projective-rotation.

**摘要:** 方位估计是摄像机和物体姿态估计等各种视觉和机器人任务的核心。深度学习提供了一种开发基于图像的方位估计器的方法；然而，这种估计器通常需要在大量标记的数据集上进行训练，而收集这些数据集可能是时间密集型的。在这项工作中，我们探索是否可以使用来自未标记数据的自我监督学习来缓解这个问题。具体地说，我们假设可以访问相邻姿势之间的相对方向的估计，这样可以通过局部对齐方法获得。虽然自监督学习已经成功地应用于平移对象关键点，但在这项工作中，我们证明了幼稚地将相对监督应用于旋转群$SO(3)$往往由于旋转空间的非凸性而无法收敛。为了应对这一挑战，我们提出了一种新的自监督方向估计算法，该算法利用修改的Rodrigue参数将闭合流形$so(3)$立体投影到开流形$mathbb{R}^{3}$，从而允许在开放的欧氏空间中进行优化。我们在两种情况下对该算法的有效性进行了实验验证：(1)直接优化旋转参数；(2)优化卷积神经网络的参数，从图像中预测目标的方向。在这两种情况下，我们证明了我们提出的算法比单纯在$SO(3)$空间中运行的算法能够更快地收敛到一致的相对方向框架。欲了解更多信息，请登录：https://sites.google.com/view/deep-projective-rotation.。

**[Paper URL](https://proceedings.mlr.press/v205/okorn23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/okorn23a/okorn23a.pdf)** 

# Learning Markerless Robot-Depth Camera Calibration and End-Effector Pose Estimation
**题目:** 学习无标记机器人深度摄像机校准和末端效应器姿势估计

**作者:** Bugra Can Sefercik, Baris Akgun

**Abstract:** Traditional approaches to extrinsic calibration use fiducial markers and learning-based approaches rely heavily on simulation data. In this work, we present a learning-based markerless extrinsic calibration system that uses a depth camera and does not rely on simulation data. We learn models for end-effector (EE) segmentation, single-frame rotation prediction and keypoint detection, from automatically generated real-world data. We use a transformation trick to get EE pose estimates from rotation predictions and a matching algorithm to get EE pose estimates from keypoint predictions. We further utilize the iterative closest point algorithm, multiple-frames, filtering and outlier detection to increase calibration robustness. Our evaluations with training data from multiple camera poses and test data from previously unseen poses give sub-centimeter and sub-deciradian average calibration and pose estimation errors. We also show that a carefully selected single training pose gives comparable results.

**摘要:** 传统的外部校准方法使用基准标记，而基于学习的方法严重依赖于模拟数据。在这项工作中，我们提出了一种基于学习的无标记外部校准系统，该系统使用深度相机，不依赖于模拟数据。我们从自动生成的真实世界数据中学习末端效应器(EE)分割、单帧旋转预测和关键点检测的模型。我们使用变换技巧从旋转预测中获得EE姿势估计，并使用匹配算法从关键点预测中获得EE姿势估计。我们进一步利用迭代最近点算法、多帧、滤波和离群点检测来增强校准的健壮性。我们对来自多个摄像机姿势的训练数据和来自以前未见过的姿势的测试数据进行了评估，得到了亚厘米和亚分贝的平均校准和姿势估计误差。我们还表明，精心选择的单一训练姿势可以给出类似的结果。

**[Paper URL](https://proceedings.mlr.press/v205/sefercik23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/sefercik23a/sefercik23a.pdf)** 

# Visuotactile Affordances for Cloth Manipulation with Local Control
**题目:** 具有局部控制的布料操纵的视觉功能

**作者:** Neha Sunil, Shaoxiong Wang, Yu She, Edward Adelson, Alberto Rodriguez Garcia

**Abstract:** Cloth in the real world is often crumpled, self-occluded, or folded in on itself such that key regions, such as corners, are not directly graspable, making manipulation difficult. We propose a system that leverages visual and tactile perception to unfold the cloth via grasping and sliding on edges. Doing so, the robot is able to grasp two adjacent corners, enabling subsequent manipulation tasks like folding or hanging. We develop tactile perception networks that classify whether an edge is grasped and estimate the pose of the edge. We use the edge classification network to supervise a visuotactile edge grasp affordance network that can grasp edges with a 90% success rate. Once an edge is grasped, we demonstrate that the robot can slide along the cloth to the adjacent corner using tactile pose estimation/control in real time.

**摘要:** 现实世界中的布料通常会起皱、自遮挡或折叠，使得角落等关键区域无法直接抓取，从而使得操纵变得困难。我们提出了一种利用视觉和触觉感知的系统，通过抓住边缘和滑动来展开布料。这样，机器人能够抓住两个相邻的角，从而能够执行折叠或悬挂等后续操纵任务。我们开发了触觉感知网络，可以对边缘是否被抓住进行分类并估计边缘的姿态。我们使用边缘分类网络来监督可视化的边缘抓取启示网络，该网络可以以90%的成功率抓取边缘。一旦抓住边缘，我们证明机器人可以使用实时触觉姿态估计/控制沿着布料滑动到邻近的角落。

**[Paper URL](https://proceedings.mlr.press/v205/sunil23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/sunil23a/sunil23a.pdf)** 

# Is Anyone There? Learning a Planner Contingent on Perceptual Uncertainty
**题目:** 有人在吗？学习计划者取决于感知的不确定性

**作者:** Charles Packer, Nicholas Rhinehart, Rowan Thomas McAllister, Matthew A. Wright, Xin Wang, Jeff He, Sergey Levine, Joseph E. Gonzalez

**Abstract:** Robots in complex multi-agent environments should reason about the intentions of observed and currently unobserved agents. In this paper, we present a new learning-based method for prediction and planning in complex multi-agent environments where the states of the other agents are partially-observed. Our approach, Active Visual Planning (AVP), uses high-dimensional observations to learn a flow-based generative model of multi-agent joint trajectories, including unobserved agents that may be revealed in the near future, depending on the robot’s actions. Our predictive model is implemented using deep neural networks that map raw observations to future detection and pose trajectories and is learned entirely offline using a dataset of recorded observations (not ground-truth states). Once learned, our predictive model can be used for contingency planning over the potential existence, intentions, and positions of unobserved agents. We demonstrate the effectiveness of AVP on a set of autonomous driving environments inspired by real-world scenarios that require reasoning about the existence of other unobserved agents for safe and efficient driving. In these environments, AVP achieves optimal closed-loop performance, while methods that do not reason about potential unobserved agents exhibit either overconfident or underconfident behavior.

**摘要:** 在复杂的多智能体环境中，机器人应该对观察到的和当前未观察到的智能体的意图进行推理。在本文中，我们提出了一种新的基于学习的预测和规划方法，用于复杂的多智能体环境中，其中其他智能体的状态是部分观察的。我们的方法，主动视觉规划(AVP)，使用高维观察来学习基于流的多智能体关节轨迹的生成模型，包括可能在不久的将来被揭示的未观察到的智能体，这取决于机器人的动作。我们的预测模型是使用深度神经网络实现的，该网络将原始观测映射到未来的检测和姿势轨迹，并使用记录的观测数据集(而不是地面真实状态)完全离线学习。一旦学习，我们的预测模型就可以用于对未被观察到的特工的潜在存在、意图和位置进行应急计划。我们在一组受现实世界场景启发的自动驾驶环境中展示了AVP的有效性，这些场景需要推理其他未被观察到的代理的存在，以实现安全和高效的驾驶。在这些环境中，AVP实现了最佳的闭环性能，而不对潜在的未被观察到的代理进行推理的方法则表现出过度自信或缺乏自信的行为。

**[Paper URL](https://proceedings.mlr.press/v205/packer23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/packer23a/packer23a.pdf)** 

# Touching a NeRF: Leveraging Neural Radiance Fields for Tactile Sensory Data Generation
**题目:** 触摸NeRF：利用神经辐射场进行触觉感觉数据生成

**作者:** Shaohong Zhong, Alessandro Albini, Oiwi Parker Jones, Perla Maiolino, Ingmar Posner

**Abstract:** Tactile perception is key for robotics applications such as manipulation. However, tactile data collection is time-consuming, especially when compared to vision. This limits the use of the tactile modality in machine learning solutions in robotics. In this paper, we propose a generative model to simulate realistic tactile sensory data for use in downstream tasks. Starting with easily-obtained camera images, we train Neural Radiance Fields (NeRF) for objects of interest. We then use NeRF-rendered RGB-D images as inputs to a conditional Generative Adversarial Network model (cGAN) to generate tactile images from desired orientations. We evaluate the generated data quantitatively using the Structural Similarity Index and Mean Squared Error metrics, and also using a tactile classification task both in simulation and in the real world. Results show that by augmenting a manually collected dataset, the generated data is able to increase classification accuracy by around 10%. In addition, we demonstrate that our model is able to transfer from one tactile sensor to another with a small fine-tuning dataset.

**摘要:** 触觉感知是机器人应用(如操纵)的关键。然而，触觉数据收集是耗时的，特别是与视觉相比。这限制了触觉通道在机器人学机器学习解决方案中的使用。在这篇文章中，我们提出了一个产生式模型来模拟真实的触觉感觉数据，用于下游任务。从容易获得的摄像机图像开始，我们为感兴趣的对象训练神经辐射场(NERF)。然后，我们使用NERF渲染的RGB-D图像作为条件生成对抗性网络模型(CGAN)的输入，以从期望的方向生成触觉图像。我们使用结构相似性指数和均方误差度量对生成的数据进行定量评估，并在模拟和真实世界中使用触觉分类任务。结果表明，通过扩充人工采集的数据集，生成的数据能够将分类准确率提高约10%。此外，我们证明了我们的模型能够通过一个小的微调数据集从一个触觉传感器转移到另一个触觉传感器。

**[Paper URL](https://proceedings.mlr.press/v205/zhong23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zhong23a/zhong23a.pdf)** 

# HTRON: Efficient Outdoor Navigation with Sparse Rewards via  Heavy Tailed Adaptive Reinforce Algorithm
**题目:** HTRON：通过重尾自适应强化算法实现稀疏奖励的高效户外导航

**作者:** Kasun Weerakoon, Souradip Chakraborty, Nare Karapetyan, Adarsh Jagan Sathyamoorthy, Amrit Bedi, Dinesh Manocha

**Abstract:** We present a novel approach to improve the performance of deep reinforcement learning (DRL) based outdoor robot navigation systems. Most, existing DRL methods are based on carefully designed dense reward functions that learn the efficient behavior in an environment. We circumvent this issue by working only with sparse rewards (which are easy to design) and propose a novel adaptive Heavy-Tailed Reinforce algorithm for Outdoor Navigation called HTRON. Our main idea is to utilize heavy-tailed policy parametrizations which implicitly induce exploration in sparse reward settings. We evaluate the performance of HTRON against Reinforce, PPO, and TRPO algorithms in three different outdoor scenarios: goal-reaching, obstacle avoidance, and uneven terrain navigation. We observe average an increase of 34.41% in terms of success rate, a 15.15% decrease in the average time steps taken to reach the goal, and a 24.9% decrease in the elevation cost compared to the navigation policies obtained by the other methods. Further, we demonstrate that our algorithm can be transferred directly into a Clearpath Husky robot to perform outdoor terrain navigation in real-world scenarios.

**摘要:** 提出了一种改进基于深度强化学习(DRL)的室外机器人导航系统性能的新方法。大多数现有的DRL方法都是基于精心设计的密集奖励函数，这些函数学习环境中的有效行为。为了避免这个问题，我们提出了一种新颖的户外导航自适应重尾增强算法HTRON。我们的主要思想是利用重尾策略参数化，它隐含地在稀疏奖励环境中诱导探索。我们评估了HTRON在三种不同的户外场景中的性能：到达目标、避障和不平坦的地形导航。我们观察到，与其他方法获得的导航策略相比，平均成功率提高了34.41%，达到目标所需的平均时间步长减少了15.15%，高程成本降低了24.9%。进一步，我们证明了我们的算法可以直接转移到ClearPath赫斯基机器人上，在真实场景中执行室外地形导航。

**[Paper URL](https://proceedings.mlr.press/v205/weerakoon23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/weerakoon23a/weerakoon23a.pdf)** 

# Planning with Spatial-Temporal Abstraction from Point Clouds for Deformable Object Manipulation
**题目:** 利用点云的时空抽象进行可变形物体操纵的规划

**作者:** Xingyu Lin, Carl Qi, Yunchu Zhang, Zhiao Huang, Katerina Fragkiadaki, Yunzhu Li, Chuang Gan, David Held

**Abstract:** Effective planning of long-horizon deformable object manipulation requires suitable abstractions at both the spatial and temporal levels. Previous methods typically either focus on short-horizon tasks or make strong assumptions that full-state information is available, which prevents their use on deformable objects. In this paper, we propose PlAnning with Spatial-Temporal Abstraction (PASTA), which incorporates both spatial abstraction (reasoning about objects and their relations to each other) and temporal abstraction (reasoning over skills instead of low-level actions). Our framework maps high-dimension 3D observations such as point clouds into a set of latent vectors and plans over skill sequences on top of the latent set representation. We show that our method can effectively perform  challenging sequential deformable object manipulation tasks in the real world, which require combining multiple tool-use skills such as cutting with a knife, pushing with a pusher, and spreading dough with a roller. Additional materials can be found at our project website: https://sites.google.com/view/pasta-plan.

**摘要:** 长视距可变形物体操纵的有效规划需要在空间和时间两个层次上进行适当的抽象。以前的方法通常要么专注于短期任务，要么强烈假设完整的状态信息是可用的，这阻止了它们在可变形对象上的使用。在本文中，我们提出了时空抽象规划(PAPA)，它结合了空间抽象(关于对象及其相互关系的推理)和时间抽象(基于技能的推理而不是低层动作)。我们的框架将高维3D观测(如点云)映射到一组潜在向量中，并在潜在集合表示的基础上规划技能序列。我们证明了我们的方法可以有效地执行现实世界中具有挑战性的顺序可变形对象操纵任务，这些任务需要结合多种工具使用技能，如用刀切割、用推进器推、用滚子摊面团。更多材料可在我们的项目网站上找到：https://sites.google.com/view/pasta-plan.

**[Paper URL](https://proceedings.mlr.press/v205/lin23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/lin23b/lin23b.pdf)** 

# Don’t Start From Scratch: Leveraging Prior Data to Automate Robotic Reinforcement Learning
**题目:** 不要从头开始：利用先前数据自动化机器人强化学习

**作者:** Homer Rich Walke, Jonathan Heewon Yang, Albert Yu, Aviral Kumar, Jędrzej Orbik, Avi Singh, Sergey Levine

**Abstract:** Reinforcement learning (RL) algorithms hold the promise of enabling autonomous skill acquisition for robotic systems. However, in practice, real-world robotic RL typically requires time consuming data collection and frequent human intervention to reset the environment. Moreover, robotic policies learned with RL often fail when deployed beyond the carefully controlled setting in which they were learned. In this work, we study how these challenges of real-world robotic learning can all be tackled by effective utilization of diverse offline datasets collected from previously seen tasks. When faced with a new task, our system adapts previously learned skills to quickly learn to both perform the new task and return the environment to an initial state, effectively performing its own environment reset. Our empirical results demonstrate that incorporating prior data into robotic reinforcement learning enables autonomous learning, substantially improves sample-efficiency of learning, and enables better generalization.

**摘要:** 强化学习(RL)算法有望实现机器人系统的自主技能获取。然而，在实践中，真实世界的机器人RL通常需要耗时的数据收集和频繁的人工干预来重置环境。此外，使用RL学习的机器人策略在部署到学习它们的精心控制的设置之外时通常会失败。在这项工作中，我们研究了如何通过有效利用从以前看到的任务中收集的各种离线数据集来解决现实世界机器人学习的这些挑战。当面对新的任务时，我们的系统会适应以前学到的技能，快速学习执行新任务并将环境恢复到初始状态，有效地执行自己的环境重置。我们的实验结果表明，在机器人强化学习中加入先验数据可以实现自主学习，显著提高学习的样本效率，并实现更好的泛化。

**[Paper URL](https://proceedings.mlr.press/v205/walke23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/walke23a/walke23a.pdf)** 

# LaRa: Latents and Rays for Multi-Camera Bird’s-Eye-View Semantic Segmentation
**题目:** LaRa：多摄像机鸟瞰语义分割的潜在和光线

**作者:** Florent Bartoccioni, Eloi Zablocki, Andrei Bursuc, Patrick Perez, Matthieu Cord, Karteek Alahari

**Abstract:** Recent works in autonomous driving have widely adopted the bird’seye-view (BEV) semantic map as an intermediate representation of the world. Online prediction of these BEV maps involves non-trivial operations such as multi-camera data extraction as well as fusion and projection into a common topview grid. This is usually done with error-prone geometric operations (e.g., homography or back-projection from monocular depth estimation) or expensive direct dense mapping between image pixels and pixels in BEV (e.g., with MLP or attention). In this work, we present ‘LaRa’, an efficient encoder-decoder, transformer-based model for vehicle semantic segmentation from multiple cameras. Our approach uses a system of cross-attention to aggregate information over multiple sensors into a compact, yet rich, collection of latent representations. These latent representations, after being processed by a series of selfattention blocks, are then reprojected with a second cross-attention in the BEV space. We demonstrate that our model outperforms the best previous works using transformers on nuScenes. The code and trained models are available at https://github.com/valeoai/LaRa.

**摘要:** 近年来，自动驾驶领域的研究已广泛采用鸟瞰(BEV)语义地图作为世界的中间表示。这些BEV地图的在线预测涉及到非常重要的操作，如多摄像机数据提取以及融合和投影到公共俯视网格中。这通常通过容易出错的几何运算(例如，单应性或来自单目深度估计的反投影)或在图像像素与BEV中的像素之间昂贵的直接密集映射(例如，使用MLP或注意)来完成。在这项工作中，我们提出了一种高效的编解码器，基于变换的多摄像机车辆语义分割模型‘LARA’。我们的方法使用交叉注意系统来将多个传感器上的信息聚合到一个紧凑但丰富的潜在表征集合中。这些潜在的表征被一系列的自我注意块处理后，在BEV空间进行第二次交叉注意的重新投射。我们演示了我们的模型比以前在nuScenes上使用转换器的最好的工作要好。代码和经过训练的模型可在https://github.com/valeoai/LaRa.上找到

**[Paper URL](https://proceedings.mlr.press/v205/bartoccioni23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/bartoccioni23a/bartoccioni23a.pdf)** 

# Leveraging Fully Observable Policies for Learning under Partial Observability
**题目:** 利用完全可观察的策略进行部分可观察性下的学习

**作者:** Hai Huu Nguyen, Andrea Baisero, Dian Wang, Christopher Amato, Robert Platt

**Abstract:** Reinforcement learning in partially observable domains is challenging due to the lack of observable state information. Thankfully, learning offline in a simulator with such state information is often possible. In particular, we propose a method for partially observable reinforcement learning that uses a fully observable policy (which we call a \emph{state expert}) during training to improve performance. Based on Soft Actor-Critic (SAC), our agent balances performing actions similar to the state expert and getting high returns under partial observability. Our approach can leverage the fully-observable policy for exploration and parts of the domain that are fully observable while still being able to learn under partial observability. On six robotics domains, our method outperforms pure imitation, pure reinforcement learning, the sequential or parallel combination of both types, and a recent state-of-the-art method in the same setting. A successful policy transfer to a physical robot in a manipulation task from pixels shows our approach’s practicality in learning interesting policies under partial observability.

**摘要:** 由于缺乏可观测状态信息，在部分可观测领域中的强化学习是具有挑战性的。谢天谢地，在模拟器中使用这种状态信息进行离线学习通常是可能的。特别地，我们提出了一种部分可观测强化学习方法，该方法在训练过程中使用完全可观测策略(我们称之为状态专家)来提高性能。在软行为者-批评者(SAC)的基础上，我们的代理在执行类似于状态专家的动作和在部分可观测性下获得高回报之间进行权衡。我们的方法可以利用完全可观察的策略来探索和部分可观察的领域，同时仍然能够在部分可观察的情况下学习。在六个机器人领域，我们的方法优于纯模仿、纯强化学习、这两种类型的顺序或并行组合，以及在相同设置下的最新方法。在从像素到物理机器人的操作任务中，策略的成功转移表明了该方法在学习部分可观测性下的有趣策略方面的实用性。

**[Paper URL](https://proceedings.mlr.press/v205/nguyen23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/nguyen23a/nguyen23a.pdf)** 

# Modularity through Attention: Efficient Training and Transfer of Language-Conditioned Policies for Robot Manipulation
**题目:** 通过注意力实现模块化：机器人操纵的受影响条件政策的有效训练和转移

**作者:** Yifan Zhou, Shubham Sonawani, Mariano Phielipp, Simon Stepputtis, Heni Amor

**Abstract:** Language-conditioned policies allow robots to interpret and execute human instructions. Learning such policies requires a substantial investment with regards to time and compute resources. Still, the resulting controllers are highly device-specific and cannot easily be transferred to a robot with different morphology, capability, appearance or dynamics. In this paper, we propose a sample-efficient approach for training language-conditioned manipulation policies that allows for rapid transfer across different types of robots. By introducing a novel method, namely Hierarchical Modularity, and adopting supervised attention across multiple sub-modules, we bridge the divide between modular and end-to-end learning and enable the reuse of functional building blocks. In both simulated and real world robot manipulation experiments, we demonstrate that our method outperforms the current state-of-the-art methods and can transfer policies across 4 different robots in a sample-efficient manner. Finally, we show that the functionality of learned sub-modules is maintained beyond the training process and can be used to introspect the robot decision-making process.

**摘要:** 受语言制约的策略允许机器人解释和执行人类的指令。学习此类策略需要在时间和计算资源方面进行大量投资。尽管如此，得到的控制器是高度特定于设备的，不容易转移到具有不同形态、能力、外观或动力学的机器人上。在本文中，我们提出了一种样本高效的方法来训练语言条件操作策略，允许在不同类型的机器人之间快速转移。通过引入一种新的方法，即分层模块化，并在多个子模块之间采用有监督的注意，我们在模块化和端到端学习之间架起了一座桥梁，实现了功能构建块的重用。在模拟和真实的机器人操作实验中，我们证明了我们的方法比目前最先进的方法性能更好，并且能够以样本高效的方式在4个不同的机器人之间传递策略。最后，我们证明了学习的子模块的功能在训练过程之外保持不变，并且可以用来反思机器人的决策过程。

**[Paper URL](https://proceedings.mlr.press/v205/zhou23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zhou23b/zhou23b.pdf)** 

# PI-QT-Opt: Predictive Information Improves Multi-Task Robotic Reinforcement Learning at Scale
**题目:** PI-QT-Opt：预测信息大规模改善多任务机器人强化学习

**作者:** Kuang-Huei Lee, Ted Xiao, Adrian Li, Paul Wohlhart, Ian Fischer, Yao Lu

**Abstract:** The predictive information, the mutual information between the past and future, has been shown to be a useful representation learning auxiliary loss for training reinforcement learning agents, as the ability to model what will happen next is critical to success on many control tasks. While existing studies are largely restricted to training specialist agents on single-task settings in simulation, in this work, we study modeling the predictive information for robotic agents and its importance for general-purpose agents that are trained to master a large repertoire of diverse skills from large amounts of data. Specifically, we introduce Predictive Information QT-Opt (PI-QT-Opt), a QT-Opt agent augmented with an auxiliary loss that learns representations of the predictive information to solve up to 297 vision-based robot manipulation tasks in simulation and the real world with a single set of parameters. We demonstrate that modeling the predictive information significantly improves success rates on the training tasks and leads to better zero-shot transfer to unseen novel tasks. Finally, we evaluate PI-QT-Opt on real robots, achieving substantial and consistent improvement over QT-Opt in multiple experimental settings of varying environments, skills, and multi-task configurations.

**摘要:** 预测信息，即过去和未来之间的相互信息，已经被证明是训练强化学习代理的有用的表征学习辅助损失，因为对接下来发生的事情进行建模的能力对于许多控制任务的成功至关重要。虽然现有的研究在很大程度上仅限于在仿真中的单任务环境下培训专家代理，但在这项工作中，我们研究了机器人代理的预测信息建模及其对通用代理的重要性，这些代理经过训练，可以从大量数据中掌握大量不同的技能。具体地说，我们引入了预测信息Qt-opt(PI-Qt-opt)，这是一种增加了辅助损失的Qt-opt代理，它学习预测信息的表示，以单组参数在模拟和真实世界中解决多达297个基于视觉的机器人操作任务。我们证明，对预测信息建模显著提高了训练任务的成功率，并导致更好的零射击转移到看不见的新任务。最后，我们在真实机器人上对PI-QT-OPT进行了评估，在不同环境、技能和多任务配置的多个实验环境下，取得了比QT-OPT显著且一致的改进。

**[Paper URL](https://proceedings.mlr.press/v205/lee23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/lee23a/lee23a.pdf)** 

# Learning Model Predictive Controllers with Real-Time Attention for Real-World Navigation
**题目:** 具有实时关注力的学习模型预测控制器用于现实世界导航

**作者:** Xuesu Xiao, Tingnan Zhang, Krzysztof Marcin Choromanski, Tsang-Wei Edward Lee, Anthony Francis, Jake Varley, Stephen Tu, Sumeet Singh, Peng Xu, Fei Xia, Sven Mikael Persson, Dmitry Kalashnikov, Leila Takayama, Roy Frostig, Jie Tan, Carolina Parada, Vikas Sindhwani

**Abstract:** Despite decades of research, existing navigation systems still face real-world challenges when deployed in the wild, e.g., in cluttered home environments or in human-occupied public spaces. To address this, we present a new class of implicit control policies combining the benefits of imitation learning with the robust handling of system constraints from Model Predictive Control (MPC). Our approach, called Performer-MPC, uses a learned cost function parameterized by vision context embeddings provided by Performers—a low-rank implicit-attention Transformer. We jointly train the cost function and construct the controller relying on it, effectively solving end-to-end the corresponding bi-level optimization problem. We show that the resulting policy improves standard MPC performance by leveraging a few expert demonstrations of the desired navigation behavior in different challenging real-world scenarios. Compared with a standard MPC policy, Performer-MPC achieves >40% better goal reached in cluttered environments and >65% better on social metrics when navigating around humans.

**摘要:** 尽管经过了几十年的研究，现有的导航系统在野外部署时仍然面临着现实世界的挑战，例如在杂乱的家庭环境中或在人类占据的公共场所。为了解决这一问题，我们提出了一类新的隐式控制策略，结合了模拟学习的好处和对模型预测控制(MPC)中系统约束的稳健处理。我们的方法被称为Performer-MPC，它使用一个学习的代价函数，该代价函数由Performers提供的视觉上下文嵌入来参数化--一个低等级的内隐注意转换器。我们联合训练代价函数，并依赖它构造控制器，有效地解决了相应的端到端双层优化问题。我们表明，所产生的策略通过利用在不同挑战的真实世界场景中所需导航行为的几个专家演示来提高标准MPC的性能。与标准的MPC策略相比，Performer-MPC在杂乱的环境中实现的目标提高了40%以上，在人类周围导航时的社交指标提高了65%以上。

**[Paper URL](https://proceedings.mlr.press/v205/xiao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/xiao23a/xiao23a.pdf)** 

# In-Hand Object Rotation via Rapid Motor Adaptation
**题目:** 通过快速运动适应实现手中物体旋转

**作者:** Haozhi Qi, Ashish Kumar, Roberto Calandra, Yi Ma, Jitendra Malik

**Abstract:** Generalized in-hand manipulation has long been an unsolved challenge of robotics. As a small step towards this grand goal, we demonstrate how to design and learn a simple adaptive controller to achieve in-hand object rotation using only fingertips. The controller is trained entirely in simulation on only cylindrical objects, which then – without any fine-tuning – can be directly deployed to a real robot hand to rotate dozens of objects with diverse sizes, shapes, and weights over the z-axis. This is achieved via rapid online adaptation of the robot’s controller to the object properties using only proprioception history. Furthermore, natural and stable finger gaits automatically emerge from training the control policy via reinforcement learning. Code and more videos are available at https://github.com/HaozhiQi/Hora .

**摘要:** 广义在手操纵长期以来一直是机器人技术尚未解决的挑战。作为朝着这一宏伟目标迈出的一小步，我们演示了如何设计和学习一个简单的自适应控制器，以仅使用指尖实现手中物体旋转。控制器仅在圆柱形物体上进行模拟训练，然后无需任何微调即可直接部署到真正的机器人手上，以在z轴上旋转数十个具有不同大小、形状和重量的物体。这是通过仅使用主体感觉历史将机器人控制器快速在线适应对象属性来实现的。此外，通过强化学习训练控制策略会自动出现自然稳定的手指步态。代码和更多视频可在https://github.com/HaozhiQi/Hora上获取。

**[Paper URL](https://proceedings.mlr.press/v205/qi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)** 

# Learning Sampling Distributions for Model Predictive Control
**题目:** 模型预测控制的学习抽样分布

**作者:** Jacob Sacks, Byron Boots

**Abstract:** Sampling-based methods have become a cornerstone of contemporary approaches to Model Predictive Control (MPC), as they make no restrictions on the differentiability of the dynamics or cost function and are straightforward to parallelize. However, their efficacy is highly dependent on the quality of the sampling distribution itself, which is often assumed to be simple, like a Gaussian. This restriction can result in samples which are far from optimal, leading to poor performance. Recent work has explored improving the performance of MPC by sampling in a learned latent space of controls. However, these methods ultimately perform all MPC parameter updates and warm-starting between time steps in the control space. This requires us to rely on a number of heuristics for generating samples and updating the distribution and may lead to sub-optimal performance. Instead, we propose to carry out all operations in the latent space, allowing us to take full advantage of the learned distribution. Specifically, we frame the learning problem as bi-level optimization and show how to train the controller with backpropagation-through-time. By using a normalizing flow parameterization of the distribution, we can leverage its tractable density to avoid requiring differentiability of the dynamics and cost function. Finally, we evaluate the proposed approach on simulated robotics tasks and demonstrate its ability to surpass the performance of prior methods and scale better with a reduced number of samples.

**摘要:** 基于采样的方法已经成为当代模型预测控制(MPC)方法的基石，因为它们不限制动态或代价函数的可微性，并且直接并行化。然而，它们的有效性高度依赖于抽样分布本身的质量，而抽样分布通常被认为是简单的，如高斯分布。这种限制可能会导致样本远远不是最优的，从而导致性能较差。最近的工作探索了通过在学习的潜在控制空间中采样来提高预测控制的性能。然而，这些方法最终执行控制空间中时间步长之间的所有MPC参数更新和热启动。这需要我们依赖许多启发式算法来生成样本和更新分布，并可能导致次优性能。相反，我们建议在潜在空间中执行所有操作，使我们能够充分利用已知的分布。具体地说，我们将学习问题描述为两级优化，并展示了如何通过时间反向传播来训练控制器。通过使用分布的规格化流动参数，我们可以利用其易处理的密度来避免要求动态和成本函数的可微性。最后，我们在模拟机器人任务上对所提出的方法进行了评估，并证明了它能够在减少样本数量的情况下超越现有方法的性能和更好的伸缩性。

**[Paper URL](https://proceedings.mlr.press/v205/sacks23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/sacks23a/sacks23a.pdf)** 

# Embodied Concept Learner: Self-supervised Learning of Concepts and Mapping through Instruction Following
**题目:** 目标概念学习者：通过遵循指令进行概念自我监督学习和绘图

**作者:** Mingyu Ding, Yan Xu, Zhenfang Chen, David Daniel Cox, Ping Luo, Joshua B. Tenenbaum, Chuang Gan

**Abstract:** Humans, even at a very early age, can learn visual concepts and understand geometry and layout through active interaction with the environment, and generalize their compositions to complete tasks described by natural languages in novel scenes. To mimic such capability, we propose Embodied Concept Learner (ECL) in an interactive 3D environment. Specifically, a robot agent can ground visual concepts, build semantic maps and plan actions to complete tasks by learning purely from human demonstrations and language instructions, without access to ground-truth semantic and depth supervision from simulations. ECL consists of: (i) an instruction parser that translates the natural languages into executable programs; (ii) an embodied concept learner that grounds visual concepts based on language descriptions; (iii) a map constructor that estimates depth and constructs semantic maps by leveraging the learned concepts; and (iv) a program executor with deterministic policies to execute each program. ECL has several appealing benefits thanks to its modularized design. Firstly, it enables the robotic agent to learn semantics and depth unsupervisedly acting like babies, e.g., ground concepts through active interaction and perceive depth by disparities when moving forward. Secondly, ECL is fully transparent and step-by-step interpretable in long-term planning. Thirdly, ECL could be beneficial for the embodied instruction following (EIF), outperforming previous works on the ALFRED benchmark when the semantic label is not provided. Also, the learned concept can be reused for other downstream tasks, such as reasoning of object states.

**摘要:** 人类，甚至在很小的时候，就可以通过与环境的积极互动来学习视觉概念，理解几何和布局，并概括他们的组成，以完成在新奇场景中用自然语言描述的任务。为了模仿这种能力，我们提出了交互式3D环境中的具体化概念学习(ECL)。具体地说，机器人代理可以通过纯粹从人类演示和语言指令中学习，而不需要从模拟中获得地面真实语义和深度监督，来接地视觉概念、构建语义地图和计划完成任务的行动。ECL包括：(I)将自然语言翻译成可执行程序的指令解析器；(Ii)基于语言描述使视觉概念落地的具体化概念学习者；(Iii)通过利用所学习的概念来估计深度并构建语义图的地图构造器；以及(Iv)具有确定性策略以执行每个程序的程序执行器。由于它的模块化设计，ECL有几个吸引人的好处。首先，它使机器人智能体能够像婴儿一样不受监督地学习语义和深度，例如，通过主动交互来学习基本概念，并在前进时通过差异感知深度。其次，ECL在长期规划中是完全透明和循序渐进的。第三，在没有提供语义标签的情况下，ECL可能有利于具体化指令遵循(EIF)，优于之前在Alfred基准测试中的工作。此外，学习的概念还可以被重用于其他下游任务，如对象状态的推理。

**[Paper URL](https://proceedings.mlr.press/v205/ding23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ding23b/ding23b.pdf)** 

# Learning Multi-Object Dynamics with Compositional Neural Radiance Fields
**题目:** 使用合成神经辐射场学习多对象动力学

**作者:** Danny Driess, Zhiao Huang, Yunzhu Li, Russ Tedrake, Marc Toussaint

**Abstract:** We present a method to learn compositional multi-object dynamics models from image observations based on implicit object encoders, Neural Radiance Fields (NeRFs), and graph neural networks. NeRFs have become a popular choice for representing scenes due to their strong 3D prior. However, most NeRF approaches are trained on a single scene, representing the whole scene with a global model, making generalization to novel scenes, containing different numbers of objects, challenging. Instead, we present a compositional, object-centric auto-encoder framework that maps multiple views of the scene to a set of latent vectors representing each object separately. The latent vectors parameterize individual NeRFs from which the scene can be reconstructed. Based on those latent vectors, we train a graph neural network dynamics model in the latent space to achieve compositionality for dynamics prediction. A key feature of our approach is that the latent vectors are forced to encode 3D information through the NeRF decoder, which enables us to incorporate structural priors in learning the dynamics models, making long-term predictions more stable compared to several baselines. Simulated and real world experiments show that our method can model and learn the dynamics of compositional scenes including rigid and deformable objects. Video: https://dannydriess.github.io/compnerfdyn/

**摘要:** 提出了一种基于隐式目标编码器、神经辐射场(NERF)和图神经网络的图像合成多目标动力学模型学习方法。由于其强大的3D先验能力，NERF已成为表示场景的流行选择。然而，大多数NERF方法都是在单个场景上训练的，用一个全局模型来表示整个场景，这使得对包含不同数量对象的新场景的泛化具有挑战性。相反，我们提出了一个构图的、以对象为中心的自动编码器框架，该框架将场景的多个视图映射到一组分别表示每个对象的潜在向量。潜在向量将各个神经网络参数化，从这些神经网络可以重建场景。基于这些潜在向量，我们在潜在空间中训练一个图神经网络动力学模型，以达到动态预测的组合性。我们方法的一个关键特征是，通过NERF解码器迫使潜在向量对3D信息进行编码，这使得我们能够在学习动力学模型时加入结构先验，使得长期预测比几个基线更稳定。仿真实验和真实场景实验表明，该方法能够对包括刚性物体和可变形物体在内的合成场景进行建模和学习。视频：https://dannydriess.github.io/compnerfdyn/

**[Paper URL](https://proceedings.mlr.press/v205/driess23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/driess23a/driess23a.pdf)** 

# Inner Monologue: Embodied Reasoning through Planning with Language Models
**题目:** 内心独白：通过语言模型规划进行有序推理

**作者:** Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Tomas Jackson, Noah Brown, Linda Luu, Sergey Levine, Karol Hausman, brian ichter

**Abstract:** Recent works have shown how the reasoning capabilities of Large Language Models (LLMs) can be applied to domains beyond natural language processing, such as planning and interaction for robots. These embodied problems require an agent to understand many semantic aspects of the world: the repertoire of skills available, how these skills influence the world, and how changes to the world map back to the language. LLMs planning in embodied environments need to consider not just what skills to do, but also how and when to do them - answers that change over time in response to the agent’s own choices. In this work, we investigate to what extent LLMs used in such embodied contexts can reason over sources of feedback provided through natural language, without any additional training. We propose that by leveraging environment feedback, LLMs are able to form an inner monologue that allows them to more richly process and plan in robotic control scenarios. We investigate a variety of sources of feedback, such as success detection, scene description, and human interaction. We find that closed-loop language feedback significantly improves high level instruction completion on three domains, including simulated and real table top rearrangement tasks and long-horizon mobile manipulation tasks in a kitchen environment in the real world.

**摘要:** 最近的工作表明，大型语言模型(LLM)的推理能力可以应用到自然语言处理以外的领域，如机器人的规划和交互。这些具体的问题需要代理人理解世界的许多语义方面：可用的技能宝库，这些技能如何影响世界，以及世界的变化如何映射回语言。在具体化环境中规划LLM不仅需要考虑要做什么技能，还需要考虑如何以及何时做这些技能--答案会随着时间的推移而变化，以响应代理人自己的选择。在这项工作中，我们调查了在这样的具体化语境中使用的LLM在多大程度上可以推理通过自然语言提供的反馈来源，而不需要任何额外的培训。我们认为，通过利用环境反馈，LLM能够形成内部独白，使它们能够在机器人控制场景中更丰富地处理和规划。我们调查了各种反馈来源，例如成功检测、场景描述和人类交互。我们发现，闭环语言反馈显著提高了三个领域的高级指令完成，包括模拟和真实桌面重排任务以及现实世界厨房环境中的长视距移动操作任务。

**[Paper URL](https://proceedings.mlr.press/v205/huang23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/huang23c/huang23c.pdf)** 

# TAX-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation
**题目:** TAX-Pose：机器人操纵的特定任务交叉姿势估计

**作者:** Chuer Pan, Brian Okorn, Harry Zhang, Ben Eisner, David Held

**Abstract:** How do we imbue robots with the ability to efficiently manipulate unseen objects and transfer relevant skills based on demonstrations? End-to-end learning methods often fail to generalize to novel objects or unseen configurations. Instead, we focus on the task-specific pose relationship between relevant parts of interacting objects. We conjecture that this relationship is a generalizable notion of a manipulation task that can transfer to new objects in the same category; examples include the relationship between the pose of a pan relative to an oven or the pose of a mug relative to a mug rack. We call this task-specific pose relationship “cross-pose” and provide a mathematical definition of this concept. We propose a vision-based system that learns to estimate the cross-pose between two objects for a given manipulation task using learned cross-object correspondences. The estimated cross-pose is then used to guide a downstream motion planner to manipulate the objects into the desired pose relationship (placing a pan into the oven or the mug onto the mug rack). We demonstrate our method’s capability to generalize to unseen objects, in some cases after training on only 10 demonstrations in the real world. Results show that our system achieves state-of-the-art performance in both simulated and real-world experiments across a number of tasks. Supplementary information and videos can be found at https://sites.google.com/view/tax-pose/home.

**摘要:** 我们如何向机器人灌输高效操作看不见的物体的能力，并基于演示传授相关技能？端到端的学习方法往往不能概括为新的对象或看不见的配置。相反，我们关注的是交互对象的相关部分之间特定于任务的姿势关系。我们推测，这种关系是可以转移到同一类别的新对象上的操作任务的一般概念；例如，平底锅相对于烤箱的姿势或杯子相对于马克杯架的姿势之间的关系。我们把这种特定于任务的姿势关系称为“交叉姿势”，并给出了这个概念的数学定义。我们提出了一个基于视觉的系统，它使用学习到的跨对象对应关系来学习估计给定操作任务中两个对象之间的交叉姿态。然后，估计的交叉姿势被用于指导下游运动规划者将对象操纵到期望的姿势关系(将平底锅放入烤箱或将杯子放到马克杯架上)。我们展示了我们的方法对看不见的对象进行泛化的能力，在某些情况下，我们只在现实世界中进行了10次演示培训。结果表明，我们的系统在多个任务的模拟和真实世界实验中都获得了最先进的性能。有关补充信息和视频，请访问https://sites.google.com/view/tax-pose/home.。

**[Paper URL](https://proceedings.mlr.press/v205/pan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/pan23a/pan23a.pdf)** 

# SSL-Lanes: Self-Supervised Learning for Motion Forecasting in Autonomous Driving
**题目:** SSL-Lanes：自动驾驶运动预测的自我监督学习

**作者:** Prarthana Bhattacharyya, Chengjie Huang, Krzysztof Czarnecki

**Abstract:** Self-supervised learning (SSL) is an emerging technique that has been successfully employed to train convolutional neural networks (CNNs) and graph neural networks (GNNs) for more transferable, generalizable, and robust representation learning. However its potential in motion forecasting for autonomous driving has rarely been explored. In this study, we report the first systematic exploration and assessment of incorporating self-supervision into motion forecasting. We first propose to investigate four novel self-supervised learning tasks for motion forecasting with theoretical rationale and quantitative and qualitative comparisons on the challenging large-scale Argoverse dataset. Secondly, we point out that our auxiliary SSL-based learning setup not only outperforms forecasting methods which use transformers, complicated fusion mechanisms and sophisticated online dense goal candidate optimization algorithms in terms of performance accuracy, but also has low inference time and architectural complexity. Lastly, we conduct several experiments to understand why SSL improves motion forecasting.

**摘要:** 自监督学习是一种新兴的学习技术，已被成功地用于训练卷积神经网络(CNN)和图神经网络(GNN)，以获得更好的可传递性、泛化和健壮的表示学习。然而，它在自动驾驶运动预测方面的潜力却很少被发掘。在这项研究中，我们报告了首次将自我监督纳入运动预测的系统探索和评估。我们首先提出了四个新的自监督学习任务，用于运动预测，在具有挑战性的大规模ArgoVerse数据集上进行了理论基础和定量和定性比较。其次，我们指出，我们的基于SSL语言的辅助学习系统不仅在性能上优于使用变压器、复杂的融合机制和复杂的在线密集目标候选优化算法的预测方法，而且具有较低的推理时间和体系结构复杂性。最后，我们进行了几个实验，以了解为什么SSL提高了运动预测。

**[Paper URL](https://proceedings.mlr.press/v205/bhattacharyya23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/bhattacharyya23a/bhattacharyya23a.pdf)** 

# VIRDO++: Real-World, Visuo-tactile Dynamics and Perception of Deformable Objects
**题目:** VIRDO++：现实世界、视觉触觉动态和可变形物体的感知

**作者:** Youngsun Wi, Andy Zeng, Pete Florence, Nima Fazeli

**Abstract:** Deformable objects manipulation can benefit from representations that seamlessly integrate vision and touch while handling occlusions. In this work, we present a novel approach for, and real-world demonstration of, multimodal visuo-tactile state-estimation and dynamics prediction for deformable objects. Our approach, VIRDO++, builds on recent progress in multimodal neural implicit representations for deformable object state-estimation (VIRDO) via a new formulation for deformation dynamics and a complementary state-estimation algorithm that (i) maintains a belief over deformations, and (ii) enables practical real-world application by removing the need for privileged contact information. In the context of two real-world robotic tasks, we show: (i) high-fidelity cross-modal state-estimation and prediction of deformable objects from partial visuo-tactile feedback, and (ii) generalization to unseen objects and contact formations.

**摘要:** 可变形对象操纵可以受益于无缝集成视觉和触摸同时处理遮挡的表示。在这项工作中，我们提出了一种新的方法，用于可变形对象的多模式视觉触觉状态估计和动态预测，并在现实世界中进行了演示。我们的方法VIRDO++基于可变形物体状态估计（VIRDO）的多模式神经隐式表示的最新进展，通过变形动力学的新公式和补充状态估计算法，该算法（i）保持对变形的信念，并且（ii）通过消除对特权联系信息的需求来实现实际的现实世界应用。在两个现实世界的机器人任务的背景下，我们展示了：（i）根据部分视觉触觉反馈对可变形对象的高保真跨模式状态估计和预测，以及（ii）对不可见对象和接触形成的概括。

**[Paper URL](https://proceedings.mlr.press/v205/wi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wi23a/wi23a.pdf)** 

# Detecting Incorrect Visual Demonstrations for Improved Policy Learning
**题目:** 检测不正确的视觉演示以改善政策学习

**作者:** Mostafa Hussein, Momotaz Begum

**Abstract:** Learning tasks only from raw video demonstrations is the current state of the art in robotics visual imitation learning research. The implicit assumption here is that all video demonstrations show an optimal/sub-optimal way of performing the task. What if that is not true? What if one or more videos show a wrong way of executing the task? A task policy learned from such incorrect demonstrations can be potentially unsafe for robots and humans. It is therefore important to analyze the video demonstrations for correctness before handing them over to the policy learning algorithm. This is a challenging task, especially due to the very large state space. This paper proposes a framework to autonomously detect incorrect video demonstrations of sequential tasks consisting of several sub-tasks. We analyze the demonstration pool to identify video(s) for which task-features follow a ‘disruptive’ sequence. We analyze entropy to measure this disruption and – through solving a minmax problem – assign poor weights to incorrect videos. We evaluated the framework with two real-world video datasets: our custom-designed Tea-Making with a YuMi robot and the publicly available 50-Salads. Experimental results show the effectiveness of the proposed framework in detecting incorrect video demonstrations even when they make up 40% of the demonstration set. We also show that various state-of-the-art imitation learning algorithms learn a better policy when incorrect demonstrations are discarded from the training pool.

**摘要:** 仅从原始视频演示中学习任务是机器人学视觉模拟学习研究的现状。这里隐含的假设是，所有视频演示都显示了执行任务的最佳/次优方式。如果这不是真的呢？如果一段或多段视频显示了执行任务的错误方式，该怎么办？从这种不正确的演示中学习到的任务策略可能对机器人和人类都不安全。因此，在将视频演示移交给策略学习算法之前，分析视频演示的正确性非常重要。这是一项具有挑战性的任务，特别是考虑到非常大的状态空间。提出了一种自动检测由多个子任务组成的顺序任务的错误视频演示的框架。我们对演示池进行分析，以确定视频(S)的任务特征遵循的是“颠覆性”序列。我们通过分析熵来衡量这种干扰，并通过解决最小最大值问题来为不正确的视频分配不好的权重。我们使用两个真实世界的视频数据集对该框架进行了评估：我们使用YuMi机器人定制设计的泡茶和公开销售的50个沙拉。实验结果表明，即使错误视频占演示集的40%，该框架也能有效地检测出错误视频。我们还表明，当不正确的演示从训练池中丢弃时，各种最先进的模仿学习算法学习更好的策略。

**[Paper URL](https://proceedings.mlr.press/v205/hussein23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/hussein23a/hussein23a.pdf)** 

# Concept Learning for Interpretable Multi-Agent Reinforcement Learning
**题目:** 可解释多智能体强化学习的概念学习

**作者:** Renos Zabounidis, Joseph Campbell, Simon Stepputtis, Dana Hughes, Katia P. Sycara

**Abstract:** Multi-agent robotic systems are increasingly operating in real-world environments in close proximity to humans, yet are largely controlled by policy models with inscrutable deep neural network representations. We introduce a method for incorporating interpretable concepts from a domain expert into models trained through multi-agent reinforcement learning, by requiring the model to first predict such concepts then utilize them for decision making. This allows an expert to both reason about the resulting concept policy models in terms of these high-level concepts at run-time, as well as intervene and correct mispredictions to improve performance. We show that this yields improved interpretability and training stability, with benefits to policy performance and sample efficiency in a simulated and real-world cooperative-competitive multi-agent game.

**摘要:** 多智能体机器人系统越来越多地在现实世界环境中与人类非常接近，但在很大程度上由具有难以理解的深度神经网络表示的政策模型控制。我们引入了一种方法，将来自领域专家的可解释概念整合到通过多智能体强化学习训练的模型中，要求模型首先预测此类概念，然后利用它们进行决策。这使得专家能够在运行时根据这些高级概念对产生的概念策略模型进行推理，以及干预和纠正错误预测以提高性能。我们表明，这可以提高可解释性和训练稳定性，并对模拟和现实世界的合作竞争多智能体游戏中的政策性能和样本效率有利。

**[Paper URL](https://proceedings.mlr.press/v205/zabounidis23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zabounidis23a/zabounidis23a.pdf)** 

# Latent Plans for Task-Agnostic Offline Reinforcement Learning
**题目:** 任务不可知离线强化学习的潜在计划

**作者:** Erick Rosete-Beas, Oier Mees, Gabriel Kalweit, Joschka Boedecker, Wolfram Burgard

**Abstract:** Everyday tasks of long-horizon and comprising a sequence of multiple implicit subtasks still impose a major challenge in offline robot control. While a number of prior methods aimed to address this setting with variants of imitation and offline reinforcement learning, the learned behavior is typically narrow and often struggles to reach configurable long-horizon goals. As both paradigms have complementary strengths and weaknesses, we propose a novel hierarchical approach that combines the strengths of both methods to learn task-agnostic long-horizon policies from high-dimensional camera observations. Concretely, we combine a low-level policy that learns latent skills via imitation learning and a high-level policy learned from offline reinforcement learning for skill-chaining the latent behavior priors. Experiments in various simulated and real robot control tasks show that our formulation enables producing previously unseen combinations of skills to reach temporally extended goals by “stitching” together latent skills through goal chaining with an order-of-magnitude improvement in performance upon state-of-the-art baselines. We even learn one multi-task visuomotor policy for 25 distinct manipulation tasks in the real world which outperforms both imitation learning and offline reinforcement learning techniques.

**摘要:** 由一系列隐含的多个子任务组成的长时间的日常任务仍然是离线机器人控制的一大挑战。虽然之前的一些方法旨在通过模仿和离线强化学习的变体来解决这种情况，但学习的行为通常是狭隘的，往往难以达到可配置的长期目标。由于这两种范式各有优缺点，我们提出了一种新的分层方法，它结合了两种方法的优点，可以从高维摄像机观测中学习与任务无关的长期策略。具体地说，我们结合了通过模仿学习学习潜在技能的低层策略和通过离线强化学习学习的高级策略来技能链潜在行为先验。在各种模拟和真实的机器人控制任务中的实验表明，我们的公式能够产生以前未曾见过的技能组合，通过目标链将潜在技能“缝合”在一起，并在最先进的基线上实现数量级的性能改进，从而实现暂时扩展的目标。我们甚至学习了一个多任务视觉运动策略，用于现实世界中25个不同的操作任务，这比模仿学习和离线强化学习技术都要好。

**[Paper URL](https://proceedings.mlr.press/v205/rosete-beas23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/rosete-beas23a/rosete-beas23a.pdf)** 

# Manipulation via Membranes: High-Resolution and Highly Deformable Tactile Sensing and Control
**题目:** 通过膜操纵：高分辨率和高度可变形的触觉传感和控制

**作者:** Miquel Oller, Mireia Planas i Lisbona, Dmitry Berenson, Nima Fazeli

**Abstract:** Collocated tactile sensing is a fundamental enabling technology for dexterous manipulation. However, deformable sensors introduce complex dynamics between the robot, grasped object, and environment that must be considered for fine manipulation. Here, we propose a method to learn soft tactile sensor membrane dynamics that accounts for sensor deformations caused by the physical interaction between the grasped object and environment. Our method combines the perceived 3D geometry of the membrane with proprioceptive reaction wrenches to predict future deformations conditioned on robot action. Grasped object poses are recovered from membrane geometry and reaction wrenches, decoupling interaction dynamics from the tactile observation model. We benchmark our approach on two real-world contact-rich tasks: drawing with a grasped marker and in-hand pivoting. Our results suggest that explicitly modeling membrane dynamics achieves better task performance and generalization to unseen objects than baselines.

**摘要:** 并置触觉感知是灵巧操作的基本使能技术。然而，可变形传感器在机器人、被抓取物体和环境之间引入了复杂的动力学，必须考虑才能进行精细操作。在这里，我们提出了一种学习软触觉传感器膜动力学的方法，该方法考虑了被抓取物体与环境之间的物理作用引起的传感器变形。我们的方法将感知到的膜的3D几何形状与本体感觉反应扳手相结合，以预测未来在机器人动作条件下的变形。抓取的物体姿势从膜几何和反应扳手中恢复，将交互动力学从触觉观察模型中分离出来。我们的方法以两个真实世界中接触丰富的任务为基准：用手持记号笔绘图和手把手旋转。我们的结果表明，与基线相比，显式建模膜动力学实现了更好的任务性能和对不可见对象的泛化。

**[Paper URL](https://proceedings.mlr.press/v205/oller23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/oller23a/oller23a.pdf)** 

# Real-Time Generation of Time-Optimal Quadrotor Trajectories with Semi-Supervised Seq2Seq Learning
**题目:** 利用半监督Seq 2 Seq学习实时生成时间最优的四螺旋桨轨迹

**作者:** Gilhyun Ryou, Ezra Tal, Sertac Karaman

**Abstract:** Generating time-optimal quadrotor trajectories is challenging due to the complex dynamics of high-speed, agile flight. In this paper, we propose a data-driven method for real-time time-optimal trajectory generation that is suitable for complicated system models. We utilize a temporal deep neural network with sequence-to-sequence learning to find the optimal trajectories for sequences of a variable number of waypoints. The model is efficiently trained in a semi-supervised manner by combining supervised pretraining using a minimum-snap baseline method with Bayesian optimization and reinforcement learning. Compared to the baseline method, the trained model generates up to 20 % faster trajectories at an order of magnitude less computational cost. The optimized trajectories are evaluated in simulation and real-world flight experiments, where the improvement is further demonstrated.

**摘要:** 由于高速、敏捷飞行的复杂动态，生成时间最优的四螺旋桨轨迹具有挑战性。本文提出了一种适用于复杂系统模型的实时时间最优轨迹生成的数据驱动方法。我们利用具有序列到序列学习的时间深度神经网络来为可变数量路点的序列找到最佳轨迹。通过将使用最小快照基线方法的监督预训练与Bayesian优化和强化学习相结合，以半监督方式有效训练该模型。与基线方法相比，经过训练的模型生成的轨迹速度可提高20%，计算成本则降低了一个数量级。在模拟和现实世界飞行实验中评估优化的轨迹，并进一步证明了改进。

**[Paper URL](https://proceedings.mlr.press/v205/ryou23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ryou23a/ryou23a.pdf)** 

# Sim-to-Real via Sim-to-Seg: End-to-end Off-road Autonomous Driving Without Real Data
**题目:** 通过Sim-to-Seg Sim-to-Real：无需真实数据的端到端越野自动驾驶

**作者:** John So, Amber Xie, Sunggoo Jung, Jeffrey Edlund, Rohan Thakker, Ali-akbar Agha-mohammadi, Pieter Abbeel, Stephen James

**Abstract:** Autonomous driving is complex, requiring sophisticated 3D scene understanding, localization, mapping, and control. Rather than explicitly modelling and fusing each of these components, we instead consider an end-to-end approach via reinforcement learning (RL). However, collecting exploration driving data in the real world is impractical and dangerous. While training in simulation and deploying visual sim-to-real techniques has worked well for robot manipulation, deploying beyond controlled workspace viewpoints remains a challenge. In this paper, we address this challenge by presenting Sim2Seg, a re-imagining of RCAN that crosses the visual reality gap for off-road autonomous driving, without using any real-world data. This is done by learning to translate randomized simulation images into simulated segmentation and depth maps, subsequently enabling real-world images to also be translated. This allows us to train an end-to-end RL policy in simulation, and directly deploy in the real-world. Our approach, which can be trained in 48 hours on 1 GPU, can perform equally as well as a classical perception and control stack that took thousands of engineering hours over several months to build. We hope this work motivates future end-to-end autonomous driving research.

**摘要:** 自动驾驶很复杂，需要复杂的3D场景理解、定位、地图绘制和控制。我们没有显式地对这些组件进行建模和融合，而是考虑通过强化学习(RL)实现端到端的方法。然而，在现实世界中收集勘探驾驶数据是不切实际的，也是危险的。虽然模拟培训和部署可视化模拟现实技术对机器人操作很有效，但在受控工作空间视点之外进行部署仍然是一个挑战。在本文中，我们通过提供Sim2Seg来解决这一挑战，这是一种RCAN的重新想象，它跨越了越野自动驾驶的视觉现实差距，而不使用任何真实世界的数据。这是通过学习将随机模拟图像转换为模拟分割和深度图来实现的，随后也可以转换真实世界的图像。这使得我们可以在模拟中训练端到端的RL策略，并直接在现实世界中部署。我们的方法可以在48小时内在1个GPU上进行训练，可以像传统的感知和控制堆栈一样执行，后者在几个月的时间里花费了数千个工程小时来构建。我们希望这项工作能激励未来端到端的自动驾驶研究。

**[Paper URL](https://proceedings.mlr.press/v205/so23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/so23a/so23a.pdf)** 

# Learning Road Scene-level Representations via Semantic Region Prediction
**题目:** 通过语义区域预测学习道路场景级表示

**作者:** Zihao Xiao, Alan Yuille, Yi-Ting Chen

**Abstract:** In this work, we tackle two vital tasks in automated driving systems, i.e., driver intent prediction and risk object identification from egocentric images. Mainly, we investigate the question: what would be good road scene-level representations for these two tasks? We contend that a scene-level representation must capture higher-level semantic and geometric representations of traffic scenes around ego-vehicle while performing actions to their destinations. To this end, we introduce the representation of semantic regions, which are areas where ego-vehicles visit while taking an afforded action (e.g., left-turn at 4-way intersections). We propose to learn scene-level representations via a novel semantic region prediction task and an automatic semantic region labeling algorithm. Extensive evaluations are conducted on the HDD and nuScenes datasets, and the learned representations lead to state-of-the-art performance for driver intention prediction and risk object identification.

**摘要:** 在这项工作中，我们解决了自动驾驶系统中的两项重要任务，即根据以自我为中心的图像进行驾驶员意图预测和风险对象识别。我们主要研究这样一个问题：对于这两项任务来说，什么是良好的道路场景级表示？我们认为，场景级表示必须在向目的地执行动作时捕获自我车辆周围交通场景的更高级语义和几何表示。为此，我们引入了语义区域的表示，这些区域是自我车辆在采取规定动作时访问的区域（例如，在四向路口左转）。我们建议通过一种新颖的语义区域预测任务和一种自动语义区域标记算法来学习场景级表示。对硬盘和nuScenes数据集进行了广泛的评估，学习到的表示为驾驶员意图预测和风险对象识别提供了最先进的性能。

**[Paper URL](https://proceedings.mlr.press/v205/xiao23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/xiao23b/xiao23b.pdf)** 

# GenLoco: Generalized Locomotion Controllers for Quadrupedal Robots
**题目:** GenLoco：用于四足机器人的广义运动控制器

**作者:** Gilbert Feng, Hongbo Zhang, Zhongyu Li, Xue Bin Peng, Bhuvan Basireddy, Linzhu Yue, ZHITAO SONG, Lizhi Yang, Yunhui Liu, Koushil Sreenath, Sergey Levine

**Abstract:** Recent years have seen a surge in commercially-available and affordable quadrupedal robots, with many of these platforms being actively used in research and industry. As the availability of legged robots grows, so does the need for controllers that enable these robots to perform useful skills. However, most learning-based frameworks for controller development focus on training robot-specific controllers, a process that needs to be repeated for every new robot. In this work, we introduce a framework for training generalized locomotion (GenLoco) controllers for quadrupedal robots. Our framework synthesizes general-purpose locomotion controllers that can be deployed on a large variety of quadrupedal robots with similar morphologies. We present a simple but effective morphology randomization method that procedurally generates a diverse set of simulated robots for training. We show that by training a controller on this large set of simulated robots, our models acquire more general control strategies that can be directly transferred to novel simulated and real-world robots with diverse morphologies, which were not observed during training.

**摘要:** 近年来，市场上可以买到、价格实惠的四足机器人激增，其中许多平台被积极用于研究和工业。随着腿部机器人的普及，对控制器的需求也越来越大，以使这些机器人能够执行有用的技能。然而，大多数基于学习的控制器开发框架专注于培训特定于机器人的控制器，这一过程需要对每个新机器人重复。在这项工作中，我们介绍了一种用于训练四足机器人广义运动(GenLoco)控制器的框架。我们的框架综合了通用的运动控制器，可以部署在各种形状相似的四足机器人上。我们提出了一种简单而有效的形态随机化方法，该方法可以程序化地生成一组用于训练的不同的模拟机器人。我们表明，通过在大量模拟机器人上训练控制器，我们的模型获得了更通用的控制策略，这些控制策略可以直接转移到具有不同形态的新型模拟和真实世界机器人上，而这些控制策略在训练过程中没有观察到。

**[Paper URL](https://proceedings.mlr.press/v205/feng23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/feng23a/feng23a.pdf)** 

# Towards Long-Tailed 3D Detection
**题目:** 走向长尾3D检测

**作者:** Neehar Peri, Achal Dave, Deva Ramanan, Shu Kong

**Abstract:** Contemporary autonomous vehicle (AV) benchmarks have advanced techniques for training 3D detectors, particularly on large-scale lidar data. Surprisingly, although semantic class labels naturally follow a long-tailed distribution, contemporary benchmarks focus on only a few common classes (e.g., pedestrian and car) and neglect many rare classes in-the-tail (e.g., debris and stroller). However, AVs must still detect rare classes to ensure safe operation. Moreover, semantic classes are often organized within a hierarchy, e.g., tail classes such as child and construction-worker are arguably subclasses of pedestrian. However, such hierarchical relationships are often ignored, which may lead to misleading estimates of performance and missed opportunities for algorithmic innovation. We address these challenges by formally studying the problem of Long-Tailed 3D Detection (LT3D), which evaluates on all classes, including those in-the-tail. We evaluate and innovate upon popular 3D detection codebases, such as CenterPoint and PointPillars, adapting them for LT3D. We develop hierarchical losses that promote feature sharing across common-vs-rare classes, as well as improved detection metrics that award partial credit to "reasonable" mistakes respecting the hierarchy (e.g., mistaking a child for an adult). Finally, we point out that fine-grained tail class accuracy is particularly improved via multimodal fusion of RGB images with LiDAR; simply put, small fine-grained classes are challenging to identify from sparse (lidar) geometry alone, suggesting that multimodal cues are crucial to long-tailed 3D detection. Our modifications improve accuracy by 5% AP on average for all classes, and dramatically improve AP for rare classes (e.g., stroller AP improves from 3.6 to 31.6).

**摘要:** 现代自动驾驶汽车(AV)基准拥有先进的技术来训练3D探测器，特别是在大规模激光雷达数据上。令人惊讶的是，尽管语义类标签自然遵循长尾分布，但当代基准只关注几个常见的类(例如行人和汽车)，而忽略了许多罕见的尾部类(例如碎片和婴儿车)。然而，AVS仍然必须检测到罕见的类别，以确保安全运行。此外，语义类通常是在层次结构内组织的，例如，诸如孩子和建筑工人之类的尾类可以被证明是行人的子类。然而，这种等级关系经常被忽视，这可能会导致对性能的误导估计，并错失算法创新的机会。我们通过正式研究长尾3D检测(LT3D)问题来解决这些挑战，LT3D对所有类别进行评估，包括尾部中的类别。我们对流行的3D检测代码库进行了评估和创新，如CenterPoint和PointPillars，使其适用于LT3D。我们开发了分层损失，以促进常见与罕见类之间的特征共享，以及改进的检测度量，将部分信用授予尊重分层的“合理”错误(例如，将儿童误认为成人)。最后，我们指出，通过RGB图像与LiDAR的多模式融合，细粒度尾类的准确率得到了特别的提高；简而言之，仅从稀疏(激光雷达)几何形状识别小的细粒度类是具有挑战性的，这表明多模式线索对于长尾3D检测至关重要。我们的改进将所有类别的AP的准确率平均提高了5%，并显著提高了罕见类别的AP(例如，婴儿车AP从3.6提高到31.6)。

**[Paper URL](https://proceedings.mlr.press/v205/peri23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/peri23a/peri23a.pdf)** 

# MIRA: Mental Imagery for Robotic Affordances
**题目:** MIRA：机器人功能的心理意象

**作者:** Yen-Chen Lin, Pete Florence, Andy Zeng, Jonathan T. Barron, Yilun Du, Wei-Chiu Ma, Anthony Simeonov, Alberto Rodriguez Garcia, Phillip Isola

**Abstract:** Humans form mental images of 3D scenes to support counterfactual imagination, planning, and motor control. Our abilities to predict the appearance and affordance of the scene from previously unobserved viewpoints aid us in performing manipulation tasks (e.g., 6-DoF kitting) with a level of ease that is currently out of reach for existing robot learning frameworks. In this work, we aim to build artificial systems that can analogously plan actions on top of imagined images. To this end, we introduce Mental Imagery for Robotic Affordances (MIRA), an action reasoning framework that optimizes actions with novel-view synthesis and affordance prediction in the loop. Given a set of 2D RGB images, MIRA builds a consistent 3D scene representation, through which we synthesize novel orthographic views amenable to pixel-wise affordances prediction for action optimization. We illustrate how this optimization process enables us to generalize to unseen out-of-plane rotations for 6-DoF robotic manipulation tasks given a limited number of demonstrations, paving the way toward machines that autonomously learn to understand the world around them for planning actions.

**摘要:** 人类在脑海中形成3D场景的图像，以支持反现实的想象、规划和运动控制。我们能够从以前未观察到的视点预测场景的外观和启示，这有助于我们以目前现有机器人学习框架无法达到的轻松程度执行操作任务(例如，6-DOF套件)。在这项工作中，我们的目标是建立人工系统，能够在想象的图像上类似地计划行动。为此，我们引入了机器人Affordance的心理意象(MIRA)，这是一个动作推理框架，通过循环中的新颖视图合成和启发性预测来优化动作。在给定一组2D RGB图像的情况下，Mira构建了一致的3D场景表示，通过该表示，我们合成了符合像素级启示预测的新的正射视图，以用于动作优化。我们举例说明了这种优化过程如何使我们能够在有限数量的演示中将6-DOF机器人操纵任务的平面外旋转推广到看不见的位置，从而为自主学习了解周围世界以规划行动的机器铺平了道路。

**[Paper URL](https://proceedings.mlr.press/v205/lin23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/lin23c/lin23c.pdf)** 

# CAtNIPP: Context-Aware Attention-based Network for Informative Path Planning
**题目:** CAtNIPP：用于信息路径规划的上下文感知基于注意力的网络

**作者:** Yuhong Cao, Yizhuo Wang, Apoorva Vashisth, Haolin Fan, Guillaume Adrien Sartoretti

**Abstract:** Informative path planning (IPP) is an NP-hard problem, which aims at planning a path allowing an agent to build an accurate belief about a quantity of interest throughout a given search domain, within constraints on resource budget (e.g., path length for robots with limited battery life). IPP requires frequent online replanning as this belief is updated with every new measurement (i.e., adaptive IPP), while balancing short-term exploitation and longer-term exploration to avoid suboptimal, myopic behaviors. Encouraged by the recent developments in deep reinforcement learning, we introduce CAtNIPP, a fully reactive, neural approach to the adaptive IPP problem. CAtNIPP relies on self-attention for its powerful ability to capture dependencies in data at multiple spatial scales. Specifically, our agent learns to form a context of its belief over the entire domain, which it uses to sequence local movement decisions that optimize short- and longer-term search objectives. We experimentally demonstrate that CAtNIPP significantly outperforms state-of-the-art non-learning IPP solvers in terms of solution quality and computing time once trained, and present experimental results on hardware.

**摘要:** 信息性路径规划(IPP)是一个NP-Hard问题，它的目标是规划一条路径，允许智能体在资源预算(例如，电池寿命有限的机器人的路径长度)的约束下，对给定搜索领域内的兴趣量建立准确的信念。IPP需要频繁的在线重新规划，因为这种信念随着每次新的衡量标准(即适应性IPP)而更新，同时平衡短期开发和长期探索，以避免次优的、短视的行为。受深度强化学习的最新发展的鼓舞，我们引入了CAtNIPP，这是一种用于自适应IPP问题的完全反应性的神经方法。CAtNIPP依赖于自我关注，因为它具有在多个空间尺度上捕获数据依赖关系的强大能力。具体地说，我们的代理学习在整个领域形成其信念的上下文，并使用该上下文来排序局部移动决策，以优化短期和长期搜索目标。实验证明，CAtNIPP在训练后的解质量和计算时间方面明显优于最先进的非学习IPP求解器，并给出了硬件上的实验结果。

**[Paper URL](https://proceedings.mlr.press/v205/cao23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/cao23b/cao23b.pdf)** 

# Learning Diverse and Physically Feasible Dexterous Grasps with Generative Model and Bilevel Optimization
**题目:** 使用生成模型和二层优化学习多样化且物理可行的灵巧抓取

**作者:** Albert Wu, Michelle Guo, Karen Liu

**Abstract:** To fully utilize the versatility of a multi-fingered dexterous robotic hand for executing diverse object grasps, one must consider the rich physical constraints introduced by hand-object interaction and object geometry. We propose an integrative approach of combining a generative model and a bilevel optimization (BO) to plan diverse grasp configurations on novel objects. First, a conditional variational autoencoder trained on merely six YCB objects predicts the finger placement directly from the object point cloud. The prediction is then used to seed a nonconvex BO that solves for a grasp configuration under collision, reachability, wrench closure, and friction constraints. Our method achieved an 86.7% success over 120 real world grasping trials on 20 household objects, including unseen and challenging geometries. Through quantitative empirical evaluations, we confirm that grasp configurations produced by our pipeline are indeed guaranteed to satisfy kinematic and dynamic constraints. A video summary of our results is available at youtu.be/9DTrImbN99I.

**摘要:** 为了充分利用多指灵巧手的多功能性来执行不同的物体抓取，必须考虑手与物体相互作用和物体几何带来的丰富的物理约束。我们提出了一种结合产生式模型和双层优化(BO)的综合方法来规划新对象上的不同抓取构形。首先，只对6个YCB对象训练的条件变分自动编码器直接从对象点云预测手指位置。然后，该预测被用于在碰撞、可达性、扳手闭合和摩擦约束下求解抓取配置的非凸BO的种子。我们的方法在120个真实世界中对20个家用物体的抓取试验中获得了86.7%的成功率，其中包括看不见的和具有挑战性的几何图形。通过定量的经验评估，我们确认我们的管道产生的抓取配置确实保证满足运动学和动力学约束。我们的结果的视频摘要可以在youtu.be/9DTrImbN99I上获得。

**[Paper URL](https://proceedings.mlr.press/v205/wu23b.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wu23b/wu23b.pdf)** 

# Verified Path Following Using Neural Control Lyapunov Functions
**题目:** 使用神经控制李雅普诺夫函数验证路径跟踪

**作者:** Alec Reed, Guillaume O Berger, Sriram Sankaranarayanan, Chris Heckman

**Abstract:** We present a framework that uses control Lyapunov functions (CLFs) to implement provably stable path-following controllers for autonomous mobile platforms. Our approach is based on learning a guaranteed CLF for path following by using recent approaches — combining machine learning with automated theorem proving — to train a neural network feedback law along with a CLF that guarantees stabilization for driving along low-curvature reference paths. We discuss how key properties of the CLF can be exploited to extend the range of  the curvatures for which the stability guarantees remain valid. We then demonstrate that our approach yields a controller that obeys theoretical guarantees in simulation, but also performs well in practice. We show our method is both a verified method of control and better than a common MPC implementation in computation time. Additionally, we implement the controller on-board on a $\frac18$-scale autonomous vehicle testing platform and present results for various robust path following scenarios.

**摘要:** 提出了一种利用控制李雅普诺夫函数(CLF)实现自主移动平台可证明稳定的路径跟踪控制器的框架。我们的方法是基于学习路径跟踪的有保证的CLF，通过使用最新的方法-结合机器学习和自动定理证明-来训练神经网络反馈律以及保证沿着低曲率参考路径行驶的稳定性的CLF。我们讨论了如何利用CLF的关键性质来扩展稳定性保证保持有效的曲率的范围。然后，我们证明了我们的方法产生的控制器在仿真中服从理论保证，但在实践中也表现得很好。我们证明了我们的方法既是一种经过验证的控制方法，而且在计算时间上优于普通的MPC实现。此外，我们还在一个价值18美元的自主车辆测试平台上实现了控制器，并给出了各种鲁棒路径跟踪场景的结果。

**[Paper URL](https://proceedings.mlr.press/v205/reed23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/reed23a/reed23a.pdf)** 

# Task-Relevant Failure Detection for Trajectory Predictors in Autonomous Vehicles
**题目:** 自动驾驶车辆轨迹预测器的任务相关故障检测

**作者:** Alec Farid, Sushant Veer, Boris Ivanovic, Karen Leung, Marco Pavone

**Abstract:** In modern autonomy stacks, prediction modules are paramount to planning motions in the presence of other mobile agents. However, failures in prediction modules can mislead the downstream planner into making unsafe decisions. Indeed, the high uncertainty inherent to the task of trajectory forecasting ensures that such mispredictions occur frequently. Motivated by the need to improve safety of autonomous vehicles without compromising on their performance, we develop a probabilistic run-time monitor that detects when a "harmful" prediction failure occurs, i.e., a task-relevant failure detector. We achieve this by propagating trajectory prediction errors to the planning cost to reason about their impact on the AV. Furthermore, our detector comes equipped with performance measures on the false-positive and the false-negative rate and allows for data-free calibration. In our experiments we compared our detector with various others and found that our detector has the highest area under the receiver operator characteristic curve.

**摘要:** 在现代自主堆栈中，预测模块对于在其他移动代理存在的情况下规划运动至关重要。然而，预测模块中的故障可能会误导下游计划者做出不安全的决定。事实上，轨迹预测任务固有的高度不确定性确保了这种错误预测经常发生。为了提高自动驾驶车辆的安全性，同时又不影响其性能，我们开发了一种概率运行时间监测器，用于检测何时发生有害的预测故障，即与任务相关的故障检测器。我们通过将轨迹预测误差传播到规划成本来推断它们对AV的影响来实现这一点。此外，我们的检测器配备了假阳性和假阴性率的性能衡量标准，并允许无数据校准。在我们的实验中，我们将我们的探测器与其他探测器进行了比较，发现我们的探测器在接收算子特性曲线下的面积最大。

**[Paper URL](https://proceedings.mlr.press/v205/farid23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/farid23a/farid23a.pdf)** 

# Safe Control Under Input Limits with Neural Control Barrier Functions
**题目:** 具有神经控制屏障功能的输入极限下的安全控制

**作者:** Simin Liu, Changliu Liu, John Dolan

**Abstract:** We propose new methods to synthesize control barrier function (CBF) based safe controllers that avoid input saturation, which can cause safety violations. In particular, our method is created for high-dimensional, general nonlinear systems, for which such tools are scarce. We leverage techniques from machine learning, like neural networks and deep learning, to simplify this challenging problem in nonlinear control design. The method consists of a learner-critic architecture, in which the critic gives counterexamples of input saturation and the learner optimizes a neural CBF to eliminate those counterexamples. We provide empirical results on a 10D state, 4D input quadcopter-pendulum system. Our learned CBF avoids input saturation and maintains safety over nearly 100% of trials.

**摘要:** 我们提出了新的方法来合成基于控制屏障函数（CBF）的安全控制器，以避免输入饱和，从而导致安全违规。特别是，我们的方法是为多维、一般非线性系统创建的，而此类工具很少。我们利用神经网络和深度学习等机器学习技术来简化非线性控制设计中的这个具有挑战性的问题。该方法由学习者-评论者架构组成，其中评论者给出输入饱和度的反例，学习者优化神经CBF以消除这些反例。我们提供了10 D状态、4D输入四轴陀螺摆系统的经验结果。我们学到的CBF避免了输入饱和并在近100%的试验中保持安全性。

**[Paper URL](https://proceedings.mlr.press/v205/liu23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/liu23e/liu23e.pdf)** 

# Eliciting Compatible Demonstrations for Multi-Human Imitation Learning
**题目:** 激发多人模仿学习的兼容演示

**作者:** Kanishk Gandhi, Siddharth Karamcheti, Madeline Liao, Dorsa Sadigh

**Abstract:** Imitation learning from human-provided demonstrations is a strong approach for learning policies for robot manipulation. While the ideal dataset for imitation learning is homogenous and low-variance - reflecting a single, optimal method for performing a task - natural human behavior has a great deal of heterogeneity, with several optimal ways to demonstrate a task. This multimodality is inconsequential to human users, with task variations manifesting as subconscious choices; for example, reaching down, then across to grasp an object, versus reaching across, then down. Yet, this mismatch presents a problem for interactive imitation learning, where sequences of users improve on a policy by iteratively collecting new, possibly conflicting demonstrations. To combat this problem of demonstrator incompatibility, this work designs an approach for 1) measuring the compatibility of a new demonstration given a base policy, and 2) actively eliciting more compatible demonstrations from new users. Across two simulation tasks requiring long-horizon, dexterous manipulation and a real-world “food plating” task with a Franka Emika Panda arm, we show that we can both identify incompatible demonstrations via post-hoc filtering, and apply our compatibility measure to actively elicit compatible demonstrations from new users, leading to improved task success rates across simulated and real environments.

**摘要:** 从人类提供的演示中进行模仿学习是学习机器人操作策略的一种强有力的方法。虽然模仿学习的理想数据集是同质和低方差-反映了执行任务的单一、最佳方法-但自然人类行为具有很大的异质性，有几种最佳方式来演示任务。这种多模式对人类用户来说是无关紧要的，任务变化表现为潜意识的选择；例如，向下伸手，然后越过抓住一个物体，而不是伸手越过，然后向下。然而，这种不匹配给交互式模仿学习带来了一个问题，即用户序列通过迭代收集新的、可能相互冲突的演示来改进策略。为了解决演示者不相容的问题，本工作设计了一种方法，用于1)在给定基本策略的情况下测量新演示的兼容性，以及2)积极地从新用户那里引发更多相容的演示。在两个需要长时间、灵活操作的模拟任务和使用Franka Emika Panda手臂的真实世界“食物处理”任务中，我们展示了我们都可以通过后自组织过滤识别不兼容的演示，并应用我们的兼容性措施积极地从新用户那里吸引兼容的演示，从而提高模拟和真实环境中的任务成功率。

**[Paper URL](https://proceedings.mlr.press/v205/gandhi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/gandhi23a/gandhi23a.pdf)** 

# When the Sun Goes Down: Repairing Photometric Losses for All-Day Depth Estimation
**题目:** 当太阳落山时：修复全天深度估计的测光损失

**作者:** Madhu Vankadari, Stuart Golodetz, Sourav Garg, Sangyun Shin, Andrew Markham, Niki Trigoni

**Abstract:** Self-supervised deep learning methods for joint depth and ego-motion estimation can yield accurate trajectories without needing ground-truth training data. However, as they typically use photometric losses, their performance can degrade significantly when the assumptions these losses make (e.g. temporal illumination consistency, a static scene, and the absence of noise and occlusions) are violated. This limits their use for e.g. nighttime sequences, which tend to contain many point light sources (including on dynamic objects) and low signal-to-noise ratio (SNR) in darker image regions. In this paper, we show how to use a combination of three techniques to allow the existing photometric losses to work for both day and nighttime images. First, we introduce a per-pixel neural intensity transformation to compensate for the light changes that occur between successive frames. Second, we predict a per-pixel residual flow map that we use to correct the reprojection correspondences induced by the estimated ego-motion and depth from the networks. And third, we denoise the training images to improve the robustness and accuracy of our approach. These changes allow us to train a single model for both day and nighttime images without needing separate encoders or extra feature networks like existing methods. We perform extensive experiments and ablation studies on the challenging Oxford RobotCar dataset to demonstrate the efficacy of our approach for both day and nighttime sequences.

**摘要:** 用于关节深度和自我运动估计的自监督深度学习方法可以在不需要地面真实训练数据的情况下产生准确的轨迹。但是，由于它们通常使用光度学损失，因此当违反这些损失所做的假设(例如，时间照明一致性、静态场景以及没有噪波和遮挡)时，它们的性能可能会显著下降。这限制了它们在例如夜间序列中的使用，夜间序列往往包含许多点光源(包括在动态对象上)和较暗图像区域中的低信噪比(SNR)。在这篇文章中，我们展示了如何使用三种技术的组合，以允许现有的光度损失同时适用于白天和夜间的图像。首先，我们引入了每个像素的神经强度变换来补偿连续帧之间发生的光线变化。其次，我们预测了每像素余流图，我们使用该图来校正由网络估计的自我运动和深度引起的重投影对应。第三，对训练图像进行去噪处理，提高了方法的稳健性和准确性。这些变化允许我们为白天和夜间图像训练单一模型，而不需要像现有方法那样单独的编码器或额外的特征网络。我们在具有挑战性的牛津RobotCar数据集上进行了广泛的实验和消融研究，以证明我们的方法对白天和夜间序列的有效性。

**[Paper URL](https://proceedings.mlr.press/v205/vankadari23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/vankadari23a/vankadari23a.pdf)** 

# Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes
**题目:** 在杂乱场景中实现规模平衡的6-DoF抓取检测

**作者:** Haoxiang Ma, Di Huang

**Abstract:** In this paper, we focus on the problem of feature learning in the presence of scale imbalance for 6-DoF grasp detection and propose a novel approach to especially address the difficulty in dealing with small-scale samples. A Multi-scale Cylinder Grouping (MsCG) module is presented to enhance local geometry representation by combining multi-scale cylinder features and global context. Moreover, a Scale Balanced Learning (SBL) loss and an Object Balanced Sampling (OBS) strategy are designed, where SBL enlarges the gradients of the samples whose scales are in low frequency by apriori weights while OBS captures more points on small-scale objects with the help of an auxiliary segmentation network. They alleviate the influence of the uneven distribution of grasp scales in training and inference respectively. In addition, Noisy-clean Mix (NcM) data augmentation is introduced to facilitate training, aiming to bridge the domain gap between synthetic and raw scenes in an efficient way by generating more data which mix them into single ones at instance-level. Extensive experiments are conducted on the GraspNet-1Billion benchmark and competitive results are reached with significant gains on small-scale cases. Besides, the performance of real-world grasping highlights its generalization ability.

**摘要:** 本文针对6-DOF抓取检测中存在的尺度不平衡情况下的特征学习问题，提出了一种新的方法来解决小规模样本下的特征学习问题。提出了一种多尺度圆柱体分组(MsCG)模块，通过结合多尺度圆柱体特征和全局上下文增强局部几何表示。此外，设计了尺度平衡学习(SBL)损失和对象平衡采样(OBS)策略，其中SBL通过先验加权扩大尺度在低频段样本的梯度，而OBS通过辅助分割网络捕捉小尺度对象上更多的点。它们分别缓解了训练和推理中掌握尺度分布不均的影响。此外，为了便于训练，引入了噪声-干净混合(NCM)数据增强，旨在通过生成更多的数据来在实例级将合成场景和原始场景混合为单一数据，从而有效地弥合合成场景和原始场景之间的域差距。在GraspNet-10亿基准上进行了广泛的实验，并在小规模案例上获得了具有竞争力的结果。此外，真实世界抓取的表现突出了它的泛化能力。

**[Paper URL](https://proceedings.mlr.press/v205/ma23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/ma23a/ma23a.pdf)** 

# Few-Shot Preference Learning for Human-in-the-Loop RL
**题目:** 人在环RL的少偏好镜头学习

**作者:** Donald Joseph Hejna III, Dorsa Sadigh

**Abstract:** While reinforcement learning (RL) has become a more popular approach for robotics, designing sufficiently informative reward functions for complex tasks has proven to be extremely difficult due their inability to capture human intent and policy exploitation. Preference based RL algorithms seek to overcome these challenges by directly learning reward functions from human feedback. Unfortunately, prior work either requires an unreasonable number of queries implausible for any human to answer or overly restricts the class of reward functions to guarantee the elicitation of the most informative queries, resulting in models that are insufficiently expressive for realistic robotics tasks. Contrary to most works that focus on query selection to \emph{minimize} the amount of data required for learning reward functions, we take an opposite approach: \emph{expanding} the pool of available data by viewing human-in-the-loop RL through the more flexible lens of multi-task learning. Motivated by the success of meta-learning, we pre-train preference models on prior task data and quickly adapt them for new tasks using only a handful of queries. Empirically, we reduce the amount of online feedback needed to train manipulation policies in Meta-World by 20$\times$, and demonstrate the effectiveness of our method on a real Franka Panda Robot. Moreover, this reduction in query-complexity allows us to train robot policies from actual human users. Videos of our results can be found at https://sites.google.com/view/few-shot-preference-rl/home.

**摘要:** 虽然强化学习(RL)已经成为机器人学中一种更受欢迎的方法，但事实证明，为复杂任务设计足够信息的奖励函数是极其困难的，因为它们无法捕捉人类的意图和策略开发。基于偏好的RL算法试图通过直接从人类反馈中学习奖励函数来克服这些挑战。不幸的是，以前的工作要么需要不合理数量的查询，任何人都不可能回答，要么过度限制奖励函数的类别，以保证引出最有信息量的查询，导致模型不足以表达现实的机器人任务。与大多数专注于查询选择以最小化学习奖励函数所需的数据量的工作相反，我们采取了相反的方法：通过更灵活的多任务学习的镜头来查看人在循环中的RL来扩展可用的数据池。在元学习成功的激励下，我们根据以前的任务数据预先训练偏好模型，并只使用少数几个查询就可以快速地使它们适应新的任务。经验性地，我们减少了在元世界中训练操纵策略所需的在线反馈量20美元\倍$，并在真实的Franka Panda机器人上演示了我们的方法的有效性。此外，这种查询复杂性的降低允许我们从实际的人类用户训练机器人策略。我们的研究结果的视频可以在https://sites.google.com/view/few-shot-preference-rl/home.上找到

**[Paper URL](https://proceedings.mlr.press/v205/iii23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/iii23a/iii23a.pdf)** 

# Visuo-Tactile Transformers for Manipulation
**题目:** 用于操纵的视觉触觉变形金刚

**作者:** Yizhou Chen, Mark Van der Merwe, Andrea Sipos, Nima Fazeli

**Abstract:** Learning representations in the joint domain of vision and touch can improve manipulation dexterity, robustness, and sample-complexity by exploiting mutual information and complementary cues. Here, we present Visuo-Tactile Transformers (VTTs), a novel multimodal representation learning approach suited for model-based reinforcement learning and planning. Our approach extends the Visual Transformer to handle visuo-tactile feedback. Specifically, VTT uses tactile feedback together with self and cross-modal attention to build latent heatmap representations that focus attention on important task features in the visual domain. We demonstrate the efficacy of VTT for representation learning with a comparative evaluation against baselines on four simulated robot tasks and one real world block pushing task. We conduct an ablation study over the components of VTT to highlight the importance of cross-modality in representation learning for robotic manipulation.

**摘要:** 视觉和触摸联合领域中的学习表示可以通过利用互信息和补充线索来提高操纵灵活性、鲁棒性和样本复杂性。在这里，我们介绍了视觉触觉变形器（VTT），这是一种新型的多模式表示学习方法，适合基于模型的强化学习和规划。我们的方法扩展了视觉Transformer来处理视觉触觉反馈。具体来说，VTT使用触觉反馈以及自我和跨模式注意力来构建潜在热图表示，将注意力集中在视觉领域中的重要任务特征上。我们通过与四个模拟机器人任务和一个现实世界块推送任务的基线进行比较评估，证明了VTT在表示学习中的功效。我们对VTT的组件进行了一项消融研究，以强调跨模式在机器人操纵的表示学习中的重要性。

**[Paper URL](https://proceedings.mlr.press/v205/chen23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/chen23d/chen23d.pdf)** 

# Offline Reinforcement Learning at Multiple Frequencies
**题目:** 多频率离线强化学习

**作者:** Kaylee Burns, Tianhe Yu, Chelsea Finn, Karol Hausman

**Abstract:** To leverage many sources of offline robot data, robots must grapple with the heterogeneity of such data. In this paper, we focus on one particular aspect of this challenge: learning from offline data collected at different control frequencies. Across labs, the discretization of controllers, sampling rates of sensors, and demands of a task of interest may differ, giving rise to a mixture of frequencies in an aggregated dataset. We study how well offline reinforcement learning (RL) algorithms can accommodate data with a mixture of frequencies during training. We observe that the $Q$-value propagates at different rates for different discretizations, leading to a number of learning challenges for off-the-shelf offline RL algorithms. We present a simple yet effective solution that enforces consistency in the rate of  $Q$-value updates to stabilize learning. By scaling the value of $N$ in $N$-step returns with the discretization size, we effectively balance $Q$-value propagation, leading to more stable convergence. On three simulated robotic control problems, we empirically find that this simple approach significantly outperforms naïve mixing both terms of absolute performance and training stability, while also improving over using only the data from a single control frequency.

**摘要:** 为了利用许多离线机器人数据来源，机器人必须努力解决这些数据的异构性。在本文中，我们关注这一挑战的一个特定方面：从以不同控制频率收集的离线数据中学习。在不同的实验室中，控制器的离散化、传感器的采样率和感兴趣的任务的需求可能会有所不同，从而导致聚合数据集中的频率混合。我们研究了离线强化学习(RL)算法在训练过程中如何适应混合频率的数据。我们观察到，对于不同的离散化，$Q$-值以不同的速率传播，这导致了现成的离线RL算法的一些学习挑战。我们提出了一个简单而有效的解决方案，它强制执行$Q$值更新的一致性，以稳定学习。通过将$N$步回报中的$N$值与离散化大小进行缩放，我们有效地平衡了$Q$值传播，导致了更稳定的收敛。在三个模拟的机器人控制问题上，我们的经验发现，这种简单的方法在绝对性能和训练稳定性方面都明显优于天真混合，同时也比只使用单一控制频率的数据有所改善。

**[Paper URL](https://proceedings.mlr.press/v205/burns23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/burns23a/burns23a.pdf)** 

# Learning the Dynamics of Compliant Tool-Environment Interaction for Visuo-Tactile Contact Servoing
**题目:** 学习视觉-触觉接触伺服的合规工具-环境交互的动力学

**作者:** Mark Van der Merwe, Dmitry Berenson, Nima Fazeli

**Abstract:** Many manipulation tasks require the robot to control the contact between a grasped compliant tool and the environment, e.g. scraping a frying pan with a spatula. However, modeling tool-environment interaction is difficult, especially when the tool is compliant, and the robot cannot be expected to have the full geometry and physical properties (e.g., mass, stiffness, and friction) of all the tools it must use. We propose a framework that learns to predict the effects of a robot’s actions on the contact between the tool and the environment given visuo-tactile perception. Key to our framework is a novel contact feature representation that consists of a binary contact value, the line of contact, and an end-effector wrench. We propose a method to learn the dynamics of these contact features from real world data that does not require predicting the geometry of the compliant tool. We then propose a controller that uses this dynamics model for visuo-tactile contact servoing and show that it is effective at performing scraping tasks with a spatula, even in scenarios where precise contact needs to be made to avoid obstacles.

**摘要:** 许多操作任务需要机器人控制抓取的顺应性工具与环境之间的接触，例如用铲子刮平底锅。然而，建模工具-环境交互是困难的，特别是当工具是柔顺的，并且不能期望机器人具有其必须使用的所有工具的全部几何和物理属性(例如，质量、刚度和摩擦)。我们提出了一个框架，它学习预测机器人的动作对工具与环境之间的接触的影响，并给出视觉-触觉感知。我们框架的关键是一种新的接触特征表示，它由二进制接触值、接触线和末端执行器扳手组成。我们提出了一种方法，可以从真实世界的数据中学习这些接触特征的动态特性，而不需要预测符合要求的工具的几何形状。然后，我们提出了一种控制器，它使用这个动力学模型来进行视觉-触觉接触伺服，并证明了它在使用铲子执行刮擦任务时是有效的，即使在需要进行精确接触以避开障碍物的场景中也是如此。

**[Paper URL](https://proceedings.mlr.press/v205/merwe23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/merwe23a/merwe23a.pdf)** 

# Multi-Robot Scene Completion: Towards Task-Agnostic Collaborative Perception
**题目:** 多机器人场景完成：迈向任务不可知的协作感知

**作者:** Yiming Li, Juexiao Zhang, Dekun Ma, Yue Wang, Chen Feng

**Abstract:** Collaborative perception learns how to share information among multiple robots to perceive the environment better than individually done. Past research on this has been task-specific, such as detection or segmentation. Yet this leads to different information sharing for different tasks, hindering the large-scale deployment of collaborative perception. We propose the first task-agnostic collaborative perception paradigm that learns a single collaboration module in a self-supervised manner for different downstream tasks. This is done by a novel task termed multi-robot scene completion, where each robot learns to effectively share information for reconstructing a complete scene viewed by all robots. Moreover, we propose a spatiotemporal autoencoder (STAR) that amortizes over time the communication cost by spatial sub-sampling and temporal mixing. Extensive experiments validate our method’s effectiveness on scene completion and collaborative perception in autonomous driving scenarios. Our code is available at https://coperception.github.io/star/.

**摘要:** 协作感知学习如何在多个机器人之间共享信息，以比单独完成的更好地感知环境。过去对此的研究是针对特定任务的，例如检测或分割。然而，这导致了不同任务的不同信息共享，阻碍了协同感知的大规模部署。我们提出了第一个与任务无关的协作感知范式，该范式以自我监督的方式为不同的下游任务学习单个协作模块。这是通过一种名为多机器人场景完成的新颖任务来完成的，其中每个机器人学习有效地共享信息，以重建由所有机器人观看的完整场景。此外，我们还提出了一种时空自动编码器(STAR)，它通过空间子采样和时间混合来摊销通信开销。大量实验验证了该方法在自动驾驶场景中的场景完成和协同感知方面的有效性。我们的代码可以在https://coperception.github.io/star/.上找到

**[Paper URL](https://proceedings.mlr.press/v205/li23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/li23e/li23e.pdf)** 

# USHER: Unbiased Sampling for Hindsight Experience Replay
**题目:** USHER：后见之明体验重播的无偏见抽样

**作者:** Liam Schramm, Yunfu Deng, Edgar Granados, Abdeslam Boularias

**Abstract:** Dealing with sparse rewards is a long-standing challenge in reinforcement learning (RL). Hindsight Experience Replay (HER) addresses this problem by reusing failed trajectories for one goal as successful trajectories for another. This allows for both a minimum density of reward and for generalization across multiple goals. However, this strategy is known to result in a biased value function, as the update rule underestimates the likelihood of bad outcomes in a stochastic environment. We propose an asymptotically unbiased importance-sampling-based algorithm to address this problem without sacrificing performance on deterministic environments. We show its effectiveness on a range of robotic systems, including challenging high dimensional stochastic environments.

**摘要:** 处理稀疏奖励是强化学习（RL）中的一个长期挑战。事后诸葛亮体验回放（HER）通过将一个目标的失败轨迹重复使用为另一个目标的成功轨迹来解决这个问题。这既可以实现最低密度的奖励，又可以实现多个目标的概括。然而，众所周知，这种策略会导致有偏差的价值函数，因为更新规则低估了随机环境中不良结果的可能性。我们提出了一种基于渐进无偏重要性抽样的算法来解决这个问题，而不会牺牲确定性环境下的性能。我们展示了它在一系列机器人系统上的有效性，包括具有挑战性的多维随机环境。

**[Paper URL](https://proceedings.mlr.press/v205/schramm23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/schramm23a/schramm23a.pdf)** 

# Fast Lifelong Adaptive Inverse Reinforcement Learning from Demonstrations
**题目:** 来自演示的快速终身自适应反向强化学习

**作者:** Letian Chen, Sravan Jayanthi, Rohan R Paleja, Daniel Martin, Viacheslav Zakharov, Matthew Gombolay

**Abstract:** Learning from Demonstration (LfD) approaches empower end-users to teach robots novel tasks via demonstrations of the desired behaviors, democratizing access to robotics. However, current LfD frameworks are not capable of fast adaptation to heterogeneous human demonstrations nor the large-scale deployment in ubiquitous robotics applications. In this paper, we propose a novel LfD framework, Fast Lifelong Adaptive Inverse Reinforcement learning (FLAIR). Our approach (1) leverages learned strategies to construct policy mixtures for fast adaptation to new demonstrations, allowing for quick end-user personalization, (2) distills common knowledge across demonstrations, achieving accurate task inference; and (3) expands its model only when needed in lifelong deployments, maintaining a concise set of prototypical strategies that can approximate all behaviors via policy mixtures. We empirically validate that FLAIR achieves adaptability (i.e., the robot adapts to heterogeneous, user-specific task preferences), efficiency (i.e., the robot achieves sample-efficient adaptation), and scalability (i.e., the model grows sublinearly with the number of demonstrations while maintaining high performance). FLAIR surpasses benchmarks across three control tasks with an average 57% improvement in policy returns and an average 78% fewer episodes required for demonstration modeling using policy mixtures. Finally, we demonstrate the success of FLAIR in a table tennis task and find users rate FLAIR as having higher task ($p<.05$) and personalization ($p<.05$) performance.

**摘要:** 从演示中学习(LFD)方法使最终用户能够通过演示所需的行为来教授机器人新任务，从而普及对机器人的访问。然而，当前的LFD框架不能快速适应不同种类的人类演示，也不能在无处不在的机器人应用中大规模部署。本文提出了一种新的LFD框架--快速终身自适应逆强化学习(FLAIR)。我们的方法(1)利用学习的策略来构建策略混合以快速适应新的演示，从而允许快速的最终用户个性化；(2)提取演示中的共同知识，实现准确的任务推理；以及(3)仅在终身部署需要时扩展其模型，维护一组简明的原型策略，可以通过策略混合来近似所有行为。我们通过实验验证了FLAIR达到了适应性(即机器人适应不同的、用户特定的任务偏好)、效率(即机器人实现了样本效率的适应)和可扩展性(即，模型在保持高性能的同时，随着演示次数的增加而亚线性增长)。FLAIR在三个控制任务上超过基准，策略回报平均提高57%，使用策略组合进行演示建模所需的插曲平均减少78%。最后，我们展示了FLAIR在乒乓球任务中的成功，发现用户认为FLAIR具有更高的任务($p<0.05$)和个性化($p<0.05$)表现。

**[Paper URL](https://proceedings.mlr.press/v205/chen23e.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/chen23e/chen23e.pdf)** 

# Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics
**题目:** 剩余技能策略：学习可适应的基于技能的动作空间，用于机器人强化学习

**作者:** Krishan Rana, Ming Xu, Brendan Tidd, Michael Milford, Niko Suenderhauf

**Abstract:** Skill-based reinforcement learning (RL) has emerged as a promising strategy to leverage prior knowledge for accelerated robot learning. Skills are typically extracted from expert demonstrations and are embedded into a latent space from which they can be sampled as actions by a high-level RL agent. However, this \textit{skill space} is expansive, and not all skills are relevant for a given robot state, making exploration difficult. Furthermore, the downstream RL agent is limited to learning structurally similar tasks to those used to construct the skill space. We firstly propose accelerating exploration in the skill space using state-conditioned generative models to directly bias the high-level agent towards only \textit{sampling} skills relevant to a given state based on prior experience. Next, we propose a low-level residual policy for fine-grained \textit{skill adaptation} enabling downstream RL agents to adapt to unseen task variations. Finally, we validate our approach across four challenging manipulation tasks that differ from those used to build the skill space, demonstrating our ability to learn across task variations while significantly accelerating exploration, outperforming prior works.

**摘要:** 基于技能的强化学习(RL)是一种利用先验知识加速机器人学习的很有前途的策略。技能通常从专家演示中提取，并嵌入到一个潜在空间中，高级RL代理可以从那里将它们作为动作进行采样。然而，这个技能空间是扩展的，并不是所有的技能都与给定的机器人状态相关，这使得探索变得困难。此外，下游RL代理仅限于学习与用于构建技能空间的任务在结构上相似的任务。我们首先提出使用状态条件生成模型来加速技能空间的探索，以基于先前的经验直接将高级代理偏向于与给定状态相关的技能。接下来，我们提出了一种针对细粒度文本{技能适应}的低级别剩余策略，使下游RL代理能够适应未知的任务变化。最后，我们在四个具有挑战性的操作任务中验证了我们的方法，这些任务与构建技能空间所用的任务不同，展示了我们跨任务变化学习的能力，同时显著加快了探索速度，表现优于以前的工作。

**[Paper URL](https://proceedings.mlr.press/v205/rana23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/rana23a/rana23a.pdf)** 

# Learning Representations that Enable Generalization in Assistive Tasks
**题目:** 能够在辅助任务中进行概括的学习表示

**作者:** Jerry Zhi-Yang He, Zackory Erickson, Daniel S. Brown, Aditi Raghunathan, Anca Dragan

**Abstract:** Recent work in sim2real has successfully enabled robots to act in physical environments by training in simulation with a diverse “population” of environments (i.e. domain randomization). In this work, we focus on enabling generalization in \emph{assistive tasks}: tasks in which the robot is acting to assist a user (e.g. helping someone with motor impairments with bathing or with scratching an itch). Such tasks are particularly interesting relative to prior sim2real successes because the environment now contains a \emph{human who is also acting}. This complicates the problem because the diversity of human users (instead of merely physical environment parameters) is more difficult to capture in a population, thus increasing the likelihood of encountering out-of-distribution (OOD) human policies at test time. We advocate that generalization to such OOD policies benefits from (1) learning a good latent representation for human policies that test-time humans can accurately be mapped to, and (2) making that representation adaptable with test-time interaction data, instead of relying on it to perfectly capture the space of human policies based on the simulated population only. We study how to best learn such a representation by evaluating on purposefully constructed OOD test policies. We find that sim2real methods that encode environment (or population) parameters and work well in tasks that robots do in isolation, do not work well in \emph{assistance}.  In assistance, it seems crucial to train the representation based on the \emph{history of interaction} directly, because that is what the robot will have access to at test time. Further, training these representations to then \emph{predict human actions} not only gives them better structure, but also enables them to be fine-tuned at test-time, when the robot observes the partner act.

**摘要:** 最近在sim2Real中的工作已经成功地使机器人能够通过在不同的环境中进行模拟训练(即，域随机化)来在物理环境中行动。在这项工作中，我们专注于在\emph{辅助任务}中实现泛化：在这些任务中，机器人正在行动以帮助用户(例如，帮助有运动障碍的人洗澡或挠痒痒)。与之前的sim2Real成功相比，这样的任务特别有趣，因为现在的环境包含了一个也在表演的人。这使问题复杂化，因为在人群中更难捕获人类用户的多样性(而不仅仅是物理环境参数)，从而增加了在测试时遇到分布外(OOD)人类策略的可能性。我们主张对这样的OOD策略的推广受益于(1)学习测试时间人类可以准确映射到的人类策略的良好潜在表示，以及(2)使该表示与测试时间交互数据相适应，而不是依赖于它来完美地捕捉仅基于模拟种群的人类策略的空间。我们研究如何通过对有目的地构建的OOD测试策略进行评估来最好地学习这样的表示。我们发现，对环境(或种群)参数进行编码并在机器人单独完成的任务中工作得很好的Sim2Real方法，在\emph{辅助}中不能很好地工作。在辅助方面，直接基于交互历史训练表示似乎是至关重要的，因为这是机器人在测试时可以访问的。此外，训练这些表示然后预测人类行为不仅可以使它们具有更好的结构，而且能够在测试时当机器人观察到伙伴行为时对它们进行微调。

**[Paper URL](https://proceedings.mlr.press/v205/he23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/he23a/he23a.pdf)** 

# Learning to Correct Mistakes: Backjumping in Long-Horizon Task and Motion Planning
**题目:** 学会纠正错误：长视野任务和运动规划中的倒退

**作者:** Yoonchang Sung, Zizhao Wang, Peter Stone

**Abstract:** As robots become increasingly capable of manipulation and long-term autonomy, long-horizon task and motion planning problems are becoming increasingly important.  A key challenge in such problems is that early actions in the plan may make future actions infeasible. When reaching a dead-end in the search, most existing planners use backtracking, which exhaustively reevaluates motion-level actions, often resulting in inefficient planning, especially when the search depth is large. In this paper, we propose to learn backjumping heuristics which identify the culprit action directly using supervised learning models to guide the task-level search. Based on evaluations of two different tasks, we find that our method significantly improves planning efficiency compared to backtracking and also generalizes to problems with novel numbers of objects.

**摘要:** 随着机器人的操纵和长期自主能力越来越强，长期任务和运动规划问题变得越来越重要。  此类问题的一个关键挑战是计划中的早期行动可能会使未来的行动变得不可行。当搜索进入死胡同时，大多数现有的规划者都会使用回溯，彻底重新评估运动级动作，这通常会导致规划效率低下，尤其是当搜索深度很大时。在本文中，我们建议学习向后跳启发式方法，该方法直接使用监督学习模型来识别罪魁祸首动作，以指导任务级搜索。基于对两个不同任务的评估，我们发现与回溯相比，我们的方法显着提高了规划效率，并且还推广到具有新颖对象数量的问题。

**[Paper URL](https://proceedings.mlr.press/v205/sung23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/sung23a/sung23a.pdf)** 

# Lyapunov Design for Robust and Efficient Robotic Reinforcement Learning
**题目:** 鲁棒高效的机器人强化学习的李亚普诺夫设计

**作者:** Tyler Westenbroek, Fernando Castaneda, Ayush Agrawal, Shankar Sastry, Koushil Sreenath

**Abstract:** Recent advances in the reinforcement learning (RL) literature have enabled roboticists to automatically train complex policies in simulated environments. However, due to the poor sample complexity of these methods, solving RL problems using real-world data remains a challenging problem. This paper introduces a novel cost-shaping method which aims to reduce the number of samples needed to learn a stabilizing controller. The method adds a term involving a Control Lyapunov Function (CLF) – an ‘energy-like’ function from the model-based control literature – to typical cost formulations. Theoretical results demonstrate the new costs lead to stabilizing controllers when smaller discount factors are used, which is well-known to reduce sample complexity. Moreover, the addition of the CLF term ‘robustifies’ the search for a stabilizing controller by ensuring that even highly sub-optimal polices will stabilize the system. We demonstrate our approach with two hardware examples where we learn stabilizing controllers for a cartpole and an A1 quadruped with only seconds and a few minutes of fine-tuning data, respectively. Furthermore, simulation benchmark studies show that obtaining stabilizing policies by optimizing our proposed costs requires orders of magnitude less data compared to standard cost designs.

**摘要:** 强化学习(RL)文献的最新进展使机器人专家能够在模拟环境中自动训练复杂的策略。然而，由于这些方法的样本复杂性较差，使用真实世界的数据来解决RL问题仍然是一个具有挑战性的问题。本文介绍了一种新的代价整形方法，旨在减少学习镇定控制器所需的样本数。该方法在典型的成本公式中增加了一个涉及控制李亚普诺夫函数(CLF)的术语。CLF是基于模型的控制文献中的一种类似能量的函数。理论结果表明，当使用较小的折扣因子时，新的成本导致控制器稳定，这是众所周知的降低样本复杂度的方法。此外，CLF术语的加入通过确保即使是高度次优的策略也将使系统稳定，从而增强了对稳定控制器的搜索。我们用两个硬件例子来演示我们的方法，在这两个例子中，我们学习了分别只用几秒钟和几分钟的微调数据就能稳定蜗牛和A1四足动物的控制器。此外，模拟基准研究表明，与标准成本设计相比，通过优化我们建议的成本来获得稳定政策所需的数据要少几个数量级。

**[Paper URL](https://proceedings.mlr.press/v205/westenbroek23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/westenbroek23a/westenbroek23a.pdf)** 

# ROAD: Learning an Implicit Recursive Octree Auto-Decoder to Efficiently Encode 3D Shapes
**题目:** ROAD：学习隐式回归八叉树自动解码器以高效编码3D收件箱

**作者:** Sergey Zakharov, Rares Andrei Ambrus, Katherine Liu, Adrien Gaidon

**Abstract:** Compact and accurate representations of 3D shapes are central to many perception and robotics tasks. State-of-the-art learning-based methods can reconstruct single objects but scale poorly to large datasets. We present a novel recursive implicit representation to efficiently and accurately encode large datasets of complex 3D shapes by recursively traversing an implicit octree in latent space. Our implicit Recursive Octree Auto-Decoder (ROAD) learns a hierarchically structured latent space enabling state-of-the-art reconstruction results at a compression ratio above 99%. We also propose an efficient curriculum learning scheme that naturally exploits the coarse-to-fine properties of the underlying octree spatial representation. We explore the scaling law relating latent space dimension, dataset size, and reconstruction accuracy, showing that increasing the latent space dimension is enough to scale to large shape datasets. Finally, we show that our learned latent space encodes a coarse-to-fine hierarchical structure yielding reusable latents across different levels of details, and we provide qualitative evidence of generalization to novel shapes outside the training set.

**摘要:** 紧凑和准确的3D形状表示是许多感知和机器人任务的核心。最先进的基于学习的方法可以重建单个对象，但扩展到大型数据集的能力很差。提出了一种新的递归隐式表示方法，通过递归遍历潜在空间中的隐式八叉树来高效、准确地编码复杂三维形状的大数据集。我们的隐式递归八叉树自动解码器(ROAD)学习了分层结构的潜在空间，使最先进的重建结果在压缩比超过99%的情况下能够实现。我们还提出了一种高效的课程学习方案，该方案自然地利用了底层八叉树空间表示从粗到精的特性。我们探索了潜在空间维度、数据集大小和重建精度之间的缩放规律，表明增加潜在空间维度足以扩展到大型形状数据集。最后，我们证明了我们的学习潜在空间编码了一种从粗到精的层次结构，在不同的细节层次上产生了可重用的潜伏期，并且我们提供了在训练集之外对新形状进行泛化的定性证据。

**[Paper URL](https://proceedings.mlr.press/v205/zakharov23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/zakharov23a/zakharov23a.pdf)** 

# Safe Robot Learning in Assistive Devices through Neural Network Repair
**题目:** 通过神经网络修复实现机器人在辅助设备中的安全学习

**作者:** Keyvan Majd, Geoffrey Mitchell Clark, Tanmay Khandait, Siyu Zhou, Sriram Sankaranarayanan, Georgios Fainekos, Heni Amor

**Abstract:** Assistive robotic devices are a particularly promising field of application for neural networks (NN) due to the need for personalization and hard-to-model human-machine interaction dynamics. However, NN based estimators and controllers may produce potentially unsafe outputs over previously unseen data points. In this paper, we introduce an algorithm for updating NN control policies to satisfy a given set of formal safety constraints, while also optimizing the original loss function.  Given a set of mixed-integer linear constraints, we define the NN repair problem as a Mixed Integer Quadratic Program (MIQP). In extensive experiments, we demonstrate the efficacy of our repair method in generating safe policies for a lower-leg prosthesis.

**摘要:** 由于需要个性化和难以建模的人机交互动力学，辅助机器人设备是神经网络（NN）一个特别有前途的应用领域。然而，基于神经网络的估计器和控制器可能会在之前未见过的数据点上产生潜在不安全的输出。本文中，我们引入了一种更新NN控制策略的算法，以满足一组给定的形式安全约束，同时优化原始损失函数。  给定一组混合整线性约束，我们将神经网络修复问题定义为混合迭代二次规划（MIQP）。在大量的实验中，我们证明了我们的修复方法在为小腿假肢制定安全策略方面的有效性。

**[Paper URL](https://proceedings.mlr.press/v205/majd23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/majd23a/majd23a.pdf)** 

# Contrastive Decision Transformers
**题目:** 对比决策变形金刚

**作者:** Sachin G. Konan, Esmaeil Seraj, Matthew Gombolay

**Abstract:** Decision Transformers (DT) have drawn upon the success of Transformers by abstracting Reinforcement Learning as a target-return-conditioned, sequence modeling problem. In our work, we claim that the distribution of DT’s target-returns represents a series of different tasks that agents must learn to handle. Work in multi-task learning has shown that separating the representations of input data belonging to different tasks can improve performance. We draw from this approach to construct ConDT (Contrastive Decision Transformer). ConDT leverages an enhanced contrastive loss to train a return-dependent transformation of the input embeddings, which we empirically show clusters these embeddings by their return. We find that ConDT significantly outperforms DT in Open-AI Gym domains by 10% and 39% in visually challenging Atari domains.

**摘要:** 决策变形者（DT）利用了变形者的成功，将强化学习抽象为目标返回条件的序列建模问题。在我们的工作中，我们声称DT目标回报的分布代表了代理必须学习处理的一系列不同任务。多任务学习的工作表明，分离属于不同任务的输入数据的表示可以提高性能。我们借鉴这种方法来构建ConDT（对比决策Transformer）。ConDT利用增强的对比损失来训练输入嵌入的依赖回报的转换，我们根据经验展示了通过回报对这些嵌入进行集群。我们发现，在具有视觉挑战性的Atari领域，ConDT在开放AI Gym领域的表现明显优于DT 10%和39%。

**[Paper URL](https://proceedings.mlr.press/v205/konan23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/konan23a/konan23a.pdf)** 

# DiffStack: A Differentiable and Modular Control Stack for Autonomous Vehicles
**题目:** DistStack：用于自动驾驶汽车的差异化和模块化控制栈

**作者:** Peter Karkus, Boris Ivanovic, Shie Mannor, Marco Pavone

**Abstract:** Autonomous vehicle (AV) stacks are typically built in a modular fashion, with explicit components performing detection, tracking, prediction, planning, control, etc. While modularity improves reusability, interpretability, and generalizability, it also suffers from compounding errors, information bottlenecks, and integration challenges. To overcome these challenges, a prominent approach is to convert the AV stack into an end-to-end neural network and train it with data. While such approaches have achieved impressive results, they typically lack interpretability and reusability, and they eschew principled analytical components, such as planning and control, in favor of deep neural networks. To enable the joint optimization of AV stacks while retaining modularity, we present DiffStack, a differentiable and modular stack for prediction, planning, and control. Crucially, our model-based planning and control algorithms leverage recent advancements in differentiable optimization to produce gradients, enabling optimization of upstream components, such as prediction, via backpropagation through planning and control. Our results on the nuScenes dataset indicate that end-to-end training with DiffStack yields substantial improvements in open-loop and closed-loop planning metrics by, e.g., learning to make fewer prediction errors that would affect planning. Beyond these immediate benefits, DiffStack opens up new opportunities for fully data-driven yet modular and interpretable AV architectures.

**摘要:** 自动驾驶汽车(AV)堆栈通常以模块化方式构建，由显式组件执行检测、跟踪、预测、规划、控制等。模块化在提高可重用性、可解释性和通用性的同时，也存在组合错误、信息瓶颈和集成挑战。为了克服这些挑战，一个突出的方法是将反病毒堆栈转换为端到端的神经网络，并用数据对其进行训练。虽然这些方法取得了令人印象深刻的结果，但它们通常缺乏可解释性和可重用性，而且它们避开了原则性分析组件，如计划和控制，而倾向于深度神经网络。为了在保持模块化的同时实现AV堆栈的联合优化，我们提出了DiffStack，这是一种可区分的模块化堆栈，用于预测、规划和控制。至关重要的是，我们基于模型的计划和控制算法利用可微优化方面的最新进展来产生梯度，通过计划和控制的反向传播实现上游组件的优化，如预测。我们在nuScenes数据集上的结果表明，使用DiffStack的端到端培训通过学习减少会影响规划的预测错误，在开环和闭环规划指标方面产生了实质性的改进。除了这些立竿见影的好处外，DiffStack还为完全由数据驱动但模块化且可解释的反病毒架构带来了新的机遇。

**[Paper URL](https://proceedings.mlr.press/v205/karkus23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/karkus23a/karkus23a.pdf)** 

# Learning and Retrieval from Prior Data for Skill-based Imitation Learning
**题目:** 基于技能的模仿学习的先前数据的学习和检索

**作者:** Soroush Nasiriany, Tian Gao, Ajay Mandlekar, Yuke Zhu

**Abstract:** Imitation learning offers a promising path for robots to learn general-purpose tasks, but traditionally has enjoyed limited scalability due to high data supervision requirements and brittle generalization. Inspired by recent work on skill-based imitation learning, we investigate whether leveraging prior data from previous related tasks can enable learning novel tasks in a more robust, data-efficient manner. To make effective use of the prior data, the agent must internalize knowledge from the prior data and contextualize this knowledge in novel tasks. To that end we propose a skill-based imitation learning framework that extracts temporally-extended sensorimotor skills from prior data and subsequently learns a policy for the target task with respect to these learned skills. We find a number of modeling choices significantly improve performance on novel tasks, namely representation learning objectives to enable more predictable and consistent skill representations and a retrieval-based data augmentation procedure to increase the scope of supervision for the policy. On a number of multi-task manipulation domains, we demonstrate that our method significantly outperforms existing imitation learning and offline reinforcement learning approaches. Videos and code are available at https://ut-austin-rpl.github.io/sailor

**摘要:** 模仿学习为机器人学习通用任务提供了一条很有前途的途径，但传统上由于高数据监管要求和脆弱的泛化，可伸缩性有限。受最近关于基于技能的模仿学习的研究的启发，我们调查了利用先前相关任务的先前数据是否能够以更稳健、更有效的方式学习新任务。为了有效地利用先前的数据，代理必须将先前数据中的知识内部化，并在新的任务中将这些知识背景化。为此，我们提出了一个基于技能的模仿学习框架，它从先前的数据中提取时间扩展的感觉运动技能，并随后学习关于这些学习技能的目标任务的策略。我们发现，许多建模选择显著提高了新任务的性能，即表征学习目标，以实现更可预测和一致的技能表征，以及基于检索的数据增强过程，以扩大策略的监督范围。在多个多任务操作领域上，我们证明了我们的方法明显优于现有的模仿学习和离线强化学习方法。有关视频和代码，请访问https://ut-austin-rpl.github.io/sailor

**[Paper URL](https://proceedings.mlr.press/v205/nasiriany23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/nasiriany23a/nasiriany23a.pdf)** 

# Learning Semantics-Aware Locomotion Skills from Human Demonstration
**题目:** 从人类演示中学习语义意识的运动技能

**作者:** Yuxiang Yang, Xiangyun Meng, Wenhao Yu, Tingnan Zhang, Jie Tan, Byron Boots

**Abstract:** The semantics of the environment, such as the terrain type and property, reveals important information for legged robots to adjust their behaviors. In this work, we present a framework that learns semantics-aware locomotion skills from perception for quadrupedal robots, such that the robot can traverse through complex offroad terrains with appropriate speeds and gaits using perception information. Due to the lack of high-fidelity outdoor simulation, our framework needs to be trained directly in the real world, which brings unique challenges in data efficiency and safety. To ensure sample efficiency, we pre-train the perception model with an off-road driving dataset. To avoid the risks of real-world policy exploration, we leverage human demonstration to train a speed policy that selects a desired forward speed from camera image. For maximum traversability, we pair the speed policy with a gait selector, which selects a robust locomotion gait for each forward speed. Using only 40 minutes of human demonstration data, our framework learns to adjust the speed and gait of the robot based on perceived terrain semantics, and enables the robot to walk over 6km without failure at close-to-optimal speed

**摘要:** 环境的语义，如地形类型和属性，为腿部机器人调整其行为提供了重要信息。在这项工作中，我们提出了一个框架，它从感知中学习四足机器人的语义感知运动技能，使机器人能够利用感知信息以适当的速度和步态穿越复杂的越野地形。由于缺乏高保真的室外仿真，我们的框架需要直接在真实世界中进行训练，这在数据效率和安全方面带来了独特的挑战。为了确保样本效率，我们用越野驾驶数据集预训练感知模型。为了避免现实世界政策探索的风险，我们利用人类演示来训练速度策略，该策略从摄像机图像中选择所需的前进速度。为了获得最大的可通过性，我们将速度策略与步态选择器配对，该选择器为每个前进速度选择一个健壮的运动步态。该框架仅使用40分钟的人类演示数据，就可以根据感知的地形语义来学习调整机器人的速度和步态，并使机器人能够以接近最佳的速度无故障地行走6公里

**[Paper URL](https://proceedings.mlr.press/v205/yang23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/yang23a/yang23a.pdf)** 

# TRITON: Neural Neural Textures for Better Sim2Real
**题目:** TRITON：神经神经纹理，打造更好的Sim2 Real

**作者:** Ryan D. Burgert, Jinghuan Shang, Xiang Li, Michael S. Ryoo

**Abstract:** Unpaired image translation algorithms can be used for sim2real tasks, but many fail to generate temporally consistent results. We present a new approach that combines differentiable rendering with image translation to achieve temporal consistency over indefinite timescales, using surface consistency losses and neu- ral neural textures. We call this algorithm TRITON (Texture Recovering Image Translation Network): an unsupervised, end-to-end, stateless sim2real algorithm that leverages the underlying 3D geometry of input scenes by generating realistic- looking learnable neural textures. By settling on a particular texture for the objects in a scene, we ensure consistency between frames statelessly. TRITON is not lim- ited to camera movements — it can handle the movement and deformation of ob- jects as well, making it useful for downstream tasks such as robotic manipulation. We demonstrate the superiority of our approach both qualitatively and quantita- tively, using robotic experiments and comparisons to ground truth photographs. We show that TRITON generates more useful images than other algorithms do. Please see our project website: tritonpaper.github.io

**摘要:** 未配对的图像转换算法可以用于简单的任务，但许多算法无法产生时间上一致的结果。我们提出了一种结合可微绘制和图像平移的新方法，利用表面一致性损失和神经纹理来实现无限时间尺度上的时间一致性。我们将该算法称为Triton(纹理恢复图像转换网络)：一种无监督、端到端、无状态的sim2Real算法，它通过生成看起来逼真的、可学习的神经纹理来利用输入场景的底层3D几何。通过确定场景中对象的特定纹理，我们可以无状态地确保帧之间的一致性。Triton不受相机移动的限制-它还可以处理物体的移动和变形，这使得它对于机器人操纵等下游任务很有用。我们使用机器人实验和与地面真实照片的比较，从定性和定量两个方面证明了我们方法的优越性。我们证明了Triton算法比其他算法生成了更多有用的图像。请查看我们的项目网站：tritonPap.githorb.io

**[Paper URL](https://proceedings.mlr.press/v205/burgert23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/burgert23a/burgert23a.pdf)** 

# DayDreamer: World Models for Physical Robot Learning
**题目:** DayDreamer：物理机器人学习的世界模型

**作者:** Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, Ken Goldberg

**Abstract:** To solve tasks in complex environments, robots need to learn from experience. Deep reinforcement learning is a common approach to robot learning but requires a large amount of trial and error to learn, limiting its deployment in the physical world. As a consequence, many advances in robot learning rely on simulators. On the other hand, learning inside of simulators fails to capture the complexity of the real world, is prone to simulator inaccuracies, and the resulting behaviors do not adapt to changes in the world. The Dreamer algorithm has recently shown great promise for learning from small amounts of interaction by planning within a learned world model, outperforming pure reinforcement learning in video games. Learning a world model to predict the outcomes of potential actions enables planning in imagination, reducing the amount of trial and error needed in the real environment. However, it is unknown whether Dreamer can facilitate faster learning on physical robots. In this paper, we apply Dreamer to 4 robots to learn online and directly in the real world, without any simulators. Dreamer trains a quadruped robot to roll off its back, stand up, and walk from scratch and without resets in only 1 hour. We then push the robot and find that Dreamer adapts within 10 minutes to withstand perturbations or quickly roll over and stand back up. On two different robotic arms, Dreamer learns to pick and place objects from camera images and sparse rewards, approaching human-level teleoperation performance. On a wheeled robot, Dreamer learns to navigate to a goal position purely from camera images, automatically resolving ambiguity about the robot orientation. Using the same hyperparameters across all experiments, we find that Dreamer is capable of online learning in the real world, which establishes a strong baseline. We release our infrastructure for future applications of world models to robot learning.

**摘要:** 要解决复杂环境中的任务，机器人需要从经验中学习。深度强化学习是机器人学习的一种常见方法，但需要进行大量的试验和错误学习，限制了其在物理世界中的部署。因此，机器人学习的许多进步都依赖于模拟器。另一方面，模拟器内部的学习无法捕捉到真实世界的复杂性，容易出现模拟器的不准确，由此产生的行为不能适应世界的变化。Dreamer算法最近显示出通过在学习的世界模型中进行规划来从少量交互中学习的巨大前景，在视频游戏中的表现优于纯粹的强化学习。通过学习世界模型来预测潜在行动的结果，能够在想象中进行规划，减少在现实环境中所需的试错量。然而，目前尚不清楚Dreamer能否在物理机器人上促进更快的学习。在本文中，我们将Dreamer应用于4个机器人，在没有任何模拟器的情况下，在线和直接在真实世界中学习。Dreamer训练一个四足机器人在1小时内从背上滚下来，站起来，从头开始走，不重置。然后，我们推动机器人，发现Dreamer在10分钟内就能适应，以抵御干扰或迅速翻身并重新站立。在两个不同的机械臂上，Dreamer学习从相机图像和稀疏的奖励中挑选和放置对象，接近人类水平的遥操作性能。在轮式机器人上，Dreamer学习完全通过摄像头图像导航到目标位置，自动解决关于机器人方向的模糊问题。在所有实验中使用相同的超参数，我们发现Dreamer能够在现实世界中在线学习，这建立了一个强大的基线。我们发布了我们的基础设施，用于未来将世界模型应用于机器人学习。

**[Paper URL](https://proceedings.mlr.press/v205/wu23c.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wu23c/wu23c.pdf)** 

# INQUIRE: INteractive Querying for User-aware Informative REasoning
**题目:** 查询：交互式查询用户感知的信息推理

**作者:** Tesca Fitzgerald, Pallavi Koppol, Patrick Callaghan, Russell Quinlan Jun Hei Wong, Reid Simmons, Oliver Kroemer, Henny Admoni

**Abstract:** Research on Interactive Robot Learning has yielded several modalities for querying a human for training data, including demonstrations, preferences, and corrections. While prior work in this space has focused on optimizing the robot’s queries within each interaction type, there has been little work on optimizing over the selection of the interaction type itself. We present INQUIRE, the first algorithm to implement and optimize over a generalized representation of information gain across multiple interaction types. Our evaluations show that INQUIRE can dynamically optimize its interaction type (and respective optimal query) based on its current learning status and the robot’s state in the world, resulting in more robust performance across tasks in comparison to state-of-the art baseline methods. Additionally, INQUIRE allows for customizable cost metrics to bias its selection of interaction types, enabling this algorithm to be tailored to a robot’s particular deployment domain and formulate cost-aware, informative queries.

**摘要:** 对交互式机器人学习的研究已经产生了几种向人类查询训练数据的方式，包括演示、偏好和纠正。虽然这一领域的先前工作主要集中在优化机器人在每种交互类型中的查询，但对交互类型本身的选择进行优化的工作很少。我们提出了Query，这是第一个在多个交互类型上实现和优化信息增益的通用表示的算法。我们的评估表明，Query可以根据其当前的学习状态和机器人在世界上的状态来动态优化其交互类型(以及相应的最优查询)，与最新的基线方法相比，可以获得更好的跨任务性能。此外，Query允许可定制的成本指标来偏向交互类型的选择，使该算法能够针对机器人的特定部署领域进行量身定做，并制定具有成本意识的、信息丰富的查询。

**[Paper URL](https://proceedings.mlr.press/v205/fitzgerald23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/fitzgerald23a/fitzgerald23a.pdf)** 

# Online Dynamics Learning for Predictive Control with an Application to Aerial Robots
**题目:** 预测控制的在线动态学习及其在空中机器人中的应用

**作者:** Tom Z. Jiahao, Kong Yao Chee, M. Ani Hsieh

**Abstract:** In this work, we consider the task of improving the accuracy of dynamic models for model predictive control (MPC) in an online setting. Although prediction models can be learned and applied to model-based controllers, these models are often learned offline. In this offline setting, training data is first collected and a prediction model is learned through an elaborated training procedure. However, since the model is learned offline, it does not adapt to disturbances or model errors observed during deployment. To improve the adaptiveness of the model and the controller, we propose an online dynamics learning framework that continually improves the accuracy of the dynamic model during deployment. We adopt knowledge-based neural ordinary differential equations (KNODE) as the dynamic models, and use techniques inspired by transfer learning to continually improve the model accuracy. We demonstrate the efficacy of our framework with a quadrotor, and verify the framework in both simulations and physical experiments. Results show that our approach can account for disturbances that are possibly time-varying, while maintaining good trajectory tracking performance.

**摘要:** 在这项工作中，我们考虑了在线环境下提高模型预测控制(MPC)动态模型精度的任务。虽然预测模型可以学习并应用于基于模型的控制器，但这些模型通常是离线学习的。在这种离线设置中，首先收集训练数据，并通过详细的训练过程学习预测模型。然而，由于该模型是离线学习的，因此它不适应部署期间观察到的干扰或模型错误。为了提高模型和控制器的自适应性，我们提出了一个在线动态学习框架，在部署过程中不断提高动态模型的精度。我们采用基于知识的神经常微分方程组(Knode)作为动态模型，并利用迁移学习启发的技术来不断提高模型的精度。我们用四旋翼验证了该框架的有效性，并在模拟和物理实验中验证了该框架。结果表明，在保持良好的轨迹跟踪性能的同时，我们的方法能够考虑到可能是时变的干扰。

**[Paper URL](https://proceedings.mlr.press/v205/jiahao23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/jiahao23a/jiahao23a.pdf)** 

# Skill-based Model-based Reinforcement Learning
**题目:** 基于技能的基于模型的强化学习

**作者:** Lucy Xiaoyang Shi, Joseph J Lim, Youngwoon Lee

**Abstract:** Model-based reinforcement learning (RL) is a sample-efficient way of learning complex behaviors by leveraging a learned single-step dynamics model to plan actions in imagination. However, planning every action for long-horizon tasks is not practical, akin to a human planning out every muscle movement. Instead, humans efficiently plan with high-level skills to solve complex tasks. From this intuition, we propose a Skill-based Model-based RL framework (SkiMo) that enables planning in the skill space using a skill dynamics model, which directly predicts the skill outcomes, rather than predicting all small details in the intermediate states, step by step. For accurate and efficient long-term planning, we jointly learn the skill dynamics model and a skill repertoire from prior experience. We then harness the learned skill dynamics model to accurately simulate and plan over long horizons in the skill space, which enables efficient downstream learning of long-horizon, sparse reward tasks. Experimental results in navigation and manipulation domains show that SkiMo extends the temporal horizon of model-based approaches and improves the sample efficiency for both model-based RL and skill-based RL. Code and videos are available at https://clvrai.com/skimo

**摘要:** 基于模型的强化学习(RL)是一种通过利用已学习的单步动力学模型来规划想象中的动作来学习复杂行为的有效样本方法。然而，为长远的任务计划每一个动作是不切实际的，就像人类计划每一个肌肉运动一样。相反，人类用高水平的技能高效地计划解决复杂的任务。根据这一直觉，我们提出了一个基于技能的基于模型的RL框架(SkiMO)，它允许使用技能动态模型在技能空间中进行规划，该模型直接预测技能结果，而不是逐步预测中间状态下的所有小细节。为了准确和有效地进行长期规划，我们从先前的经验中共同学习技能动态模型和技能曲目。然后，我们利用学习到的技能动态模型来准确地模拟和规划技能空间中的长期目标，这使得能够有效地向下学习长期、稀疏奖励任务。在导航和操纵领域的实验结果表明，SkiMO扩展了基于模型的方法的时间范围，并提高了基于模型的RL和基于技能的RL的样本效率。代码和视频可在https://clvrai.com/skimo上查看

**[Paper URL](https://proceedings.mlr.press/v205/shi23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/shi23a/shi23a.pdf)** 

# Data-Efficient Model Learning for Control with Jacobian-Regularized Dynamic-Mode Decomposition
**题目:** 采用Jacobian正规化动态模式分解的数据高效控制模型学习

**作者:** Brian Edward Jackson, Jeong Hun Lee, Kevin Tracy, Zachary Manchester

**Abstract:** We present a data-efficient algorithm for learning models for model-predictive control (MPC). Our approach, Jacobian-Regularized Dynamic-Mode Decomposition (JDMD), offers improved sample efficiency over traditional Koopman approaches based on Dynamic-Mode Decomposition (DMD) by leveraging Jacobian information from an approximate prior model of the system, and improved tracking performance over traditional model-based MPC. We demonstrate JDMD’s ability to quickly learn bilinear Koopman dynamics representations across several realistic examples in simulation, including a perching maneuver for a fixed-wing aircraft with an empirically derived high-fidelity physics model. In all cases, we show that the models learned by JDMD provide superior tracking and generalization performance within a model-predictive control framework, even in the presence of significant model mismatch, when compared to approximate prior models and models learned by standard Extended DMD (EDMD).

**摘要:** 我们提出了一种用于模型预测控制（MPC）学习模型的数据高效算法。我们的方法Jacobian-Regulated Dynamic-Mode Decomsion（JDMZ）通过利用来自系统的大约先验模型的Jacobian信息，比基于Dynamic-Mode Decomsion（DMZ）的传统Koopman方法提供了更高的样本效率，并比传统的基于模型的MPC改进了跟踪性能。我们展示了JDMZ在模拟中的几个现实示例中快速学习双线性Koopman动力学表示的能力，包括具有经验推导的高保真物理模型的固定翼飞机的栖息机动。在所有情况下，我们表明，与近似先验模型和通过标准扩展DMZ（EDMZ）学习的模型相比，即使存在显着的模型不匹配，JDMZ学习的模型也能在模型预测控制框架内提供卓越的跟踪和概括性能。

**[Paper URL](https://proceedings.mlr.press/v205/jackson23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/jackson23a/jackson23a.pdf)** 

# In-Hand Gravitational Pivoting Using Tactile Sensing
**题目:** 使用触觉传感的手内重力旋转

**作者:** Jason Toskov, Rhys Newbury, Mustafa Mukadam, Dana Kulic, Akansel Cosgun

**Abstract:** We study gravitational pivoting, a constrained version of in-hand manipulation, where we aim to control the rotation of an object around the grip point of a parallel gripper. To achieve this, instead of controlling the gripper to avoid slip, we \emph{embrace slip} to allow the object to rotate in-hand. We collect two real-world datasets, a static tracking dataset and a controller-in-the-loop dataset, both annotated with object angle and angular velocity labels. Both datasets contain force-based tactile information on ten different household objects. We train an LSTM model to predict the angular position and velocity of the held object from purely tactile data. We integrate this model with a controller that opens and closes the gripper allowing the object to rotate to desired relative angles. We conduct real-world experiments where the robot is tasked to achieve a relative target angle. We show that our approach outperforms a sliding-window based MLP in a zero-shot generalization setting with unseen objects. Furthermore, we show a 16.6% improvement in performance when the LSTM model is fine-tuned on a small set of data collected with both the LSTM model and the controller in-the-loop. Code and videos are available at https://rhys-newbury.github.io/projects/pivoting/.

**摘要:** 我们学习重力枢转，这是手部操作的一种受约束的版本，我们的目标是控制对象围绕平行抓取器的夹点的旋转。为了实现这一点，我们不是控制夹爪以避免滑动，而是允许对象手部旋转。我们收集了两个真实世界的数据集，一个是静态跟踪数据集，另一个是环路控制器数据集，这两个数据集都用对象角度和角速度标签进行了标注。这两个数据集都包含十种不同家用物品的力觉信息。我们训练LSTM模型来从纯触觉数据预测持有对象的角位置和速度。我们将这个模型与一个控制器集成在一起，该控制器可以打开和关闭夹爪，允许对象旋转到所需的相对角度。我们进行了真实世界的实验，其中机器人的任务是实现相对的目标角度。我们证明了我们的方法比基于滑动窗口的MLP在不可见物体的零镜头泛化环境下的性能更好。此外，当LSTM模型根据LSTM模型和控制器在环中收集的一小部分数据进行微调时，性能提高了16.6%。代码和视频可在https://rhys-newbury.github.io/projects/pivoting/.上查看

**[Paper URL](https://proceedings.mlr.press/v205/toskov23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/toskov23a/toskov23a.pdf)** 

# CC-3DT: Panoramic 3D Object Tracking via Cross-Camera Fusion
**题目:** CC-3DT：通过跨摄像机融合实现全景3D物体跟踪

**作者:** Tobias Fischer, Yung-Hsu Yang, Suryansh Kumar, Min Sun, Fisher Yu

**Abstract:** To track the 3D locations and trajectories of the other traffic participants at any given time, modern autonomous vehicles are equipped with multiple cameras that cover the vehicle’s full surroundings. Yet, camera-based 3D object tracking methods prioritize optimizing the single-camera setup and resort to post-hoc fusion in a multi-camera setup. In this paper, we propose a method for panoramic 3D object tracking, called CC-3DT, that associates and models object trajectories both temporally and across views, and improves the overall tracking consistency. In particular, our method fuses 3D detections from multiple cameras before association, reducing identity switches significantly and improving motion modeling. Our experiments on large-scale driving datasets show that fusion before association leads to a large margin of improvement over post-hoc fusion. We set a new state-of-the-art with 12.6% improvement in average multi-object tracking accuracy (AMOTA) among all camera-based methods on the competitive NuScenes 3D tracking benchmark, outperforming previously published methods by 6.5% in AMOTA with the same 3D detector.

**摘要:** 为了在任何给定的时间跟踪其他交通参与者的3D位置和轨迹，现代自动驾驶汽车配备了多个摄像头，覆盖车辆的整个环境。然而，基于摄像机的3D目标跟踪方法优先考虑优化单摄像机设置，并在多摄像机设置中求助于后自组织融合。本文提出了一种全景三维目标跟踪方法CC-3DT，该方法在时间上和跨视点对目标轨迹进行关联和建模，提高了整体跟踪的一致性。特别是，我们的方法在关联之前融合了来自多个摄像机的3D检测，显著减少了身份切换，并改进了运动建模。我们在大规模驾驶数据集上的实验表明，关联之前的融合比后自组织融合有很大的改善。在竞争激烈的NuScenes 3D跟踪基准中，我们在所有基于摄像头的方法中设置了一个新的最先进的多目标跟踪精度(AMOTA)提高了12.6%，在使用相同3D探测器的AMOTA中，性能比之前发表的方法高出6.5%。

**[Paper URL](https://proceedings.mlr.press/v205/fischer23a.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/fischer23a/fischer23a.pdf)** 

# QuaDUE-CCM: Interpretable Distributional Reinforcement Learning using Uncertain Contraction Metrics for Precise Quadrotor Trajectory Tracking
**题目:** CLARDUE-CGM：使用不确定收缩时间表进行可解释的分布式强化学习，用于精确的四螺旋桨轨迹跟踪

**作者:** Yanran Wang, James O’Keeffe, Qiuchen Qian, David Boyle

**Abstract:** Accuracy and stability are common requirements for Quadrotor trajectory tracking systems. Designing an accurate and stable tracking controller remains challenging, particularly in unknown and dynamic environments with complex aerodynamic disturbances. We propose a Quantile-approximation-based Distributional-reinforced Uncertainty Estimator (QuaDUE) to accurately identify the effects of aerodynamic disturbances, i.e., the uncertainties between the true and estimated Control Contraction Metrics (CCMs). Taking inspiration from contraction theory and integrating the QuaDUE for uncertainties, our novel CCM-based trajectory tracking framework tracks any feasible reference trajectory precisely whilst guaranteeing exponential convergence. More importantly, the convergence and training acceleration of the distributional RL are guaranteed and analyzed, respectively, from theoretical perspectives. We also demonstrate our system under unknown and diverse aerodynamic forces. Under large aerodynamic forces (>2  m/s^2), compared with the classic data-driven approach, our QuaDUE-CCM achieves at least a 56.6% improvement in tracking error. Compared with QuaDRED-MPC, a distributional RL-based approach, QuaDUE-CCM achieves at least a 3 times improvement in contraction rate.

**摘要:** 精度和稳定性是四旋翼轨迹跟踪系统的共同要求。设计准确而稳定的跟踪控制器仍然具有挑战性，特别是在具有复杂气动扰动的未知和动态环境中。我们提出了一种基于分位数近似的分布增强不确定性估计器(QuaDUE)来准确识别气动扰动的影响，即真实控制收缩度量(CCM)和估计控制收缩度量(CCM)之间的不确定性。从收缩理论中得到启发，并结合考虑不确定性的QuaDUE，我们基于CCM的新型轨迹跟踪框架在保证指数收敛的同时，精确跟踪任何可行的参考轨迹。更重要的是，分别从理论上保证和分析了分布式RL的收敛和训练加速。我们还演示了我们的系统在未知和不同的空气动力下的情况。在大气动力下(>2米/S^2)，与经典的数据驱动方法相比，我们的QuaDUE-CCM在跟踪误差方面至少提高了56.6%。与基于RL的分布式方法QuaDRED-MPC相比，QuaDUE-CCM在缩放率方面至少提高了3倍。

**[Paper URL](https://proceedings.mlr.press/v205/wang23d.html)** 

**[Paper PDF](https://proceedings.mlr.press/v205/wang23d/wang23d.pdf)** 

