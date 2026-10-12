# Ken Goldberg 的 fixturing 研究：专题阅读与对当前项目的反方评议

审阅日期：2026-09-19。任务范围：Goldberg 作者相关的实体工装、夹持接触设计、装卸、公差、复用；不把所有泛化抓取/机器人论文混入。检索作者 CV、旧 publications、FixtureNet 项目页、合作者 Zhuang/Wagner/van der Stappen 目录，并以题名检索出版社、大学库和 DOI。下表区分正式原文、相关作者版本和只读摘要。不是宣称网上所有版本均已获得；也不把没找到的内容当作不存在。

入口：[更新的作者目录](https://docs.google.com/document/d/e/2PACX-1vSyYwcuGoXk1NpJTOWLArGGDps5ftrnyJItyky41EZBag9JS5yVm_w3LhqkH9C0vTabOwKeYE2zO4uy/pub)、[Goldberg CV](https://goldberg.berkeley.edu/cv.html)、[publications](https://goldberg.berkeley.edu/pubs/)、[FixtureNet](https://goldberg.berkeley.edu/fixturing/)、[Zhuang 目录](https://people.eecs.berkeley.edu/~jfc/yzhuang/publications.html)、[van der Stappen 目录](https://webspace.science.uu.nl/~stapp101/publications.html)。全文与转换文本暂存在 `/tmp/goldberg_fixture_sources/`。

## 最重要的四项更正

1. **不能说比 Goldberg 新增了“明确的对抗力”。** Brost–Goldberg 1996 §III-F，正式页 39–40，已经输入预期外载的方向区域和幅值，按抵抗这些任务载荷所需的最大接触反力排序；图12专门展示“抗向下力好”和“抗顺时针力矩好”是不同布局。2011/2016 Schulman–Goldberg–Abbeel 更直接以候选接触在力界下覆盖 wrench 的能力做优化。这里不只是一般 form closure。
2. **不能说首次考虑 fixture 的装入/退出。** Yu–Goldberg 1995/1998 研究传感与顺应装配；Wagner 1997 学位论文第7章甚至明确“放入物体—稳定后松开机器人—合回其他 struts—反向卸载”。不过它们的硬件、自由度和路径类别与我们不同。
3. **不能说首次复用接触结构。** Zhuang–Goldberg 1997 和同年 FixtureNet II 已讨论零件变形/重新设计后复用同一组3个 locators。FixtureNet II p6脚注明确夹紧器位置很可能仍需改变；这不是整套工装完全不变，也不是同一个刚体在多个3D姿态共享共形模块。
4. **不能把上述反证扩大成“旧方法已经解决我们的全部问题”。** 离散点选取不等于共形接触区域、连续面积、刚性骨架、真实装卸扫掠体、每姿态工具空间与基座接口的联合形状设计。这才是仍需明确、验证的技术范围。

## 逐项覆盖（相关会议/期刊版本合并，但不假装逐版都读过）

| 研究线及版本 | 获得和阅读范围 | 核心模型/结论；与我们的关系 |
|---|---|---|
| **Randy C. Brost, Kenneth Y. Goldberg.** *A Complete Algorithm for Synthesizing Modular Fixtures for Polygonal Parts*, ICRA 1994；扩展 *A Complete Algorithm for Designing Planar Fixtures Using Modular Components*, IEEE TRA 12(1), 31–46, 1996。 | 已读[1996正式全文](https://goldberg.berkeley.edu/pubs/modular-fixturing-brost-goldberg.pdf)，§I、III、例子与讨论，尤其pp39–40。 | 多边形、3个格点圆柱定位器+1个滑动夹具；无摩擦平面form closure；枚举全部可行设计、避开stay-out区域、检查夹具实体/行程/有限底板，按任意质量指标排序。明确用任务外载评价。没有3D共形区域、共享骨架和多场景装卸联合搜索。 |
| **Yan Zhuang, Ken Goldberg, Y. C. Wong.** *On the Existence of Modular Fixtures*, ICRA 1994；Zhuang–Goldberg扩展 *On the Existence of Solutions in Modular Fixturing*, IJRR 15(6), 646–656, 1996。 | 已读[作者期刊预印本](https://goldberg.berkeley.edu/pubs/existence.pdf)，全文主论证§4–6。CV与旧网页有15(5)笔误；FixtureNet II引用与期刊目录指15(6)。 | 存在任意大的多边形也无法用特定3L/1C格点硬件固定；扩大到T-slot/4C后对受限形状类给正面存在结果。提醒：硬件形式改变可行集，不能凭“底座足够大”代替模块接口和路径的存在性证明。 |
| **Richard Wagner, Yan Zhuang, Ken Goldberg.** *Fixturing Faceted Parts with Seven Modular Struts*, ISATP 1995, 133–139。 | 会议原文未独立取到；读了第一作者[1997博士论文](https://rjwagner49.com/Iris/Fixture/RWDissertation.pdf) §5 pp44–64、§5.10 p114、§7 pp121–125，作为关联原始资料。不可把1997新增结果全归给1995。 | 给定3D姿态和有向平面片，从箱体格点支出可调压杆；七接触form closure。学位论文允许object-frame任务载荷/反向载荷，按最大反力排序；有stay-out与抓点排除。其装载先移开下向杆，垂直放置到稳定子集，机器人退出后合回其他杆。与我们的完整共享H滑入不同。 |
| **Kyeonah Yu, Ken Y. Goldberg.** *Fixture Loading with Sensor-Based Motion Plans*, ISATP 1995, 362–367；*A Complete Algorithm for Fixture Loading*, IJRR 17(11), 1214–1224, 1998。 | [出版社摘要与元数据](https://journals.sagepub.com/doi/10.1177/027836499801701106)已读；正式全文未获，不能评价实现细节或实测数值。另Wagner论文§2.4有作者相关讨论。 | 3L/1C平面fixture；初始pose有不确定性，定位器二值接触传感+末端被动弹簧顺应，规划滑动/旋转接触序列，给完备算法及3-2-1规则充分条件。不能将“有装入轨迹”本身当新颖性；我们可能不同在无动夹具的3D形状/路径共同设计。 |
| **Yan Zhuang, Ken Goldberg.** *Design Rules for Tolerance-Insensitive and Multi-Purpose Fixtures*, ICAR 1997, 681–686，ESRC97-02。 | DOI [10.1109/ICAR.1997.620255](https://doi.org/10.1109/ICAR.1997.620255)由Crossref核对；独立原文未取到。作者上传摘要+下面FixtureNet II p6–8的同作者方法章节可读，后者是技术判断依据。 | 形状公差与产品重设计后复用定位器；给定旧设计的复用兼容规则，不能据此说它已联合优化若干3D姿态的H。经济动机明确是避免重造、重装fixture。 |
| **Rick Wagner, Giuseppe Castanotto, Ken Goldberg.** *FixtureNet: Interactive Computer Aided Design via the WWW*, IJHCS 46(6), 773–788, 1997；相关AI in Design 1996展示。 | 已读[正式全文](https://goldberg.berkeley.edu/pubs/FixtureNet-Journal-Paper.pdf)，算法、质量、系统和讨论。另[作者HTML版](https://goldberg.berkeley.edu/fixturing/fixture_background.htm)。 | 将modular fixture枚举作为网页CAD服务，显示候选/反力；是部署与交互贡献，不是新的共享模块算法。正文明确只是feasibility study，不能把网页演示当工业实证。 |
| **Charles Anderson, Yan Zhuang, Ken Goldberg.** *FixtureNet II: Interactive Redesign and Force Visualization on the Web*, ASME DETC97/DFM-4353, 1997。 | 已读[作者全文PDF](https://people.wou.edu/~andersc/pubs/detc97.pdf)，pp4–8；[HTML](https://people.wou.edu/~andersc/pubs/detc-html/paper.html)。 | 鼠标设外力并解非负接触反力；固定两条接触边、修改第三条时，在移动物体坐标系导出第三定位点的椭圆contact locus，实时检查是否仍可三点定位。**复用原3 locators，夹紧器可能换位置。** 未做固定共形3D模块多pose。FixtureNet III是后续软件/文档，未找到独立同名正式论文。 |
| **Jingliang Chen, Ken Goldberg, Mark H. Overmars, Dan Halperin, Karl F. Böhringer, Yan Zhuang.** *Shape Tolerance in Feeding and Fixturing*, WAFR 1998；*Computing Tolerance Parameters for Fixturing and Feeding*, Assembly Automation 22(2), 163–172, 2002。 | 已读[2002作者技术报告全文](https://webdoc.sub.gwdg.de/ebook/serien/ah/UU-CS/2002-006.pdf)，fixture模型、tolerance算法及讨论；[正式元数据](https://doi.org/10.1108/01445150210423206)。 | 凸多边形+直角定位支架；定义全类形状均成功的公差域，并给快速参数计算/成员检验。复用容差与可制造性长期已有；我们的共形小面积contact反而更需说明对打印/配准误差敏感度。 |
| **Isam N. Tahhan, Yan Zhuang, Karl F. Böhringer, Kris S. J. Pister, Ken Goldberg.** *MEMS Fixtures for Handling and Assembly of Microparts*, SPIE 1999, 129–139。 | 已读[合作者全文](https://labs.ece.uw.edu/mems/publications/1999/conferences/spie-mm-tahhan-99.pdf)，模型、制作、结果/限制。 | 微型阵列fixture cell、振动随机送入、主动闭合，计算稳定/强度/摩擦，制造原型。论文记录电短路/表面缺陷等问题，不能当成熟并行装配实证。与多pose共享H间接相关，提醒装入机制的可制作性。 |
| **Jae-Sook Cheong, Ken Goldberg, Mark H. Overmars, A. Frank van der Stappen.** *Fixturing Hinged Polygons*, ICRA 2002, 876–881；加入Elon Rimon扩展 *Immobilizing Hinged Polygons*, IJCGA 17(1), 2007。 | ICRA元数据来自[大学库](https://research-portal.uu.nl/en/publications/fixturing-hinged-polygons/)；已读[作者16页较早手稿](https://archive.dimacs.rutgers.edu/Workshops/CompAided/slides/vanderstappen.pdf)的定义、构造和结论。期刊最终全文未独立核查；root 已通过 Crossref 核对正式元数据为 17(1), 45–69, February 2007，[DOI](https://doi.org/10.1142/S0218195907002240)。 | 给定铰接多边形链的单一placement，研究无摩擦点接触数与抗小扰动immobility；不是多pose共享fixture，但说明最少contacts与稳健性并非新目标。大学库2004-005编号对应下载库出现错文，已排除该下载，以实际作者手稿为准。 |
| **K. Gopalakrishnan, Ken Goldberg.** *Gripping Parts at Concave Vertices*, ICRA 2002, 1590–1596。 | 已读[作者全文](https://goldberg.berkeley.edu/pubs/gopal-icra02.pdf)，2D/3D模型、算法与讨论。 | 两个圆柱/点jaw在凹角内收或外扩，利用多法向形成约束；small footprint便于access/insertion。是unilateral fixture的直接理论前驱，故纳入，不把全部一般gripper工作混入。 |
| **K. Gopalakrishnan, Matthew Zaluzec, Rama Koganti, Patricia Deneszczuk, Ken Goldberg.** ICRA 2003 *Unilateral Fixturing of Sheet-Metal Parts Using Modular Jaws with Plane-Cone Contacts*；Gopalakrishnan, Goldberg, Gary M. Bone, Matthew J. Zaluzec, Rama Koganti, Rich Pearson, Patricia A. Deneszczuk，*Unilateral Fixtures for Sheet-Metal Parts With Holes*, T-ASE 1(2), 110–120, 2004。 | 已读[2004正式全文](https://goldberg.berkeley.edu/pubs/Unilateral-Fixtures-T-ASE-Oct-2004.pdf)，§III–V、VII–VIII。 | 双主jaw利用孔凹角/圆锥槽固定；机构限制在一侧为焊接/检查留空间。§V输入各mesh node的力/力矩，用FEM算变形，迭代在最大违规位移处添secondary contact且避stay-out。实物测重复定位与actuation次序；正式装载建模仍列未来工作。我们的“缺啥补哪里”与“接触 vs 作业空间”也非全新概念。 |
| **K. Gopalakrishnan, Ken Goldberg.** ICRA 2004 *D-Space and Deform Closure: A Framework for Holding Deformable Parts*；WAFR 2004 *Computing Deform Closure Grasps*；IJRR 24(11), 899–910, 2005 *D-Space and Deform Closure Grasps of Deformable Parts*。 | 已读[2005作者全文](https://goldberg.berkeley.edu/pubs/D-Space-IJRR-Nov-2005.pdf)框架、定理、双jaw优化、结论；未分别逐页读两会议版。 | 线弹性三角mesh与刚性finger实体，在高维形变空间用释放所需正功定义deform closure；证明参考系不变性，优化双jaw距离、能量稳定与塑性破坏余量。不是当前刚体H问题；但“释放难度/能量”和物理保持之间已有直接理论。 |
| **Mike Tao Zhang, Ken Goldberg.** *Fixture-Based Industrial Robot Calibration for Silicon Wafer Handling*, Industrial Robot 32(1), 43–48, 2005。 | 已读[出版社摘要/元数据](https://doi.org/10.1108/01439910510573282)；发现多作者相关预印本但本地获取失败，不混淆作者表/模型版本。 | fixture定义关键标定点，末端更换后重教一个点并补偿其他点，推导容差要求并做实体例子。它有明确省重新示教/停机的因果链；我们的每pose卸装不能借用这种免重新定位承诺。 |
| **John D. Schulman, Ken Goldberg, Pieter Abbeel.** *Grasping and Fixturing as Submodular Coverage Problems*, ISRR 2011；书章正式出版2016，pp571–583（Crossref已核）。 | 已读[作者16页全文](https://goldberg.berkeley.edu/pubs/SchulmanGoldbergAbbeel_ISRR2011.pdf)，§3–7与结论；书章DOI [10.1007/978-3-319-29363-9_32](https://doi.org/10.1007/978-3-319-29363-9_32)。 | 给定有限候选接触点及各自固定wrench set，选至多K点；用有界总接触力/每点接触力下的抵抗wrench质量，支持函数+SATURATE，LP松弛+B&B。§3依据含重力/表面力的task set定义椭球norm。它不优化连续面积、contact实体、骨架或轨迹。固定K启发实现与放宽基数的理论保证要分开；不可说普通greedy本身保证全局最优。 |

## 未纳入核心但需要划清的边界

- Goldberg CV另列早期 *Automatic Design and Assembly of Fixtures using Modular Components*（Goldberg、Bekey、King、Requicha，NSF grantees项记录不完整）、1995 *Mechanical Design Evaluation Via the Internet*，以及教材的modular fixturing内容。已识别，正文/完整元数据未获，不能冒充读过。
- 两项fixture设计专利 US 5,546,314 (1996)、US 5,856,924 (1999)，Brost/Goldberg/Canny/Wallack。是专利不是新增论文；本次未逐项权利要求审查。
- Zhang–Goldberg 的gripper jaw/trapezoidal modules/shape tolerance/part alignment、2017 parallel-jaw tips、2016/2018 energy-bounded caging、2020 6DFC 和Minimal Work等是相邻接触/夹爪文献；除了上面直接fixture理论前驱未逐篇纳入核心。不能因此断言它们没有相关结果。
- Brost–Peters 1998 *Automatic Design of 3-D Fixtures and Assembly Pallets* **没有Goldberg署名**，但与本项目非常近，不应因为作者筛选而忽略。其[出版社摘要](https://journals.sagepub.com/doi/10.1177/027836499801701201)已明确task constraints、shape variation、loading和economical production；比较整个领域新颖性时必须进一步读。

## 对我们解释的批判，以及可以保留的动机

**最强反方提问：为什么不为每个pose独立打印一个fixture？** 现在换pose仍要抓回物体、取H、换B、重装H，因此没有自动减少装夹次数。若一体打印独立fixture已经便宜，新增分体连接、间隙和装配反而有代价。“同一个东西用两次”本身不证明值得。

可以保留的工程动机是：**在多个离散工位/姿态之间，复用物体适配层，把场景变化留给接口简单的基座。** 这不要求对象一直附着H，也不应宣传一次装夹。其净收益需要实测：制造数量/时间、后处理/试配/装配、接口代价、作业遮挡、可承载集合和稳健性。动机本身不必从未有人提过；图形学算法贡献应在自动发现这样的可共享几何。

需要特别比较“共用几个heads、骨架每pose各自做”。若代价主要来自共形头而骨架便宜，那么锁定共同骨架可能只缩小可行集。只有固定head相对位置和骨架带来额外制造/装配/验证收益，整个H共享才有根据。

数学上，若H相对对象的接触几何、摩擦和力限保持不变，且忽略B及结构变形，每pose世界wrench集合W_k经正确的wrench对偶坐标变换拉回对象坐标后，共享H只需抵抗其并集（符号统一后看反力能力包含关系）。这是一项推导，不是引用别人论文：**不能仅靠把同一力学测试循环K次就宣称新的multi-pose搜索原理。** 需要展示真正不等价的共享几何约束：同一H在每个场景均能装入、避开各自工具区、具有可实现的B接口/结构。

即便工作区也只是对象表面固定不接触集合，其多pose约束可合并成对象空间禁区并集，仍需诚实呈现。复杂性必须来自实际被保留的场景关系，不能靠描述添出来。

## 给root的推荐结论

可以这样讲：**“我们研究怎样把多个工况所需的专用支撑，合成为少量可制造、可装卸的物体适配几何，同时保留每个工况的操作空间和承载能力。”** 当前固定一个H是受限版本，不是宣称所有pose都该共享一个H。

关键实验：同等成功/载荷/工具空间/装卸要求下，比较独立fixture、共享heads但独立frame、统一wrench集+单场景设计、我们共享H联合设计。用真实案例证明共享是净收益而非只把蓝色涂成一样；报告共享失败或不划算的例子。算法上比较Goldberg式有限contact-set方法时，清楚说明新优化变量和不再满足的独立接触/固定候选假设。

与另两agent讨论后共识：经济motivation无需新颖，但必须成立；不能因旧复用动机存在就否定该应用。真正薄弱的是尚未证明为什么固定完整H比共享heads更值得，以及joint形状/路径搜索比合并载荷后单场景设计更必要。

## root的目录补查补充

- 最新作者目录也检查了；2025 *Automating Multi-Turn Cable Routing on the NIST Fixture Board with a Bi-Manual Robot and Caging Grippers* 属于使用fixture的电缆布线，非实体fixture设计，本轮边界列出，不宣称全文已读。
- **Mike Tao Zhang, Ken Goldberg. Calibration of Wafer Handling Robots: A Fixturing Approach**, CASE 2007, 255–260，DOI [10.1109/COASE.2007.4341769](https://doi.org/10.1109/COASE.2007.4341769)。root已核元数据/摘要，未获全文；须与2005 journal分列（摘要分别强调deterministic/statistical tolerance），不凭同作者/近题名当完全同版。
- SATURATE补充精度：每个方向上的support函数贡献是submodular；最坏方向的min本身不能直接当submodular，论文采用截断目标的SATURATE结构。

- Yu–Goldberg 装载文另经作者/大学渠道、OpenAlex、Semantic Scholar 补检，仍无可获取公开全文。1995工作另见早期题名 *Loading Planar Fixtures in the Presence of Uncertainty*，暂不计为独立新论文。
