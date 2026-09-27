# 多姿态共享支撑：新颖性冲突核查

2026-09-26。用户要求独立 agent 检查先例；以下包含主 agent 核查和独立检索记录。定向检索，不是系统综述；不修改当前研究决定或算法。

## 本轮判断

后续讨论补充：用户确认将**指定任务载荷、三维可装卸实体、多姿态复用**共同作为贡献主线，简明表述与三篇论文的区别已写入 [研究主线的贡献目标](multipose_rigid_fixture_design.md#2026-09-26用户确认的简明贡献与论文区别)。下面对“共享几何”这一宽泛主张的先例核查，不能代替对完整力学与三维设计问题的比较。Jiang 2023 §IV 虽允许外部 wrench，实际使用缺乏具体任务载荷信息时的正负单位力矩指标；我们的目标是覆盖工作区域、方向及幅值范围定义的配对六维需求。该差异和三维接触／实体设计应明确进入论文定位，具体方法与收益仍待验证。

“跨任务共同优化接触、摆放和共享形状”的宽泛主张已有强先例。当前一体、可重新摆放的落地被动支撑问题尚未在本轮检索中找到完整对应解法，但额外约束的组合不自动构成贡献。

最接近的新来源是 Jiang, Doshi, Gondhalekar, Rodriguez, *Parallel-Jaw Gripper and Grasp Co-Optimization for Sets of Planar Objects*, IROS 2023，DOI [10.1109/IROS55552.2023.10342241](https://doi.org/10.1109/IROS55552.2023.10342241)，[全文](https://arxiv.org/html/2310.18425v1)，[MIT 元数据](https://dspace.mit.edu/entities/publication/bbe31beb-a91e-4179-8c2e-8c3a76587a73)。主 agent 核对 §§III、V、VI、VII、IX：共同工具坐标、逐抓取配置、共享曲面、几何兼容及特征复用已有具体方法。其接触数及面/夹爪归属是输入。特别是 §VII-B 的跨任务折中例子，已覆盖上一轮建议的抽象动机故事。

保留为待验证研究问题：自动选择工件/地面接触角色、支撑摆放与完整材料连接，在多任务扫掠和承载限制下生成共享实体；需要证明具体算法比已有共同设计的合理扩展更有效。不能宽泛宣称共同坐标、共享几何、失败后重选接触或单任务让步本身新颖。

新增近作仅核对摘要：Omaisan & Mohamed, *Robot Aware Computational Design of Object Specific Passive Grippers for Additive Manufacturing*, arXiv, 2026-09-03，[原始记录](https://arxiv.org/abs/2609.03761)。摘要是逐对象被动抓具流程，尚不能据此判断全文是否存在更深的方法重叠。

下面保留独立 agent 的原始阅读记录；其中已说明全文与摘要证据的区别。GOFD 的回选应准确理解为设计拒绝后的候选回退，不能改写成已证明的自动冲突证据反馈。

---

# Independent targeted novelty-conflict check

Date: 2026-09-26. Read literature-review SKILL.md and project `codes/research_notes/multipose_rigid_fixture_design.md`. This is a targeted review, not exhaustive or systematic screening. No repository files changed.

## Main assessment

The abstract claim “discover cross-task compatible contacts, placements, and shared geometry” is already occupied by multi-object gripper co-design. Parent independently read the strongest hit, Jiang et al. 2023. My independent search supports this conclusion and finds older and newer related design pipelines. I did not find a paper solving the exact combination of a single rigid, freely reoriented ground-supported fixture, object/floor role switching, three-body equilibrium, complete connected material, and prescribed-task insertion/tool-clearance constraints. This absence in a bounded search is not evidence of firstness. Adding all constraints also does not automatically provide a research contribution.

## Strong original sources independently read

### Honarpardaz, Ölvander, Tarkian. Fast finger design automation for industrial robots. Robotics and Autonomous Systems 113 (2019), 120–131.
DOI https://doi.org/10.1016/j.robot.2018.12.011
Publisher https://www.sciencedirect.com/science/article/abs/pii/S0921889018304123
Self-archived full manuscript https://www.diva-portal.org/smash/get/diva2:1280983/FULLTEXT01.pdf

Read method §§4.1–4.4; closest passage §4.3, PDF pp.10–11 counting cover page. It stores candidate grasp sets; generates contact pads; ranks workpiece contact areas by concavity; successively customizes shared tips for multiple workpieces; then generates finger bodies and bases. It checks the completed fingers against the workpieces, presents collision-free candidates for manual inspection, and selects another grasp candidate if the user rejects the design. The text does not establish a certificate-driven automatic response to collision failures. The accessible manuscript has the working title “Agile Finger Design Automation...” whereas its repository cover and publisher identify the final title above. Do not confuse versions.

Conflict: multifunction rigid contact geometry, whole finger construction, cross-object collision checks, and returning to grasp choices after user rejection are existing concrete procedures. Difference: sequential customization and candidate fallback, actuated gripper with permanent fingertip/body/base roles; no free ground-support placement or role switching. “Geometric failure feeds back to contacts” is too broad a novel-method claim; richer conflict certificates and their measured benefit would need to be demonstrated.

### Hota et al. Automated Grasp Planning and Finger Design Space Search Using Multiple Grasp Quality Measures. Robotics 13(5), 74 (2024).
DOI https://doi.org/10.3390/robotics13050074
Institutional full text https://imec-publications.be/server/api/core/bitstreams/da9d0caf-b11a-4f2f-a71c-f8eae1304482/content
Metadata https://imec-publications.be/entities/publication/bd929bc9-f6d4-40e9-98ee-0c26ef691c28

Downloaded and read original §§2,3,4.6,5,7.3–7.4. pp.6–7: rigid three-contact parallel-jaw design, shared distance d_f. pp.14–16, Eq.22/Figs.9–13: score each common design by average best-grasp quality over objects; search poses, extract contact points in gripper coordinates, check contact feasibility and unwanted finger-body contact, evaluate force closure/quality. pp.28–29 explicitly restrict method to two grasp parameters and one shared shape parameter, sampled exhaustively. It produces parametric finger CAD and physical experiments.

Conflict: nested search over shared hardware and per-object contacts/poses is established. Difference: tiny fixed parameterization and topology, top grasps, active jaw motion, no free connected-solid optimization or floor-role switching. Less direct than Jiang 2023 but useful to prevent overstating broad generality.

### Fu, Chen, Su, Fu. Pose-Inspired Shape Synthesis and Functional Hybrid. IEEE TVCG 23(12) (2017), 2574–2585.
DOI https://doi.org/10.1109/TVCG.2017.2739159
Author https://johnfu1988.github.io/publication/tvcg17/index.html
Original PDF https://johnfu1988.github.io/pdf/tvcg17.pdf

Downloaded and read method/end of §4, §5, experiments §6, limitations. §4/5, manuscript p.7: choose human-affordance-related components from existing shapes, connect their relationship graphs; extend designs to one operator with several poses or several operators. §6/Fig.11: multi-pose/multi-operator composites. End limitations: users handle agent poses and remove some conflicting parts; ergonomic/structural functions are future work.

Conflict: broad fixed-geometry multifunction object synthesis is not new. Difference: human-affordance/database-part synthesis, not cross-task contact force/motion/material co-design. Do not characterize it as physical proof of a reoriented rigid fixture, or as demonstrating that exact same material patch switches floor/object roles.

## Historical and adjacent leads, with weaker access

- Balan & Bone, Automated Gripper Jaw Design and Grasp Planning for Sets of 3D Objects, Journal of Robotic Systems 20(3),147–162 (2003), DOI https://doi.org/10.1002/rob.10076. Original-author institution abstract https://experts.mcmaster.ca/scholarly-works/128199 confirms common jaw design and grasps for polyhedral objects, three cylindrical fingers, force-closure/jamming tests. Publisher issue verifies original venue https://onlinelibrary.wiley.com/toc/10974563/2003/20/3 . Some modern databases relabel the journal Journal of Field Robotics; prefer original issue. Full text inaccessible, so no detailed method inference.
- Honarpardaz et al., Generic Automated Multi-function Finger Design (2016), DOI https://doi.org/10.1088/1757-899X/157/1/012015. Institutional PDF indexed at https://liu.diva-portal.org/smash/get/diva2:1063076/FULLTEXT01.pdf but direct access timed out/returned HTML. Search indexing exposes method snippets, but use 2019 verified text for technical claims.
- Velasco & Newman, Computer-assisted gripper and fixture customization using rapid-prototyping technology, ICRA 1998, DOI https://doi.org/10.1109/ROBOT.1998.681393. 2019 original related-work section describes earlier superposed-target subtraction into shared finger blanks. A 1997 four-author preprint with similar title is indexed at https://citeseerx.ist.psu.edu/document?doi=1c8a8a3273c07dad1a6b826cb4b9aa5aa5342253&repid=rep1&type=pdf . Do not conflate those versions. Full original ICRA methods not inspected.
- Newman et al., Design Lessons for Building Agile Manufacturing Systems, IEEE TRA 16(3),228–238 (June 2000), DOI https://doi.org/10.1109/70.850641. Author-uploaded copy has §II-B “Agile Grippers and Fixtures,” pp230–231; search-exposed original text describes virtual-EDM subtraction of several parts into one finger blank and multifunction prototypes. Bibliographic search-result date 2001 is misleading; original first page says June 2000. Not needed as primary closest comparator.
- Kong & Ceglarek, Fixture workspace synthesis for reconfigurable assembly using procrustes-based pairwise configuration optimization, JMS25(1),25–38 (2006), DOI https://doi.org/10.1016/S0278-6125(06)80030-0. Publisher abstract https://www.sciencedirect.com/science/article/pii/S0278612506800300 : aligns multiple known locating layouts for a family of parts, reconfigurable fixture workspace. No full text accessed; relevant to alignment/commonality but cannot assert detailed overlap.
- He et al., Computational Design of Body-Supporting Assemblies, CGF44(7),2025 DOI https://doi.org/10.1111/cgf.70237; publisher https://diglib.eg.org/items/03b1abdd-2d96-418d-a13b-c66932d4acaa . Abstract and first page show single input human posture, optimized support pieces then procedural connectors. Motivation adjacent; not read in depth because not a multi-pose shared-rigid-body result.

## Claims to withdraw / retain conditionally

Withdraw firstness for: multiple tasks sharing fixed geometry; jointly choosing shared shape and per-task contacts/poses; expressing constraints in common tool coordinates; sharing contact features; selecting a worse individual grasp to satisfy another task; generic completed-design rejection followed by a new grasp candidate.

Conditional research target: discover object/floor contact-role assignments and placements of a single freestanding solid, while jointly maintaining three-body equilibrium, common material connectivity, finite insertion sweeps, and task access, with an algorithm exploiting these couplings and evidence stronger than existing gripper co-design adapted to 3D. This is a candidate question, not a confirmed gap or contribution.

For baselines: Jiang-style shared-shape/per-task-contact co-design is now essential conceptually. Add sequential multi-function customization / independent designs aligned, merged and repaired. Distinguish collision-aware contact fallback from certificate-guided coupling; only claim the latter if formalized and ablated.

## Search audit and limits

Used general web search to locate original papers in author pages, arXiv, publisher sites (IEEE/ACM/Wiley/Elsevier/MDPI/Eurographics), institutional repositories (DiVA, IMEC, VUB, Warwick), and DBLP metadata. ResearchGate/CiteSeer were discovery routes; final high-confidence method claims rely on original full-text documents. No subscription database search, no exhaustive patents, no complete Scopus/Scholar citation export. No date filter. Followed backward references from Hota2024 to Honarpardaz2016/2019, Balan2003, Jiang2023, Velasco1998; forward search from passive-grippers and multifunction design surfaced Hota2024 and later neighboring papers. A September2026 arXiv passive-gripper item surfaced but was not needed for overlap judgment and not read.

Representative exact queries: automatic design reusable fixture multiple workpieces topology optimization contact placement; multi purpose fixture design rigid multiple poses passive gripper computational design; multi functional shape optimization contact support fixture orientation; “fixture synthesis” “multiple”; “multi-part” “fixture” synthesis; “multi-purpose fixtures” automatic design; “fixture” “multiple orientations” design optimization; “passive gripper” “multiple objects” design; “multi-object” “passive grippers” design; “gripper” “commonality” design; “gripper” “multiple objects” “shape optimization”; “Pose-Inspired Shape Synthesis” pdf; “Automated Grasp Planning and Finger Design Space Search” pdf; “Automated gripper jaw design and grasp planning for sets of 3D objects” pdf; “Fast Finger Design Automation for Industrial Robots” pdf; “Generic Automated Multi-function Finger Design”; “Fixture Workspace Synthesis” “Procrustes”.
