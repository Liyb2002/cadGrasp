# Step 1：采样近似六维需求

当前默认每个物体 **32,768 条可达需求**。
从连续的物理施力域采样，再映射为完整六维力—力矩需求；本步不做覆盖积分。
所有候选应复用同一份样本，以便比较贡献。

| 物体 | 采样需求 | 数量与定义 | 公式图 | 三个说明例子 |
|---|---|---|---|---|
| A1-f | [samples.json](../output/A1-f/pose_1/step_1_needs/samples.json) | [domain.json](../output/A1-f/pose_1/step_1_needs/domain.json) | [domain.png](../output/A1-f/pose_1/step_1_needs/domain.png) | [proof.png](../output/A1-f/pose_1/step_1_needs/proof.png) |
| B | [samples.json](../output/B/pose_1/step_1_needs/samples.json) | [domain.json](../output/B/pose_1/step_1_needs/domain.json) | [domain.png](../output/B/pose_1/step_1_needs/domain.png) | [proof.png](../output/B/pose_1/step_1_needs/proof.png) |
| C5 | [samples.json](../output/C5/pose_1/step_1_needs/samples.json) | [domain.json](../output/C5/pose_1/step_1_needs/domain.json) | [domain.png](../output/C5/pose_1/step_1_needs/domain.png) | [proof.png](../output/C5/pose_1/step_1_needs/proof.png) |

## 数量与精度

32,768 是搜索阶段的起点。快速试跑可用 8,192，最终候选可用独立种子的 131,072 条
样本复核。默认种子 `20260907`，使用 PCG64；同一种子增加数量保留原有样本前缀。
不同种子用于独立复核，不能把只在搜索样本上表现好的方案当成已验证的最终方案。

| 有效样本数 | 一个通过样本对应的分数增量（百分点） |
|---:|---:|
| 8,192 | 0.01221 |
| 32,768 | 0.00305 |
| 131,072 | 0.00076 |

这些是分数的步长，**不是误差界**。小贡献和接近的排名需要更多样本。
有限样本可能漏掉狭小的失败或成功区域；样本全部通过不证明整个连续需求域都可行。

## 采样分布与配对需求

1. 按真实工作三角面的面积选面，在面内按面积均匀抽取 `pt`。
2. 在该处内法向 **30°** 球冠内按立体角均匀抽方向：`cos(theta)` 均匀、`phi` 均匀。
3. 在 `[0,0.5mg]` 内均匀抽力度；这里沿用当前均匀力度权重。
4. 按完整物体网格检查工具射线，拒绝遮挡载荷，继续采样直到补足指定数量。
5. 每个载荷一起计算六个分量，不独立抽六个分量，不去重映射相同的需求。

世界坐标竖直向上为 `ẑ`，所有力矩关于工件质心 `c`：

```text
demand(F_push, pt) = (mg ẑ − F_push, −(pt − c) × F_push)
```

导出的力按体重 `mg` 归一化，力矩单位为 `mg·m`，位置单位为 m。
力矩使用真实施力点 `pt`；工具可达性检查的微小射线起点偏移不改变施力点。
每行是一个独立工况，不是把所有加工力同时施加。

这批样本近似的是物理载荷映射得到的六维需求集合及其载荷分布，
不是六个独立区间，也不是在六维体积内均匀撒点。

## samples.json 接口

按列保存，所有数组的第 `i` 行对应同一次施力：

| 字段 | 含义 |
|---|---|
| `count`、`seed` | 有效需求数量与种子 |
| `pt_m` | 真实施力位置，N×3 |
| `force_push_mg` | 完整加工力，N×3 |
| `need_wrench` | `[Fx,Fy,Fz,tau_x,tau_y,tau_z]`，N×6 |
| `work_face_index` | 在 `needs.json` 工作面列表中的索引 |
| `parameters` | N×5，依次为 `u,v,theta_rad,phi_rad,magnitude_mg` |
| `weight_per_sample` | 每条有效样本的权重 `1/N` |
| `provenance` | 物理定义文件哈希、代码哈希、库版本 |

采样目标是可达载荷域中的 **表面积 × 立体角 × 均匀力度**。当前 Step 3.1 使用的估计为：

```text
covered ≈ 满足联合六维平衡的样本数 / count
```

它估计的是上述载荷分布的满足比例，不直接等于几何工作区的面积比例。
`F_push=0` 是连续力度区间的零测度端点，不另行注入或单独计分。

**Step 3.1 已接入 samples.json，逐轮计算候选与已固定接触的联合评分；Step 3.2 选择，Step 3.3 优化，Step 3 调度并验证连续域。**
结果记录样本文件哈希并保存逐样本通过矩阵；旧积分实现与排名已替换。
多个支撑的共同贡献须合并反力列重新判定，不能仅合并单块通过掩码。

## 文件与运行

```text
output/<A1-f|B|C5>/step_1_needs/
  samples.json               搜索输入：采样六维需求
  needs.json                 连续物理定义和完整几何，仍供 Step 2 读取
  domain.json + domain.png   样本数量、文件引用和需求公式
  examples.json              三个说明例子，施力点字段为 pt_m，力度为 0.125、0.25、0.5mg
  proof.json + proof.png     这三个例子的配对需求图
```

已有 `needs.json` 在仅更换样本数量或种子时保留原字节和哈希，避免使未改变的 Step 2
接触几何失效；采样前核对原始模型和姿态来源。首次缺少该文件时才从 setup 快照建立它。
其中 `measure_for_later_integration` 是早期物理导出的历史说明，当前采样测度以
`samples.json` 的 `sampling_distribution` 和 `domain.json` 的 `measure` 为准。

在仓库根目录、`cadgrasp` 环境运行：

```sh
python slides/baseline_algo/step1/needs.py A1-f B C5 --count 32768 --seed 20260907
python -m unittest discover -s slides/baseline_algo/step1 -p 'test_*.py' -v
```

`needs.py` 是完整入口，一次刷新三个物体的样本、说明例子、domain JSON、公式图与示例图。
更换数量或种子时，所有相关数量与哈希随之更新；省略物体名称则默认处理全部三个物体。
库函数 `needs.build()` 只导出数据，适合后续搜索程序直接调用；`domain.py` 与
`draw_proof.py` 仍可分别执行，用于仅重绘已有数据。
公式图沿用 [demand](../../obj_supp/demand/demand_equation.png) 的两行公式及 `pt`、`F_push`、`r_push` 定义。
`draw_proof.py` 仍可独立重绘说明例子，无需积分或模拟器。

读取计算输入：

```python
import json
import numpy as np
from pathlib import Path

folder = Path('slides/baseline_algo/output/B/pose_1/step_1_needs')
samples = json.loads((folder / 'samples.json').read_text())
needs = np.asarray(samples['need_wrench'])   # N×6
weight = samples['weight_per_sample']
```

验证覆盖分布、遮挡拒绝后补足数量、相同种子的前缀复用、六维分量的独立力学展开、
以及三个物体导出数据的几何、可达性、单位和文件引用。
