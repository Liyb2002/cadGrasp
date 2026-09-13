# Contact area: B / pose 2

The current [area figure](area_B.png) uses the shared slide renderer and the exact
B/pose_2 workpiece, working surface and final contact patches. Run
`python slides/render.py --only area` (in the cadgrasp environment).

| Configuration | Sampled joint coverage |
|---|---:|
| One small contact at C139 | 3.573608% |
| Three times that contact area (the final C139 patch) | 45.358276% |
| Final C139 + C024 patches | 86.547852% |
| Final C139 + C024 + C011 patches | 100.000000% |

These are **32,768-sample percentages**, not the historical deterministic
integrals below. All rows use the same B/pose_2 paired loads, with process
magnitudes in [0, 0.5] mg and the 30-degree reachable inward cone. The force and
moment equations share one nonnegative reaction allocation. The Step3 shared
no-uplift constraint and original-floor four-ray friction model (mu=64) are
included. This comparison does not certify final structure or insertion.

The small patch is clipped to the actual connected non-work surface; its area is
one third of the saved C139 patch within 1e-5 relative error. The last two rows
add the remaining saved patches to the larger first patch. Numbered insets show
the actual contact surfaces from their outward side, with the floor omitted
and a fixed scale for each contact across rows; main views share the same camera. `area_B_pose2.json` records the measured ratio, per-row areas, counts and
independently replayed primal/dual checks; `area_B_pose2.npz` stores the masks.

The original A1/C5 figures and tip-1 experimental data remain historical records.
`area.py` defaults to this current presentation; `area.py --legacy ...` explicitly
runs the earlier search/integration workflow and may overwrite its old images.
The historical B numbers below do not describe the current `area_B.png`.

## Historical experiments (2026-09-06)

# Area：三个物体，每个四种支撑方案

2026-09-06。只保留三张正式图：[A1-f](area_A1-f.png)、[B 兔子](area_B.png)、
[C5 椅子](area_C5.png)。每张图四行：一块高价值小支撑、同中心接触面积三倍、
两块支撑、三块支撑。第二行只扩大第一块；第三行从第一行加第二块，第四行再加第三块。
第三、四行均逐块编号并展示实际接触区近景。图中不标大小、面积、面积倍数或方案文字；
只保留辨认接触区的编号和 **Covered** 百分比，不标面积或样本分数。
部分覆盖使用确定性面积积分；全覆盖时积分比值为 1，并保留连续全域证书。

## 覆盖大小与全覆盖证据

百分比的对象是连续的“位置＋方向”施力条件，按工作区面积和方向球面面积加权，
再限制为工具可达的条件。同一个需求的力与力矩仍用同一组非负接触力求解。
图上每行只给一个 covered 百分比，不再用 `Not fully covered` 或统计置信区间代替大小。

| 物体 | 一块小支撑 | 接触面积 ×3 | 两块支撑 | 三块支撑 |
|---|---:|---:|---:|---:|
| [A1-f](area_A1-f.png) | 0.004–0.032% | 0.049–0.109% | 17.43–18.25% | 99.62–99.75% |
| [B](area_B.png) | 2.23–2.56% | 40.94–41.99% | 100% | 100% |
| [C5](area_C5.png) | 3.40–3.80% | 13.45–14.20% | 100% | 100% |

百分比由下面的确定性几何积分计算。小于 0.1% 时保留三位百分数小数，避免把正覆盖
舍入成 0%。数值积分误差与加密前后的差值保存在结果文件中，不作为图上的 confidence。
B/C5 后两行通过连续需求外包络验证，积分的分子和分母相等，所以直接得到 100%。
其余八行仍保留真实可达的失败需求和完整连续接触区的分离证据。

面积仍用于选择区域和保持第二行约三倍面积，但只保存在数值结果中，不作为图上标注。

上述结论限于当前三角网格、目标姿态及下述力学模型；不外推到未经误差包络的光滑 CAD
曲面。本页展示接触区域的具体例子，不承担“面积是否有用”的统一结论。

## 如何验证整个连续需求域

满加工力为 `K=0.5`，位置 `q` 遍历全部工作三角面，单位施力方向 `d` 遍历当地
内法向的 15° 球冠。要求：

```
D(q,d) = (e_z − 0.5 d, −0.5 (q−c) × d)
对所有允许且可达的 (q,d)，存在 λ(q,d) ≥ 0，使 A λ(q,d) = D(q,d)。
```

`continuous.py` 使用以下充分条件，而不是加密采样：

1. 对每个面，球冠的轴向分量在 `[cos 15°, 1]`，横向分量在半径 `sin 15°` 的圆盘内。
   用外切正八边形包住这个圆盘，再与轴向区间组成棱柱；对外包络加 `1e-10` 裕量。
   棱柱某些顶点长度大于 1，它们只是保守验证点，没有改变实际力的大小。
2. 取该工作三角面的三个顶点与棱柱的 16 个顶点生成需求。若
   `q=Σαᵢqᵢ`、`d=Σβⱼdⱼ`，则双线性关系给出
   `D(q,d)=ΣᵢⱼαᵢβⱼD(qᵢ,dⱼ)`；系数非负且总和为 1。
   支撑的可行力—力矩锥是凸集，因此这些包络顶点全部可行就包含整片连续需求。
3. 用实际接触面片中心找到非负接触力基，并保存每个包络顶点使用的基。
   通过逆矩阵范数与残差界检查真正的非负解，不把“小残差”直接当精确平衡。
   力矩除以物体最大尺寸作数值缩放；每个矩阵和需求元素额外允许 `1e-10` 误差。
   原面法向与细分面法向的差异也检查在此误差预算内。

全覆盖检查包含被物体遮挡的方向，所以对实际可达需求是保守的充分条件。
若包络失败，程序另外在真实球冠上解析寻找分离方向，将施力点移到实际工作面的
严格内部、方向移到球冠内部，并检查工具射线可达。最后以接触区**全部三角面顶点**
验证分离向量 `y`：每个供给列 `a` 满足 `y·a<0`，真实需求满足 `y·D>0`，且留有数值裕量。
位置依赖的仿射性使这个分离证据适用于整个连续接触区，包含任意非负法向压力分布，
不只是原面片中心。所有失败行都已保存这样的反例。

证书与反例可用 `continuous.py --check` 在不调用可行性求解器的情况下复核。
这是当前网格模型下、带上述浮点误差预算的连续域验证，不是对原始 CAD 曲面的形式化证明。
两颗球分别覆盖不能替代这项六维包含检查。

## 连续覆盖比例的确定性积分

令 `R(q,d)` 表示工具可达，`H(q,d)` 表示存在共同非负接触力满足前两个方程。
定义：

```
coverage = ∫工作区 ∫15°球冠 R(q,d) H(q,d) dΩ dA
           / ∫工作区 ∫15°球冠 R(q,d) dΩ dA
```

这是施力条件的加权占比，不是两颗球的覆盖面积之积，也不是六维空间体积。
力固定为 `K=0.5`，不加入此前已排除的零加工力。

`coverage.py` 和 `polygon_integral.cpp` 按以下结构直接计算积分：

1. **完整连续接触区转换为力—力矩锥的不等式。** 每个平面接触三角形的法向固定，
   力矩对接触位置仿射，所以其三个顶点代表整片区域。加入原地面反力后，用一个正截面
   把六维锥化为五维凸包。逐次加入违反当前凸包的真实接触顶点，直到全部顶点满足
   所有侧面，得到 `H D ≤ 0`。这一过程不使用随机需求或前一版 Monte Carlo 的样本。
2. **固定方向时，把位置积分做成多边形面积。** 对每个工作三角面、固定方向 `d`，
   `H D(q,d) ≤ 0` 对 `q` 是线性的。逐条裁切工作三角形，留下的多边形就是该方向的
   联合可行位置区域；用多边形面积公式积分，不取若干位置再数命中次数。
3. **几何处理工具遮挡。** 将可能挡住工具的物体三角面沿工具方向投影到工作面，
   从可行多边形及工作面中扣除这些投影的并集。先裁去工具起点后方的部分；重叠投影
   只扣一次。保留已有模型的 `1e-5 m` 射线起点偏移，施力点本身仍在真实工作面上。
4. **只剩方向上的二维积分。** 使用极角 `ψ`、方位角 `φ`，严格带上
   `dΩ = sin(ψ) dψ dφ`。Gauss–Kronrod 自适应求积细分数值误差较大的方向区域。
   先用 gk15 粗算，再用 gk21 和更紧容差复核；接近全覆盖时积分“未覆盖面积”，
   使误差控制集中到小的缺口。完整全覆盖由连续判据直接化简为积分比值 1。

初始极角／方位角分区固定，随后细分也完全确定，不使用随机种子、二项计数或置信水平。
一般部分比例属于数值积分结果：记录求积误差估计与不同规则加密前后的变化；这两者
不是形式化的积分误差上界，更不是统计置信区间。当前网格的完整覆盖证明仍单独保存。
求积接口见 [SciPy 的自适应 cubature 说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html)。

结果与旧搜索表不同，是因为现在积分了全部连续工作位置和方向，并使用整个接触
区域的供给锥；旧表只在 90 个位置上测试有限方向，以接触面片中心近似供给。
旧搜索数字保留原来的含义，不充当新积分结果。

## 原有限样本用于搜索和球面示意

搜索仍按工作区面积取 90 个位置，每处在球冠内随机取 24 个方向，种子为 1。
射线剔除遮挡后，A1-f/B/C5 分别留下 2160/2141/2150 条配对需求。
其历史四行通过数仍保存在结果文件：A1-f 为 2/2/280/2145，B 为 45/611/2141/2141，
C5 为 69/256/2150/2150。它们只用于候选评分与采样图示，不再作为连续覆盖数。
这些也是搜索使用的样本，并非独立测试集。

采样器为避免自交把点外移 `1e-5 m`；新的连续验证直接使用真实工作三角面的点，
仅在可达性射线起点上使用这个偏移。球面分格、平滑和浮雕仍只是显示操作。
这些旧样本不参与 covered 的积分，不能用其通过率充当连续覆盖值。

## 先算小区域价值，再比较扩大与新增

1. 在整个可接触表面按位置和法向生成 768 个分散候选；在高价值位置附近细化。
   每个候选都检验全部配对需求，以联合解数最高的合格单块起步，同分时取较小面积。
   若首轮没有正价值候选，先加密到 1536 个。
2. 初始半径 `r₀ = 0.03 × 物体最大尺寸`。候选面积至少 `0.35 πr₀²`，避免残留碎片
   夸大单位面积收益。区域按真实原网格共享边连通，限制在种子法向的 89° 内；
   不用空间近邻把薄壁正背两面误连起来。
3. 比较已有区域扩大到 `1.2、1.4、1.7、2 × r₀`，以及加入新的不相交小区域。
   每次选“新增联合完成数 / 新增面积”最高的动作。组合使用共同接触力重新求解，
   不把各块单独解集求并集；单独价值为零的块仍可有很高的组合增益。
4. 原价值搜索达到全采样可行时停止；当前展示按用户要求固定为 1、1、2、3 块。
   两块方案取搜索中首个两块状态；若搜索已有三块状态就直接采用。
   B/C5 两块已满足全部采样，第三块选剩余合格、不相交候选中独立价值最高的，
   同分取较小面积。这一行展示三块布置，不声称第三块增加了联合完成数。
   A1-f 三块完成 2145 / 2160，不再显示原来四块达到 100% 的状态。
   这是离散候选上的贪心搜索，不宣称全局最优。

每个候选及每一步备选动作都保存在 `area_<name>_search.json`，候选面片和逐需求
判定保存在 `area_<name>_candidates.npz`。用于快速评分的平面凸包缩减只保留原有
接触列；重复需求可以复用已找到的接触基，但每个新解仍检查非负性和残差。
正式四行另用全部接触列复核。

## 读图与模型

每行依次是物体的两个整体视角、逐块接触区近景、力球、力矩浮雕球、覆盖百分比。
绿色是工作区，橙色是接触区域；整体图和近景使用同一编号。四行保持相同姿态和采样。
第一块的近景相机与裁剪框按第二行最大区域固定，使第一、二行的面积增长可直接比较；
不同支撑的近景分别放大，只用编号与整体图对应。

近景从实际场景渲染，不移除或透视物体表面。A1-f 孔内第三块使用孔内空腔视角。
可见性使用面片到相机的有限线段，忽略相机后方的几何；每个近景可见其接触区域
至少 95% 的面积，并保存逐块可见比例。完整接触面片均参与联合求解。

力球画在 `−F_D`，力矩球画在 `+τ_D`。两球分别显示各自三行平衡式的判定：
橙色为通过，红色表示该方向格中存在失败样本。两球仍是采样示意，
都橙色也不能替代右侧的六维联合覆盖积分或连续全域证书。
浮雕高度是同一批需求在方向格中的最大力矩，线性归一化到 0.55，标尺圈间距按实际力矩设置。

沿用 [当前决定](../../problem_statement.md#当前决定与讨论记录)：`K=0.5`、只算满加工力，工件已在第一个
目标姿态，支撑终点完美贴合。接触力沿内法向、非负且无上限；包含工件原地面支点
的竖直反力。搜索阶段使用面片中心；连续比例使用完整接触三角面的供给锥，共同满足
[前两个方程](../equations_to_solve.md)。整体与地面的平衡、支撑实体与插入路径由后续页面处理。

需求从 90 个工作区点、每点 24 个方向开始，随机种子 1，方向锥半角 15°，
再做可达性筛选。A1-f/B/C5 分别保留 2160/2141/2150 条配对需求。
姿态准备与渲染使用临时对象副本；对象原文件和 setup 图片不被改写。

## 代码与重生成

- `area.py`：统一入口，读取当前需求、搜索、复核四种方案、生成三张图。
- `area_search.py`：候选评分及扩大／新增搜索。
- `continuous.py`：连续需求外包络验证、真实可达反例，以及证书复核。
- `coverage.py`：连续供给锥侧面、工作面裁切、方向自适应积分与加密复核。
- `polygon_integral.cpp`：多边形裁切、遮挡投影并集扣除和面积公式；按源文件哈希编译到临时目录。
- `patch.py`：实际接触网格与物体渲染；`drawing.py`：球面与四行排版。
- `views.py`：整体视角、逐块近景，以及有限视线的遮挡判断。
- `inputs.py`：隔离的姿态准备和需求采样。
- `verify_area.py`：独立代入力、力矩，检查非负系数、失败证据、面积和搜索决策。
- `test_area_search.py`：求解器和几何缩减与相机遮挡的四项回归。
- `test_continuous.py`：球冠极值、外包络、双线性插值与误差证书的四项回归。
- `test_coverage.py`：有解析答案的局部面积、球面测度积分、重叠遮挡与已知接触锥回归。
- `demand_<name>_tip1.npz`：需求输入及模型参数、对象文件哈希。
- `area_<name>_results.{json,npz}`：四行区域、实际接触列、逐需求共同解和失败证据。
- `area_<name>_continuous.npz`：通过方案的接触力基与包络顶点分配；反例保存在结果 JSON。
- `area_<name>_coverage.npz`：供给锥不等式、遮挡几何、确定性积分分区、误差估计与复核值。

```sh
python slides/obj_supp/area/area.py                 # 完整搜索、验证、生成三张图
python slides/obj_supp/area/area.py --from-search   # 复用已保存搜索，重新求解四行并画图
python slides/obj_supp/area/area.py --render-only   # 复用四行支撑，重新检查连续覆盖并画图
python slides/obj_supp/area/continuous.py          # 只更新连续覆盖证据
python slides/obj_supp/area/continuous.py --check  # 不求解，直接复核保存的连续证据
python slides/obj_supp/area/coverage.py            # 重新做确定性积分，更新结果但不画图
python slides/obj_supp/area/coverage.py --check    # 复核锥侧面、积分分区、几何求值与记录结果
python slides/obj_supp/area/verify_area.py
python -m unittest discover -s slides/obj_supp/area -p 'test_*.py'
```

命令可加 `A1-f`、`B` 或 `C5` 只生成指定物体。删除派生需求表后可从当前对象重建。
统一入口会按几何、需求、支撑结果及 Python/C++ 源码的哈希复用积分结果；失效时重新计算。
前一版统计抽样、confidence、二项区间代码与数据已被确定性积分替换。
旧图、旧单块排名图、重复生成器和旧审核目录已删除；当前结果不依赖旧文件。
