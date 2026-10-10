# 工件不着地时的力与力矩

![物体不着地的联合平衡](airborne_equations.png)

这里的不着地指**工件本身没有地面接触，由夹具承载**；夹具仍可接地，工件静止，重力保留。沿用现有无摩擦、单边、无反力上限的工件—支撑接触模型。力矩以工件质心 $c$ 为原点，$r_{\rm push}=q-c$、$r_{\rm supp}=p-c$。

需求仍是同一对六维量：

\[
(F_D,\tau_D)=\left(mg\hat z-F_{\rm push},\;-(q-c)\times F_{\rm push}\right).
\]

物体着地时，支撑反力与工件自身地面反力**共同**满足这个需求。不着地时，工件地面反力为零，必须由同一组支撑接触反力独自满足两条平衡式：

\[
\int_{A_{\rm supp,obj}}F_{\rm supp}\,dA=F_D,\qquad
\int_{A_{\rm supp,obj}}(p-c)\times F_{\rm supp}\,dA=\tau_D.
\]

因此没有一个事先确定的“着地时从需求中扣除的地面力”：地面反力与支撑反力是联合求解的未知量。移除地面接触改变可用反力锥，不能通过删除重力来表示悬空。

重力作用于质心，绕质心的重力力矩为零。没有加工力时，支撑需要提供 $(mg\hat z,0)$；沿质心向下施加 $0.5mg$ 时，需要提供 $(1.5mg\hat z,0)$。加工力偏离质心时，所需力矩由真实力臂与力的叉乘计算。沿用 $\|F_{\rm push}\|\leq0.5mg$ 时，支撑总竖直反力至少为 $0.5mg$，原“整体不上抬”必要条件自动成立，仍不能代替支撑接地、抗倾覆、摩擦或强度验收。

工件和所有载荷作用点一起竖直平移 $\Delta c$，原朝向和加工力保持相同时，$q-c$ 不变，绕质心的需求逐行不变。绕地面世界原点 $O$ 的需求为

\[
\tau_O=\tau_D+c\times F_D,\qquad
\Delta\tau_O=\Delta c\times F_D.
\]

因此离地高度会影响支撑的地面力矩及所需压力中心。新 preprocessing 默认让工件最低点离地 **10 mm**，在 `samples.npz` 中同时保存 `need_wrench`（绕质心）与 `need_wrench_world_origin`（绕地面原点）。`floor_contact.npz` 的压力中心是系统**所需**地面承载位置，不是悬空工件重新获得了地面接触，也不是夹具已通过验收。

15°、30°、60° 均为加工力相对工作面内法向的**圆锥半角**；全开角分别为30°、60°、120°。加工力大小仍在 $[0,0.5mg]$ 内，与角度及是否着地分开设置。所有状态保留同一工作面和物体自遮挡检查。

图采用 [现有两条公式](two_equations.py) 的符号与版式。只生成这一张新图：

```sh
MPLCONFIGDIR=/tmp/cadgrasp-airborne-mpl PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python slides/obj_supp/airborne_equations.py
```
