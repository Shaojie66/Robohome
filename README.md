# Robohome

家庭专属机器人管家。核心想法：现实世界缺少规范，通用机器人要为此付出大量训练成本；我们规范不了所有人，但能规范自己的家。把家"封装"好（布局、可操作部件、标准容器、物品台账、联网家电），交给机器人，就像扫地机首次建图一样，机器人不需要理解整个世界，只需要按函数调用的方式做家务。

当前阶段：**纯软件**。所有技能都是只改内存状态的模拟实现，家电也是模拟的，先把数据层、技能接口和调度跑通；之后换成 MuJoCo / Isaac Sim 仿真，硬件最后再定。

## 快速开始

```bash
pip install -e ".[dev]"

robohome validate                       # 校验样例厨房 data/kitchen_example.json
robohome run "把牛肉拿出来解冻"          # 端到端跑一条指令
robohome run "把牛肉拿出来解冻" --fault offline:microwave   # 注入故障：家电离线
robohome run "把牛肉拿出来解冻" --fault missing:box_f1      # 注入故障：台账和现实不一致
robohome skills                         # 列出技能清单（之后交给大模型当工具）
pytest
```

运行示例：

```
任务：把牛肉拿出来解冻
  ✓ 1. navigate(waypoint="fridge_front")  —  到达 fridge_front
  ✓ 2. open(fixture="freezer_door")  —  打开冷冻室门
  ✓ 3. pick(container="box_f1")  —  从 freezer 取出冷冻盒 1 号
  ✓ 4. close(fixture="freezer_door")  —  关上冷冻室门
  ✓ 5. navigate(waypoint="counter_front")  —  到达 counter_front
  ✓ 6. open(fixture="microwave_door")  —  打开微波炉门
  ✓ 7. place(container="box_f1", location="microwave")  —  把冷冻盒 1 号放进 microwave
  ✓ 8. close(fixture="microwave_door")  —  关上微波炉门
  ✓ 9. appliance(appliance="microwave", action="defrost", params={"minutes": 8})  —  调用微波炉 defrost {'minutes': 8} → running=True
完成：共 9 步
```

## 模块

| 目录 | 做什么 | 现状 |
| --- | --- | --- |
| `spec/` | 家用规范数据层：房间与点位、可操作部件、存放区、标准容器、物品台账、家电 | 数据结构 + 引用校验 + 厨房样例 |
| `skills/` | 技能库：navigate / open / close / pick / place / appliance，每个技能 = 前置条件 + 执行 | 模拟实现 |
| `appliances/` | 家电适配：统一的 `call(action, params)` 接口 | 模拟家电；之后接 Home Assistant / 米家 |
| `scheduler/` | 调度器：任务队列、安全规则、前置条件检查、失败即停、回报 | 可用 |
| `planner/` | 规划层：一句话 → 技能调用序列 | 规则规划器；下一步接大模型 |
| `intake/` | 入库：识别新物品 → 分配容器 → 更新台账 | 下一步 |

每一步执行前依次过三道关：**安全规则 → 技能前置条件 → 执行**，任何一关失败立刻停下并回报，不猜、不重试。

## 安全原则

- 机器人不进禁区（例如灶台前）
- 机器人不直接接触热源和刀具：家电运行时不开它的门、不从里面取放东西
- 进微波炉的容器必须标记为微波炉可用
- 手里拿着东西时不启动带危险的家电

机器人只负责"开启和传递"，加热、切割等工作由家电自己完成，就像调用一个封装好的函数。

## 路线

1. 数据层 + 技能接口 + 最小调度链路（本阶段）
2. 大模型规划层 + 入库模块
3. 调度器完善 + MVP 五个任务端到端，系统性故障测试
4. 仿真：把技能库换成 MuJoCo / Isaac Sim 实现，规划和调度代码不动
