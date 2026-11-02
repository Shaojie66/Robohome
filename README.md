# Robohome

家庭专属机器人管家。核心想法：现实世界缺少规范，通用机器人要为此付出大量训练成本；我们规范不了所有人，但能规范自己的家。把家"封装"好（布局、可操作部件、标准容器、物品台账、联网家电），交给机器人，就像扫地机首次建图一样，机器人不需要理解整个世界，只需要按函数调用的方式做家务。

当前阶段：**纯软件**。所有技能都是只改内存状态的模拟实现，家电也是模拟的，先把数据层、技能接口、规划、入库和调度跑通；之后换成 MuJoCo / Isaac Sim 仿真，硬件最后再定。

## 快速开始

```bash
pip install -e ".[dev]"

robohome validate                       # 校验样例厨房 data/kitchen_example.json
robohome run "把牛肉拿出来解冻"          # 端到端跑一条指令
robohome run "把牛肉拿出来解冻" --fault offline:microwave   # 注入故障：家电离线
robohome run "把牛肉拿出来解冻" --fault missing:box_f1      # 注入故障：台账和现实不一致
robohome skills                         # 列出技能清单（带参数 Schema，规划层用它约束大模型）
robohome intake "今天买了一斤五花肉、两盒酸奶和一袋苹果"   # 入库：识别并分配容器
pytest
```

用本地大模型规划和识别（默认 Ollama + qwen3:8b，不需要任何 API Key，也不需要额外的 Python 依赖）：

```bash
ollama pull qwen3:8b
robohome run "牛肉化一下，晚上要用" --planner local
robohome intake "刚从超市回来，买了排骨、虾仁和一箱牛奶" --recognizer local
robohome run "..." --planner local --model qwen3:14b --base-url http://192.168.1.10:11434/v1
```

任何兼容 OpenAI Chat Completions 接口的本地服务都能用（llama.cpp server、vLLM、LM Studio），用 `--base-url` 指过去即可；也可以设环境变量 `ROBOHOME_LLM_BASE_URL`、`ROBOHOME_LLM_MODEL`。

用 Claude 规划和识别（需要 `pip install -e ".[claude]"` 和 `ANTHROPIC_API_KEY`）：

```bash
robohome run "牛肉化一下，晚上要用" --planner claude
robohome intake "刚从超市回来，买了排骨、虾仁和一箱牛奶" --recognizer claude
robohome intake --recognizer claude --image receipt.jpg --save data/my_home.json
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
| `planner/` | 规划层：一句话 → 技能调用序列 | 规则规划器（离线）+ 大模型规划器 |
| `intake/` | 入库：识别新物品 → 分配容器 → 更新台账 | 关键词识别（离线）+ 大模型识别（文字或照片） |
| `llm/` | 大模型后端：本地模型（Ollama 等）或 Claude，接口一致 | 可用 |

每一步执行前依次过三道关：**安全规则 → 技能前置条件 → 执行**，任何一关失败立刻停下并回报，不猜、不重试。

### 大模型规划器怎么工作

1. 把技能清单（带参数 Schema）和当前世界状态交给大模型，用结构化输出约束它只能返回合法的技能调用；做不到时它会说明原因
2. 拿到计划先在世界副本上**预演**，安全规则和前置条件都会检查
3. 预演失败就把原因告诉大模型重新规划（默认最多 3 次）
4. 通过后交给调度器真正执行，执行时照样逐步检查，现实和台账不一致时失败即停

大模型只负责"想"，不直接动手。

### 入库怎么分配容器

温区必须对上；已经装着同类、还放得下的容器优先，尽量装满一个再开下一个；一个容器只装一个类别；放不下就如实报告，不硬塞。分配完直接写进台账，新物品马上能被规划器找到。

## 安全原则

- 机器人不进禁区（例如灶台前）
- 机器人不直接接触热源和刀具：家电运行时不开它的门、不从里面取放东西
- 进微波炉的容器必须标记为微波炉可用
- 手里拿着东西时不启动带危险的家电

机器人只负责"开启和传递"，加热、切割等工作由家电自己完成，就像调用一个封装好的函数。

## 路线

1. 数据层 + 技能接口 + 最小调度链路 ✓
2. 大模型规划层 + 入库模块 ✓
3. 调度器完善 + MVP 五个任务端到端，系统性故障测试
4. 仿真：把技能库换成 MuJoCo / Isaac Sim 实现，规划和调度代码不动
