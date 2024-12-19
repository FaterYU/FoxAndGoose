# FoxAndGoose

## 依赖

- Python 3.8
- Pytorch 2.4.1
- gymnasium 0.28.1

### 从 pip 安装

```bash
pip install -r requirement.txt
```

## 从 conda 安装

```bash
conda env create -f requirement.yaml
```

## 文件结构

```yaml
FoxAndGoose
├── env
│   ├── __init__.py
│   ├── GridRule.py
│   └── FoxGooseEnv.py
├── models
│   ├── __init__.py
│   └── PPO.py
├── LICENSE
├── README.md
├── Team05.py
├── TestFoxAndGeese.py
├── requirement.txt
├── requirement.yaml
├── test.py
├── train.py
└── weights
    ├── config.txt
    ├── fox
    │   ├── actor.pth
    │   ├── critic.pth
    │   ├── loss.npy
    │   ├── loss_actor.png
    │   └── loss_critic.png
    └── goose
        ├── actor.pth
        ├── critic.pth
        ├── loss.npy
        ├── loss_actor.png
        └── loss_critic.png
```

- `train.py` 训练模型
- `test.py` 测试模型
- `TestFoxAndGeese.py` 全局对战
- `Team05.py` 队伍接口
- `env` 环境
- `models` 模型
- `weights` 模型权重

**重要**: `weights` `env` `models` 文件夹应与 `Team05.py` 在同一目录下。它们对模型工作是必需的。

## 使用

### 训练

可变参数:

- save_threshold：每达到阈值回合保存模型
- num_episodes：回合数
- gamma：折扣因子
- actor_lr：演员学习率
- critic_lr：评论家学习率
- epsilon：PPO epsilon
- eps：PPO clip参数
- lmdba：PPO GAE参数
- epochs：PPO epoch数
- n_states：状态数
- n_hidden：隐藏层数
- device：设备，`cuda` 或 `cpu`

```bash
python train.py
```

结果将保存在 `runs` 文件夹中。

### Test

```bash
python test.py
```

### TestFoxAndGeese

您应该使用文件 `TestFoxAndGoose.py` 导入您的类并测试您的代码是否与测试框架一起工作。将 `module=__import__("Team00")` 中的模块名 `Team00` 替换为您自己的文件名，不包括 `.py` 扩展名。

```bash
python TestFoxAndGeese.py
```

## Special Considerations

- 描述狐狸和鹅的 `action space`

    首先，action space 需要是有限的，需要考虑如何描述两个角色的动作空间才能保证动作空间有限、完整、无重复。
    对于狐狸，其行动包含了一共八个方向，即上下左右四个方向，以及四个斜方向。在不吃掉鹅的行动步骤中，可以表述为“狐狸朝 v 方向移动 n=1 步”；在吃掉鹅的行动步骤中，可以表述为“狐狸朝 v 方向移动 n≠1 步并吃掉鹅”。而在 7x7 的棋盘上，n的取值有且仅有：1, 2 共四种。此外还需考虑狐狸在吃掉鹅的后续行动中可能出现没有下一步的情况，即狐狸无法再次移动，需要额外加多一个静止的动作，仅在已经吃掉鹅的情况下才能生效。综上，狐狸的动作空间最大为 8 * 2 + 1 = 17。
    对于鹅，其行动包含了一共八个方向，且需要同时描述多个鹅的行动。考虑到每一轮次有且仅有一只鹅可以移动，因此可以描述为“第 k 只鹅向 v 方向移动一步”。同上，鹅的动作空间最大为 15 * 8。

- 保证鹅个体之间的**无序性**

    维护鹅的位置存储序列升序即可保证训练与推理时的一致。

- 避免非法行动

    根据棋盘行动规则与狐狸、鹅的动作空间，在每一轮次中分别对狐狸与鹅的动作空间进行遮罩，保证待选的行动均为合法行动。

- 描述棋盘行动规则

    将每个行动描述为，方向+步数，枚举棋盘 33 个位置的所有可能行动，枚举过程包含优化操作。

- PPO 算法的输入状态

    将棋盘状态转化为 33 位的编码，包含狐狸、鹅、空位三种状态，加速网络训练及推理。

- 如何设计奖励函数

    考虑当前危险与潜在风险
