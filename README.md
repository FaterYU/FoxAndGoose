# FoxAndGoose

## Dependence

- Python 3.8
- Pytorch 2.4.1
- gymnasium 0.28.1

### Install from pip

```bash
pip install -r requirement.txt
```

## Install from conda

```bash
conda env create -f requirement.yaml
```

## File Structure

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

- `train.py` train the model
- `test.py` test the model
- `TestFoxAndGeese.py` Global test
- `Team05.py` Team05's Interface
- `env` Environment
- `models` PPO model
- `weights` Model weights

**Important**: The `weights` `env` `models` folder should be in the same directory as `Team05.py`. They are necessary for the model to work.

## Usage

### Train

Parameters can be modified:

- save_threshold: Save the model each threshold rounds
- num_episodes: Number of episodes
- gamma: Discount factor
- actor_lr: Actor learning rate
- critic_lr: Critic learning rate
- epsilon: PPO epsilon
- eps: PPO clip parameter
- lmdba: PPO GAE parameter
- epochs: PPO epochs
- n_states: Number of states
- n_hidden: Number of hidden layers
- device: Device, `cuda` or `cpu`

```bash
python train.py
```

Result will be saved in `runs` folder.

### Test

```bash
python test.py
```

### TestFoxAndGeese

You should use file `TestFoxAndGoose.py` to import your class and to test that your code works with the testing framework. Replace module name Team00 in `module=__import__("Team00")` with your own file name without `.py` extension.

```bash
python TestFoxAndGeese.py
```

## Special Considerations

- Describing the Action Space for Fox and Goose

  The action space must be finite, and it is necessary to consider how to describe the action space for both characters to ensure it is limited, complete, and non-redundant.
  For the fox, its actions include a total of eight directions, namely up, down, left, right, and four diagonal directions. In the action steps where the fox does not eat the goose, it can be described as "the fox moves n=1 step in direction v"; in the action steps where the fox eats the goose, it can be described as "the fox moves n≠1 steps in direction v and eats the goose". On a 7x7 board, the value of n is limited to: 1, 2, 4, 6, a total of four types. In addition, it is necessary to consider the situation where the fox may not have a next step after eating the goose, meaning the fox cannot move again, and an additional static action is required, which only takes effect after the fox has eaten the goose. In summary, the maximum action space for the fox is 8 * 4 + 1 = 33.
  For the goose, its actions include a total of eight directions, and it is necessary to describe the actions of multiple geese simultaneously. Considering that only one goose can move per round, it can be described as "the k-th goose moves one step in direction v". Similarly, the maximum action space for the goose is 15 * 8.

- Ensuring Disorder Among Goose Individuals

  Maintaining the goose position storage sequence in ascending order ensures consistency during training and inference.

- Avoiding Illegal Actions*

  Based on the board movement rules and the action spaces of the fox and the goose, mask the action spaces for the fox and the goose in each round to ensure that all available actions are legal.

- Describing Board Movement Rules

  Each action is described as direction + steps, enumerating all possible actions for the 33 positions on the board, with the enumeration process including optimization operations.

- PPO Algorithm Input State

  The board state is transformed into a 33-bit encoding, including three states: fox, goose, and empty positions, to accelerate network training and inference.

- Designing the Reward Function

  Consider current dangers and potential risks.
