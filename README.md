# Nash Q-Learning With Inferring Causal Signal Temporal Logic: A Study of Competitive Multi-Agent Systems

Code for the paper by Hadi Partovi Aria, Zhe Xu. IEEE Control Systems Letters, 2025.

NASTL-CIRL combines Nash Q-learning with causal STL inference. Agents mine causal STL formulas that explain which action sequences succeed and use their robustness as dense reward while learning Nash-consistent policies.

## Files

`factory_assembly_three_agents.py` is Case Study I (factory assembly with three robots competing for keys, doors and parts). It includes the causal STL inference and compares NASTL-CIRL with the baseline and counterfactual agents.

`factory_assembly_two_agents.py` is the two-robot version of the same environment.

`zero_sum_pacman.py` is Case Study II, the zero-sum Pacman game on a 6x6 grid with power bases and 10% slip.

## Running

```bash
pip install -r requirements.txt
python factory_assembly_three_agents.py
python zero_sum_pacman.py
```

## Citation

```bibtex
  @article{aria2025nash,
    title = {Nash Q-Learning With Inferring Causal Signal Temporal Logic: A Study of Competitive Multi-Agent Systems},
    author = {Hadi Partovi Aria and Zhe Xu},
    journal = {IEEE Control Systems Letters},
volume = {9},
pages = {2867--2872},
    year = {2025}
  }
```

## License

MIT, see `LICENSE`.
