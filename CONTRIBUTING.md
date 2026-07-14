# Contributing

1. Create a feature branch from `develop`.
2. Keep ROS 2 nodes under `src/` and configuration under `config/`.
3. Store Isaac Sim assets under `assets/isaac_sim/` by robot or environment.
4. Do not commit generated ROS 2 build folders, logs, caches, or recordings.
5. Test Python syntax before opening a pull request:

```bash
python3 -m py_compile src/gpr/gpr_subsurface_node.py
```

6. Describe scene-coordinate changes clearly because ground-truth positions are environment-specific.
