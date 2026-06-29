# Οδηγίες Εγκατάστασης και Εκτέλεσης

## 1. Clone repository

```bash
git clone https://github.com/PanagiotaGr/aegis-vio.git
cd aegis-vio
```

## 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 3. Synthetic simulation

```bash
python scripts/run_simulation.py --scenario mixed --output_dir results/simulation
```

Η simulation εκτελεί lightweight demo χωρίς να απαιτείται download EuRoC ή TUM-VI dataset.

## 4. EuRoC pipeline

```bash
python scripts/download_euroc.py --sequence MH_01_easy
python scripts/run_euroc.py --dataset_root ./datasets/MH_01_easy/mav0 --output_dir ./results/MH_01_easy
python scripts/evaluate_results.py --results_dir ./results/MH_01_easy --dataset_root ./datasets/MH_01_easy/mav0
```

## 5. Testing

```bash
pytest tests/ -v
```

## 6. ROS2 launch

```bash
ros2 launch aegis_vio aegis_vio.launch.py
```

Το ROS2 package βρίσκεται στο `ros2_ws/src/aegis_vio/` και περιλαμβάνει nodes, launch file, configuration και custom messages.
