#!/usr/bin/env bash

# ajuste se precisar
PYTHON=python3
CFG="config_files/sfanalysis.yaml"   # coloque aqui o caminho do YAML que o runner lê por padrão

# DATASETS=("Planetoid:Cora" "Amazon:Photo" "AttributedGraphDataset:BlogCatalog" "Planetoid:CiteSeer")
# GENERATORS=("GenCAT" "SkyMap" "UnAtt")
DATASETS=("Planetoid:Cora")
GENERATORS=("UnAtt")
VALUES=(0)  # n e m

for dataset in "${DATASETS[@]}"; do
  for gen in "${GENERATORS[@]}"; do
    for n in "${VALUES[@]}"; do
      # escreve o YAML para esta combinação (n = m)
      cat > "$CFG" <<EOF
sfanalysis:
  dataset_name: "${dataset}"
  mimic_data: true
  synthetic_gen: "${gen}"
  n: ${n}
  m: ${n}
  intra_cluster_edges: ${n}
EOF

      # # executa a análise
      # ${PYTHON} sfanalysis_runner.py --yaml_config_file $CFG
      for i in {1..5}; do
        echo "Execução $i"
        ${PYTHON} sfanalysis_runner.py --yaml_config_file $CFG
      done
    done
  done
done