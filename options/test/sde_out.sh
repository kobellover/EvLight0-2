export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/test/sde_out.yaml" \
  --log_dir="./log/test/sde_out/" \
  --alsologtostderr=True \
  --VISUALIZE=True
