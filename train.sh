root_dir='test01'
num_iterations=1000000
gin_file=./config.gin

python src/model.py \
  --root_dir=${root_dir} \
  --num_iterations=${num_iterations} \
  --gin_file=${gin_file}
