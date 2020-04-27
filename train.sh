root_dir='test01'
num_iterations=1000000
gin_file=./config,gin

python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn_rnn/gym/MaskedCartPole-v0/ \
  --num_iterations=${num_iterations} \
  --gin_file=${gin_file}