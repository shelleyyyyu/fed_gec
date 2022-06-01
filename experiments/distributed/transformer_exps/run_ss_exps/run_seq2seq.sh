# sh run_seq2seq.sh FedOPT "niid_cluster_clients=100_alpha=0.1" 1e-5 0.1 50 1
FL_ALG=FedOPT
PARTITION_METHOD="natural_clients=4"
C_LR=1e-5
S_LR=0.1
MU=50
ROUND=10

LOG_FILE="fedavg_transformer_ss.log"
WORKER_NUM=4
CI=0

DATA_DIR=../fednlp_data/
DATA_NAME=nlpcc
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m experiments.distributed.transformer_exps.run_ss_exps.fedavg_main_ss \
  --gpu_mapping_file "./experiments/distributed/transformer_exps/run_ss_exps/gpu_mapping.yaml" \
  --gpu_mapping_key mapping_a100_1gpu \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition_data.h5" \
  --partition_method $PARTITION_METHOD \
  --fl_algorithm $FL_ALG \
  --model_type bart \
  --model_name facebook/bart-base \
  --do_lower_case True \
  --train_batch_size 2 \
  --eval_batch_size 1 \
  --max_seq_length 64 \
  --lr $C_LR \
  --server_lr $S_LR \
  --fedprox_mu $MU \
  --epochs 1 \
  --output_dir "./${FL_ALG}_${DATA_NAME}_output/" \
  --fp16
  2> ${LOG_FILE} &


# sh run_span_extraction.sh FedAvg "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 50

# sh run_span_extraction.sh FedProx "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 50

# sh run_span_extraction.sh FedOPT "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 50