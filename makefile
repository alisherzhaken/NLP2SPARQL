train:
	python run.py \
        --do_train \
        --do_eval \
        --model_type bert \
        --model_architecture bert2bert \
        --encoder_model_name_or_path bert-base-cased \
        --decoder_model_name_or_path razent/spbert-mlm-zero \
        --source sparql \
        --target en \
        --train_filename ./NLP2SPARQL_datasets/LCQUAD/train \
        --dev_filename ./NLP2SPARQL_datasets/LCQUAD/dev \
        --output_dir ./ \
        --max_source_length 32 \
        --weight_decay 0.01 \
        --max_target_length 64 \
        --beam_size 10 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --save_inverval 15 \
        --num_train_epochs 5 
eval:
	python run.py \
        --do_test \
        --model_type bert \
        --model_architecture bert2bert \
        --encoder_model_name_or_path bert-base-cased \
        --decoder_model_name_or_path razent/spbert-mlm-zero \
        --source sparql \
        --target en \
        --load_model_path ./checkpoint-best-bleu/pytorch_model.bin \
        --test_filename ./NLP2SPARQL_datasets/LCQUAD/test \
        --dev_filename ./NLP2SPARQL_datasets/LCQUAD/dev \
        --output_dir ./ \
        --max_source_length 64 \
        --weight_decay 0.01 \
        --max_target_length 128 \
        --beam_size 10 \
        --eval_batch_size 32 \


