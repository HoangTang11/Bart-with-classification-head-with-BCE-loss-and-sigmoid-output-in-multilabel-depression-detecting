# Bart-with-classification-head-with-BCE-loss-and-sigmoid-output-in-multilabel-depression-detecting
Đây là đồ án ngành thực hiện việc cải tiến kết quả của bài báo depressionEmo.

Đoạn mã để chạy code train lẫn test:
!python seq2seq.py --mode "train" \
  --model_name "facebook/bart-base" \
  --train_path "dataset/train.json" \
  --val_path "dataset/val.json" \
  --test_path "dataset/test.json" \
  --epochs 5 \
  --batch_size 4 \
  --max_source_length 256
