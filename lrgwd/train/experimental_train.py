import tensorflow as tf

data_paths = [
    "../../../runs/massive_year_one/split_full/train_tensors.csv", 
    "../../../runs/massive_year_two/split_full/train_tensors.csv"
]

batches = tf.data.experimental.make_csv_dataset(
    file_pattern=data_paths, 
    batch_size=100,
    header=False, 
)